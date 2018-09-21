/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University
* of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include <cstdio>
#include <cstdlib>
#include <unistd.h>

#include "include/orbslam/LocalMapping.h"
#include "include/orbslam/LoopClosing.h"
#include "include/orbslam/ORBmatcher.h"
#include "include/orbslam/Optimizer.h"

#include <mutex>

namespace SIVO {

LocalMapping::LocalMapping(Map *pMap, const float bMonocular)
    : mbMonocular(bMonocular),
      mbResetRequested(false),
      mbFinishRequested(false),
      mbFinished(true),
      mpMap(pMap),
      mbAbortBA(false),
      mbStopped(false),
      mbStopRequested(false),
      mbNotStop(false),
      mbAcceptKeyFrames(true) {}

void LocalMapping::SetLoopCloser(LoopClosing *pLoopCloser) {
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker) {
    mpTracker = pTracker;
}

void LocalMapping::Run() {
    mbFinished = false;

    while (true) {
        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue
        if (CheckNewKeyFrames()) {
            // BoW conversion and insertion in Map
            ProcessNewKeyFrame();

            // Check recent MapPoints
            MapPointCulling();

            // Triangulate new MapPoints
            CreateNewMapPoints();

            if(!CheckNewKeyFrames()) {
                // Find more matches in neighbor keyframes and fuse point duplications
                SearchInNeighbors();
            }

            mbAbortBA = false;

            if (!CheckNewKeyFrames() && !stopRequested()) {
                // Local BA
                if (mpMap->KeyFramesInMap() > 2)
                    Optimizer::LocalBundleAdjustment(
                      mpCurrentKeyFrame, &mbAbortBA, mpMap);

                // Check redundant local Keyframes
                KeyFrameCulling();
            }

            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        } else if (Stop()) {
            // Safe area to stop
            while (isStopped() && !CheckFinish()) {
                usleep(3000);
            }
            if (CheckFinish()) {
                break;
            }
        }

        ResetIfRequested();

        // Tracking will see that Local Mapping is now free
        SetAcceptKeyFrames(true);

        if (CheckFinish()) {
            break;
        }

        usleep(3000);
    }

    SetFinish();
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF) {
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA = true;
}

bool LocalMapping::CheckNewKeyFrames() {
    unique_lock<mutex> lock(mMutexNewKFs);
    return (!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame() {
    {
        std::unique_lock<std::mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    const std::vector<MapPoint *> vpMapPointMatches =
      mpCurrentKeyFrame->GetMapPointMatches();

    for (size_t i = 0; i < vpMapPointMatches.size(); i++) {
        MapPoint *pMP = vpMapPointMatches[i];
        if (pMP) {
            if (!pMP->isBad()) {
                if (!pMP->IsInKeyFrame(mpCurrentKeyFrame)) {
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    pMP->UpdateNormalAndDepth();
                    pMP->ComputeDistinctiveDescriptors();
                } else {
                    // this can only happen for new stereo points inserted
                    // by Tracking
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }

    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

void LocalMapping::MapPointCulling() {
    // Check Recent Added MapPoints
    auto lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if (mbMonocular) {
        nThObs = 2;
    } else {
        nThObs = 3;
    }

    const int cnThObs = nThObs;

    while (lit != mlpRecentAddedMapPoints.end()) {
        MapPoint *pMP = *lit;
        if (pMP->isBad()) {
            lit = mlpRecentAddedMapPoints.erase(lit);
        } else if (pMP->GetFoundRatio() < 0.25f) {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        } else if (((int) nCurrentKFid - (int) pMP->mnFirstKFid) >= 2 &&
                   pMP->Observations() <= cnThObs) {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        } else if (((int) nCurrentKFid - (int) pMP->mnFirstKFid) >= 3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        else {
            lit++;
        }
    }
}

void LocalMapping::CreateNewMapPoints() {
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;

    const std::vector<KeyFrame *> vpBestNKFs =
      mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    ORBmatcher matcher(0.6, false);

    // Extract relevant information from current keyframe.

    // Extract pose
    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3, 4, CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0, 3));
    tcw1.copyTo(Tcw1.col(3));
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    // Extract camera intrinsics information
    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;
    const float ratioFactor = 1.5f * mpCurrentKeyFrame->mfScaleFactor;

    int nnew = 0;

    // Search matches with epipolar restriction and triangulate
    for (size_t i = 0; i < vpBestNKFs.size(); i++) {
        if (i > 0 && CheckNewKeyFrames()) {
            return;
        }

        KeyFrame *pKF2 = vpBestNKFs[i];

        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        cv::Mat vBaseline = Ow2 - Ow1;
        const float baseline = static_cast<float>(cv::norm(vBaseline));

        if (!mbMonocular) {
            if (baseline < pKF2->mb)
                continue;
        } else {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline / medianDepthKF2;

            if (ratioBaselineDepth < 0.01)
                continue;
        }

        // Compute Fundamental Matrix
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);

        // Search matches that fulfill epipolar constraint
        std::vector<std::pair<size_t, size_t>> vMatchedIndices;
        matcher.SearchForTriangulation(
          mpCurrentKeyFrame, pKF2, F12, vMatchedIndices, false);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3, 4, CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0, 3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = static_cast<int>(vMatchedIndices.size());
        for (int ikp = 0; ikp < nmatches; ikp++) {
            const int &idx1 = static_cast<int>(vMatchedIndices[ikp].first);
            const int &idx2 = static_cast<int>(vMatchedIndices[ikp].second);

            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysSemantic[idx1];
            const float kp1_r = mpCurrentKeyFrame->mvRight[idx1];
            bool bStereo1 = kp1_r >= 0;

            const cv::KeyPoint &kp2 = pKF2->mvKeysSemantic[idx2];
            const float kp2_r = pKF2->mvRight[idx2];
            bool bStereo2 = kp2_r >= 0;

            // Check parallax between rays
            cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx1) * invfx1,
                           (kp1.pt.y - cy1) * invfy1,
                           1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx2) * invfx2,
                           (kp2.pt.y - cy2) * invfy2,
                           1.0);

            cv::Mat ray1 = Rwc1 * xn1;
            cv::Mat ray2 = Rwc2 * xn2;
            const float cosParallaxRays = static_cast<float>(
              ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2)));

            float cosParallaxStereo = cosParallaxRays + 1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if (bStereo1) {
                cosParallaxStereo1 =
                  cos(2 * atan2(mpCurrentKeyFrame->mb / 2,
                                mpCurrentKeyFrame->mvDepth[idx1]));
            } else if (bStereo2) {
                cosParallaxStereo2 =
                  cos(2 * atan2(pKF2->mb / 2, pKF2->mvDepth[idx2]));
            }

            cosParallaxStereo = min(cosParallaxStereo1, cosParallaxStereo2);

            cv::Mat wP;
            if (cosParallaxRays < cosParallaxStereo && cosParallaxRays > 0 &&
                (bStereo1 || bStereo2 || cosParallaxRays < 0.9998)) {
                // Linear Triangulation Method
                cv::Mat A(4, 4, CV_32F);
                A.row(0) = xn1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
                A.row(1) = xn1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
                A.row(2) = xn2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
                A.row(3) = xn2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

                cv::Mat w, u, vt;
                cv::SVD::compute(
                  A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

                wP = vt.row(3).t();

                if (wP.at<float>(3) == 0) {
                    continue;
                }

                // Euclidean coordinates
                wP = wP.rowRange(0, 3) / wP.at<float>(3);

            } else if (bStereo1 && cosParallaxStereo1 < cosParallaxStereo2) {
                wP = mpCurrentKeyFrame->UnprojectStereo(idx1);
            } else if (bStereo2 && cosParallaxStereo2 < cosParallaxStereo1) {
                wP = pKF2->UnprojectStereo(idx2);
            } else {
                // No stereo and very low parallax
                continue;
            }

            cv::Mat x3Dt = wP.t();

            // Check triangulation in front of cameras
            float z1 =
              static_cast<float>(Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2));
            if (z1 <= 0) {
                continue;
            }

            float z2 =
              static_cast<float>(Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2));
            if (z2 <= 0) {
                continue;
            }

            // Check reprojection error in first keyframe
            const float &sigmaSquare1 =
              mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 =
              static_cast<float>(Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0));
            const float y1 =
              static_cast<float>(Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1));
            const float invz1 = 1.0f / z1;

            if (!bStereo1) {
                float u1 = fx1 * x1 * invz1 + cx1;
                float v1 = fy1 * y1 * invz1 + cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1) {
                    continue;
                }
            } else {
                float u1 = fx1 * x1 * invz1 + cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf * invz1;
                float v1 = fy1 * y1 * invz1 + cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_r;
                if ((errX1 * errX1 + errY1 * errY1 + errX1_r * errX1_r) >
                    7.8 * sigmaSquare1) {
                    continue;
                }
            }

            // Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 =
              static_cast<float>(Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0));
            const float y2 =
              static_cast<float>(Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1));
            const float invz2 = 1.0f / z2;

            if (!bStereo2) {
                float u2 = fx2 * x2 * invz2 + cx2;
                float v2 = fy2 * y2 * invz2 + cy2;

                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;

                if ((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2) {
                    continue;
                }
            } else {
                float u2 = fx2 * x2 * invz2 + cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf * invz2;
                float v2 = fy2 * y2 * invz2 + cy2;

                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_r;

                if ((errX2 * errX2 + errY2 * errY2 + errX2_r * errX2_r) >
                    7.8 * sigmaSquare2) {
                    continue;
                }
            }

            // Check scale consistency
            cv::Mat normal1 = wP - Ow1;
            float dist1 = static_cast<float>(cv::norm(normal1));

            cv::Mat normal2 = wP - Ow2;
            float dist2 = static_cast<float>(cv::norm(normal2));

            if (dist1 == 0 || dist2 == 0) {
                continue;
            }

            const float ratioDist = dist2 / dist1;
            const float ratioOctave =
              mpCurrentKeyFrame->mvScaleFactors[kp1.octave] /
              pKF2->mvScaleFactors[kp2.octave];

            if (ratioDist * ratioFactor < ratioOctave ||
                ratioDist > ratioOctave * ratioFactor) {
                continue;
            }

            // Triangulation is successful, now verify semantics.
            Classes class_kp1 = CheckSemantics(mpCurrentKeyFrame, idx1, wP, true);
            Classes class_kp2 = CheckSemantics(pKF2, idx2, wP, false);

            if (class_kp1 == class_kp2 && class_kp1 != Classes::VOID) {
                auto *pMP = new MapPoint(wP, mpCurrentKeyFrame, mpMap);

                pMP->AddObservation(mpCurrentKeyFrame, idx1);
                pMP->AddObservation(pKF2, idx2);

                mpCurrentKeyFrame->AddMapPoint(pMP, idx1);
                pKF2->AddMapPoint(pMP, idx2);

                pMP->ComputeDistinctiveDescriptors();

                pMP->UpdateNormalAndDepth();

                mpMap->AddMapPoint(pMP);
                mlpRecentAddedMapPoints.push_back(pMP);

                nnew++;
            }
        }
    }
}

Classes LocalMapping::CheckSemantics(const KeyFrame *pKF,
                                  const int idx,
                                  const cv::Mat &wP, bool compute_information) {
    Classes detected_class = Classes::VOID;

    auto col = static_cast<int>(pKF->mvKeysSemantic.at(idx).pt.x);
    auto row = static_cast<int>(pKF->mvKeysSemantic.at(idx).pt.y);
    float z = pKF->mvDepth.at(idx);

    double confidence = pKF->mConfidence(row, col);
    double classification_entropy = pKF->mEntropy(row, col);
    detected_class = static_cast<Classes>(pKF->mClasses(row, col));

    // For the current frame, we need to verify the information content. For the previous (matched) feature, we just
    // want to ensure that the semantic class matches.
    // For the current frame, we only return the actual class if it matches the information criteria.
    if (!compute_information) {
        return detected_class;
    }

    // Several conditions must be met for map point insertion
    // 1. Ensure depth is > 0
    bool depth_criteria = (z > 0);

    // 2. Detected class must be a deemed static class.
    bool class_criteria = (detected_class <= Classes::TERRAIN);

    // 3. Confidence must be above the threshold value.
    bool confidence_criteria = (confidence >= pKF->mThConfidence);

    if (depth_criteria && class_criteria && confidence_criteria) {
        // Extract coordinates
        double wX = wP.at<float>(0, 0);
        double wY = wP.at<float>(1, 0);
        double wZ = wP.at<float>(2, 0);

        // Calculate stereo jacobian for this point.
        StereoProjectionPoseJacobianType stereo_jacobian =
          SIVO::computeStereoJacobianPose(
            pKF->fx, pKF->fy, pKF->mb, wX, wY, wZ);

        // Extract measurement noise for this point.
        // This is done in the same manner as in the optimization.
        double Sigma2 = pKF->mvLevelSigma2[pKF->mvKeysSemantic.at(idx).octave];
        Eigen::Matrix3d measurement_noise =
          Eigen::Matrix3d::Identity() * Sigma2;

        // Compute the full covariance for the state and this point.
        StateCovarianceType Sigmacw = pKF->GetCovariance();
        StereoCovarianceType stereo_covariance = SIVO::computeStereoCovariance(
          Sigmacw, stereo_jacobian, measurement_noise);

        // Extract the mutual information between the state and
        // point.
        double mutual_information =
          SIVO::computeStereoMutualInformation(stereo_covariance);

        // The entropy reduction is the joint entropy between this
        // added feature and the classification.
        double entropy_reduction = mutual_information - classification_entropy;

        if (entropy_reduction < pKF->mThEntropyReduction) {
            // Did not meet entropy criteria, return VOID class.
            detected_class = Classes::VOID;
        }
    } else {
        // Did not meet depth/confidence/class criteria, return void.
        detected_class = Classes::VOID;
    }

    return detected_class;
}

void LocalMapping::SearchInNeighbors() {
    // Retrieve neighbor keyframes
    int nn = 10;
    if (mbMonocular) {
        nn = 20;
    }

    const std::vector<KeyFrame *> vpBestNKFs =
      mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    std::vector<KeyFrame *> vpTargetKFs;

    for (auto vit = vpBestNKFs.begin(); vit != vpBestNKFs.end(); vit++) {
        KeyFrame *pKFi = *vit;

        if (pKFi->isBad() ||
            pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId) {
            continue;
        }

        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        // Extend to some second neighbors
        const vector<KeyFrame *> vpSecondNeighKFs =
          pKFi->GetBestCovisibilityKeyFrames(5);
        for (auto vit2 = vpSecondNeighKFs.begin();
             vit2 != vpSecondNeighKFs.end();
             vit2++) {
            KeyFrame *pKFi2 = *vit2;
            if (pKFi2->isBad() ||
                pKFi2->mnFuseTargetForKF == mpCurrentKeyFrame->mnId ||
                pKFi2->mnId == mpCurrentKeyFrame->mnId) {
                continue;
            }
            vpTargetKFs.push_back(pKFi2);
        }
    }

    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    vector<MapPoint *> vpMapPointMatches =
      mpCurrentKeyFrame->GetMapPointMatches();
    for (auto vit = vpTargetKFs.begin(); vit != vpTargetKFs.end(); vit++) {
        KeyFrame *pKFi = *vit;

        matcher.Fuse(pKFi, vpMapPointMatches);
    }

    // Search matches by projection from target KFs in current KF
    vector<MapPoint *> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size() * vpMapPointMatches.size());

    for (auto vitKF = vpTargetKFs.begin(); vitKF != vpTargetKFs.end();
         vitKF++) {
        KeyFrame *pKFi = *vitKF;

        vector<MapPoint *> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for (auto vitMP = vpMapPointsKFi.begin(); vitMP != vpMapPointsKFi.end();
             vitMP++) {
            MapPoint *pMP = *vitMP;
            if (!pMP) {
                continue;
            }
            if (pMP->isBad() ||
                pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId) {
                continue;
            }
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates);

    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for (size_t i = 0, iend = vpMapPointMatches.size(); i < iend; i++) {
        MapPoint *pMP = vpMapPointMatches[i];
        if (pMP) {
            if (!pMP->isBad()) {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
}

cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2) {
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w * R2w.t();
    cv::Mat t12 = -R1w * R2w.t() * t2w + t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    return K1.t().inv() * t12x * R12 * K2.inv();
}

void LocalMapping::RequestStop() {
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop() {
    unique_lock<mutex> lock(mMutexStop);

    if (mbStopRequested && !mbNotStop) {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped() {
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested() {
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release() {
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if (mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for (list<KeyFrame *>::iterator lit = mlNewKeyFrames.begin(),
                                    lend = mlNewKeyFrames.end();
         lit != lend;
         lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames() {
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag) {
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames = flag;
}

bool LocalMapping::SetNotStop(bool flag) {
    unique_lock<mutex> lock(mMutexStop);

    if (flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA() {
    mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling() {
    // Get keyframes visible from the covisibility graph.
    std::vector<KeyFrame *> vpLocalKeyFrames =
      mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    for (auto vit = vpLocalKeyFrames.begin(); vit != vpLocalKeyFrames.end();
         vit++) {
        KeyFrame *pKF = *vit;
        if (pKF->mnId == 0) {
            continue;
        }

        const vector<MapPoint *> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs = nObs;
        int nRedundantObservations = 0;
        int nMPs = 0;
        for (size_t i = 0; i < vpMapPoints.size(); i++) {
            MapPoint *pMP = vpMapPoints[i];
            if (pMP) {
                if (!pMP->isBad()) {
                    if (!mbMonocular) {
                        if (pKF->mvDepth[i] > pKF->mThDepth ||
                            pKF->mvDepth[i] < 0) {
                            continue;
                        }
                    }

                    nMPs++;
                    if (pMP->Observations() > thObs) {
                        const int &scaleLevel = pKF->mvKeysSemantic[i].octave;
                        const map<KeyFrame *, size_t> observations =
                          pMP->GetObservations();
                        int nObs = 0;

                        for (auto mit = observations.begin();
                             mit != observations.end();
                             mit++) {
                            KeyFrame *pKFi = mit->first;
                            if (pKFi == pKF) {
                                continue;
                            }

                            const int &scaleLeveli =
                              pKFi->mvKeysSemantic[mit->second].octave;

                            if (scaleLeveli <= scaleLevel + 1) {
                                nObs++;
                                if (nObs >= thObs) {
                                    break;
                                }
                            }
                        }
                        if (nObs >= thObs) {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }

        if (nRedundantObservations > 0.9 * nMPs)
            pKF->SetBadFlag();
    }
}

cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v) {
    return (cv::Mat_<float>(3, 3) << 0,
            -v.at<float>(2),
            v.at<float>(1),
            v.at<float>(2),
            0,
            -v.at<float>(0),
            -v.at<float>(1),
            v.at<float>(0),
            0);
}

void LocalMapping::RequestReset() {
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while (1) {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if (!mbResetRequested)
                break;
        }
        usleep(3000);
    }
}

void LocalMapping::ResetIfRequested() {
    unique_lock<mutex> lock(mMutexReset);
    if (mbResetRequested) {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested = false;
    }
}

void LocalMapping::RequestFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished() {
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

}  // namespace ORB_SLAM
