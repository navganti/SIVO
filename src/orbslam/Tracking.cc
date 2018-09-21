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
#include <iostream>
#include <mutex>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "sivo_helpers/sivo_helpers.hpp"
#include "include/orbslam/Converter.h"
#include "include/orbslam/FrameDrawer.h"
#include "include/orbslam/Map.h"
#include "include/orbslam/ORBmatcher.h"
#include "include/orbslam/Optimizer.h"
#include "include/orbslam/PnPsolver.h"
#include "include/orbslam/Tracking.h"

namespace SIVO {

Tracking::Tracking(System *pSys,
                   ORBVocabulary *pVoc,
                   FrameDrawer *pFrameDrawer,
                   MapDrawer *pMapDrawer,
                   Map *pMap,
                   KeyFrameDatabase *pKFDB,
                   BayesianSegNet *pBayesianSegNet,
                   const string &strSettingPath,
                   const int sensor)
    : mState(NO_IMAGES_YET),
      mSensor(sensor),
      mbOnlyTracking(false),
      mbVO(false),
      mpORBVocabulary(pVoc),
      mpKeyFrameDB(pKFDB),
      mpBayesianSegNet(pBayesianSegNet),
      mpSystem(pSys),
      mpViewer(nullptr),
      mpFrameDrawer(pFrameDrawer),
      mpMapDrawer(pMapDrawer),
      mpMap(pMap),
      mnLastRelocFrameId(0) {
    // Load camera parameters from settings file
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if (k3 != 0) {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if (fps == 0) {
        fps = 30;
    }

    // Max/Min Frames to insert keyframes and to check relocalization
    mMinFrames = 0;
    mMaxFrames = static_cast<int>(fps);

    std::cout << std::endl << "Camera Parameters: " << std::endl;
    std::cout << "- fx: " << fx << std::endl;
    std::cout << "- fy: " << fy << std::endl;
    std::cout << "- cx: " << cx << std::endl;
    std::cout << "- cy: " << cy << std::endl;
    std::cout << "- k1: " << DistCoef.at<float>(0) << std::endl;
    std::cout << "- k2: " << DistCoef.at<float>(1) << std::endl;
    if (DistCoef.rows == 5)
        std::cout << "- k3: " << DistCoef.at<float>(4) << std::endl;
    std::cout << "- p1: " << DistCoef.at<float>(2) << std::endl;
    std::cout << "- p2: " << DistCoef.at<float>(3) << std::endl;
    std::cout << "- fps: " << fps << std::endl;

    int nRGB = fSettings["Camera.RGB"];
    mbRGB = static_cast<bool>(nRGB);

    if (mbRGB) {
        std::cout << "- color order: RGB (ignored if grayscale)" << std::endl;
    } else {
        std::cout << "- color order: BGR (ignored if grayscale)" << std::endl;
    }

    // Load ORB parameters
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(
      nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    mpORBextractorRight = new ORBextractor(
      nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    std::cout << std::endl << "ORB Extractor Parameters: " << std::endl;
    std::cout << "- Number of Features: " << nFeatures << std::endl;
    std::cout << "- Scale Levels: " << nLevels << std::endl;
    std::cout << "- Scale Factor: " << fScaleFactor << std::endl;
    std::cout << "- Initial Fast Threshold: " << fIniThFAST << std::endl;
    std::cout << "- Minimum Fast Threshold: " << fMinThFAST << std::endl;

    mThDepth = mbf * static_cast<float>(fSettings["ThDepth"]) / fx;
    std::cout << std::endl
              << "Depth Threshold (Close/Far Points): " << mThDepth
              << std::endl;

    // Extract semantic segmentation heuristics
    mThConfidence = static_cast<float>(fSettings["ThConfidence"]);
    mThEntropyReduction = static_cast<float>(fSettings["ThEntropyReduction"]);

    if (mThConfidence <= 0 || mThConfidence >= 1) {
        std::cerr << "ERROR: Semantic segmentation confidence must be between "
                     "0 and 1 inclusive"
                  << std::endl;
        exit(-1);
    } else {
        std::cout << std::endl
                  << "Semantic confidence threshold: " << mThConfidence
                  << std::endl;
    }

    std::cout << std::endl
              << "Feature selection entropy reduction threshold: "
              << mThEntropyReduction << std::endl;
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper) {
    mpLocalMapper = pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing) {
    mpLoopClosing = pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer) {
    mpViewer = pViewer;
}

cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft,
                                  const cv::Mat &imRectRight,
                                  const double &timestamp) {
    mImColour = imRectLeft;
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if (mImGray.channels() == 3) {
        if (mbRGB) {
            cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_RGB2GRAY);
        } else {
            cvtColor(mImGray, mImGray, CV_BGR2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_BGR2GRAY);
        }
    } else if (mImGray.channels() == 4) {
        std::cerr
          << "ERROR: Image input is RGBD, only COLOUR (BGR) is supported."
          << std::endl;
        exit(-1);
    } else if (mImGray.channels() == 1) {
        std::cerr
          << "ERROR: Image input is GRAY, only COLOUR (BGR) is supported."
          << std::endl;
        exit(-1);
    }

    mCurrentFrame = Frame(mImGray,
                          mImColour,
                          imGrayRight,
                          timestamp,
                          mpORBextractorLeft,
                          mpORBextractorRight,
                          mpORBVocabulary,
                          mpBayesianSegNet,
                          mK,
                          mDistCoef,
                          mbf,
                          mThDepth,
                          mThConfidence,
                          mThEntropyReduction);

    mImSemantic = mCurrentFrame.getSegmentedImage();

    Track();

    return mCurrentFrame.mTcw.clone();
}

void Tracking::Track() {
    if (mState == NO_IMAGES_YET) {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState = mState;

    // Get Map Mutex -> Map cannot be changed
    std::unique_lock<std::mutex> lock(mpMap->mMutexMapUpdate);

    if (mState == NOT_INITIALIZED) {
        // Initialize system.
        StereoInitialization();

        mpFrameDrawer->Update(this);

        if (mState != OK) {
            return;
        }
    } else {
        // System is initialized. Track Frame.
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization
        // (if tracking is lost)
        if (!mbOnlyTracking) {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
            if (mState == OK) {
                // Local Mapping might have changed some MapPoints tracked in
                // last frame
                CheckReplacedInLastFrame();

                if (mVelocity.empty() ||
                    mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
                    bOK = TrackReferenceKeyFrame();
                } else {
                    bOK = TrackWithMotionModel();
                    if (!bOK) {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
            } else {
                bOK = Relocalization();
            }
        } else {
            // Localization Mode: Local Mapping is deactivated
            if (mState == LOST) {
                bOK = Relocalization();
            } else {
                if (!mbVO) {
                    // In last frame we tracked enough MapPoints in the map
                    if (!mVelocity.empty()) {
                        bOK = TrackWithMotionModel();
                    } else {
                        bOK = TrackReferenceKeyFrame();
                    }
                } else {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and
                    // one doing relocalization.
                    // If relocalization is successful we choose that solution,
                    // otherwise we retain the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    std::vector<MapPoint *> vpMPsMM;
                    std::vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if (!mVelocity.empty()) {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization();

                    if (bOKMM && !bOKReloc) {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if (mbVO) {
                            for (int i = 0; i < mCurrentFrame.numSemanticKeys;
                                 i++) {
                                if (mCurrentFrame.mvpMapPoints[i] &&
                                    !mCurrentFrame.mvbOutlier[i]) {
                                    mCurrentFrame.mvpMapPoints[i]
                                      ->IncreaseFound();
                                }
                            }
                        }
                    } else if (bOKReloc) {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching.
        // Track the local map.
        if (!mbOnlyTracking) {
            if (bOK) {
                bOK = TrackLocalMap();
            }
        } else {
            // mbVO true means that there are few matches to MapPoints in the
            // map. We cannot retrieve a local map and therefore we do not
            // perform TrackLocalMap(). Once the system relocalizes the camera
            // we will use the local map again.
            if (bOK && !mbVO) {
                bOK = TrackLocalMap();
            }
        }

        // Update state status
        if (bOK) {
            mState = OK;
        } else {
            mState = LOST;
        }

        // Update drawer
        mpFrameDrawer->Update(this);

        // If tracking is good, check if we insert a keyframe
        if (bOK) {
            // Update motion model
            if (!mLastFrame.mTcw.empty()) {
                cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                mLastFrame.GetRotationInverse().copyTo(
                  LastTwc.rowRange(0, 3).colRange(0, 3));
                mLastFrame.GetCameraCenter().copyTo(
                  LastTwc.rowRange(0, 3).col(3));
                mVelocity = mCurrentFrame.mTcw * LastTwc;
            } else {
                mVelocity = cv::Mat();
            }

            // Update current camera pose for visualization
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            for (int i = 0; i < mCurrentFrame.numSemanticKeys; i++) {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (pMP)
                    if (pMP->Observations() < 1) {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i] =
                          static_cast<MapPoint *>(nullptr);
                    }
            }

            // Delete temporal MapPoints
            for (auto lit = mlpTemporalPoints.begin();
                 lit != mlpTemporalPoints.end();
                 ++lit) {
                MapPoint *pMP = *lit;
                delete pMP;
            }

            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            if (NeedNewKeyFrame()) {
                CreateNewKeyFrame();
            }

            // We allow points with high innovation (considered outliers by the
            // Huber Function) to pass to the new keyframe, so that bundle
            // adjustment will finally decide if they are outliers or not.
            // We don't want next frame to estimate its position with those
            // points so we discard them in the frame.
            for (int i = 0; i < mCurrentFrame.numSemanticKeys; i++) {
                if (mCurrentFrame.mvpMapPoints[i] &&
                    mCurrentFrame.mvbOutlier[i]) {
                    mCurrentFrame.mvpMapPoints[i] =
                      static_cast<MapPoint *>(nullptr);
                }
            }
        }

        // Reset if the camera get lost soon after initialization
        if (mState == LOST) {
            if (mpMap->KeyFramesInMap() <= 5) {
                std::cout << "Track lost soon after initialization, "
                             "resetting..."
                          << std::endl;
                mpSystem->Reset();
                return;
            }
        }

        if (!mCurrentFrame.mpReferenceKF) {
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        }

        mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory
    // afterwards.
    if (!mCurrentFrame.mTcw.empty()) {
        // Extract relative transform
        cv::Mat Tcr =
          mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState == LOST);
    } else {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState == LOST);
    }
}

void Tracking::StereoInitialization() {
    if (mCurrentFrame.numSemanticKeys > 500) {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));

        // Set covariance. The first frame is known as identity, therefore set
        // with very high certainty.
        StateCovarianceType state_covariance =
          StateCovarianceType::Identity() * 0.000001;
        mCurrentFrame.SetCovariance(state_covariance);

        // Create KeyFrame
        auto *pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Select good points! Traverse through all ORB matches.
        for (unsigned long i = 0;
             i < static_cast<unsigned long>(mCurrentFrame.numSemanticKeys);
             ++i) {
            // Get keypoint coordinates
            auto col =
              static_cast<int>(mCurrentFrame.mvKeysSemantic.at(i).pt.x);
            auto row =
              static_cast<int>(mCurrentFrame.mvKeysSemantic.at(i).pt.y);
            float z = mCurrentFrame.mvDepth.at(i);

            // Several conditions must be met for the map point insertion
            // 1. Ensure depth is > 0.
            if (!(z > 0)) {
                continue;
            }

            double classification_entropy = mCurrentFrame.mEntropy(row, col);

            // Extract 3d point location through unprojection
            // This is the location of the point (p) with respect to the
            // world frame (c), expressed in the world frame.
            cv::Mat wP = mCurrentFrame.UnprojectStereo(static_cast<int>(i));

            // Extract coordinates.
            double wX = wP.at<float>(0, 0);
            double wY = wP.at<float>(1, 0);
            double wZ = wP.at<float>(2, 0);

            // Calculate stereo jacobian for this point.
            StereoProjectionPoseJacobianType stereo_jacobian =
              SIVO::computeStereoJacobianPose(mCurrentFrame.fx,
                                              mCurrentFrame.fy,
                                              mCurrentFrame.mb,
                                              wX,
                                              wY,
                                              wZ);

            // Extract measurement noise for this point.
            // This is done in the same manner as in the optimization.
            double Sigma2 =
              mCurrentFrame
                .mvLevelSigma2[mCurrentFrame.mvKeysSemantic.at(i).octave];
            Eigen::Matrix3d measurement_noise =
              Eigen::Matrix3d::Identity() * Sigma2;

            // Compute the full covariance for the state and this point.
            StereoCovarianceType stereo_covariance =
              SIVO::computeStereoCovariance(
                mCurrentFrame.mSigmacw, stereo_jacobian, measurement_noise);

            // Extract the mutual information between the state and
            // point.
            double mutual_information =
              SIVO::computeStereoMutualInformation(stereo_covariance);

            // The entropy reduction is the joint entropy between this
            // added feature and the classification.
            double entropy_reduction =
              mutual_information - classification_entropy;

            // We use 0 bits for the first frame, the rest can be at the
            // threshold
            if (entropy_reduction > 0) {
                // Create new map point
                auto *pNewMP = new MapPoint(wP, pKFini, mpMap);

                auto detected_class =
                  static_cast<Classes>(mCurrentFrame.mClasses(row, col));
                pNewMP->SetSemanticInfo(detected_class);

                if (!pNewMP->isBad()) {
                    pNewMP->AddObservation(pKFini, i);

                    // Add map point to Keyframe and map.
                    pKFini->AddMapPoint(pNewMP, i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints.at(i) = pNewMP;
                }
            }
        }

        std::cout << "New map created with " << mpMap->MapPointsInMap()
                  << " points" << std::endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        // Create reference to previous frame.
        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId = static_cast<unsigned int>(mCurrentFrame.mnId);
        mpLastKeyFrame = pKFini;

        // Maintain reference to current local map points and reference frame.
        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState = OK;
    }
}

void Tracking::CheckReplacedInLastFrame() {
    int replaced = 0;

    for (int i = 0; i < mLastFrame.numSemanticKeys; i++) {
        MapPoint *pMP = mLastFrame.mvpMapPoints[i];

        if (pMP) {
            MapPoint *pRep = pMP->GetReplaced();
            if (pRep) {
                mLastFrame.mvpMapPoints[i] = pRep;
                ++replaced;
            }
        }
    }
}

bool Tracking::TrackReferenceKeyFrame() {
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7, true);
    std::vector<MapPoint *> vpMapPointMatches;

    int nmatches =
      matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

    if (nmatches < 15) {
        return false;
    }

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);
    mCurrentFrame.SetCovariance(mLastFrame.mSigmacw);

    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.numSemanticKeys; i++) {
        if (mCurrentFrame.mvpMapPoints[i]) {
            if (mCurrentFrame.mvbOutlier[i]) {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i] =
                  static_cast<MapPoint *>(nullptr);
                mCurrentFrame.mvbOutlier[i] = false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0) {
                nmatchesMap++;
            }
        }
    }

    return nmatchesMap >= 10;
}

void Tracking::UpdateLastFrame() {
    // Update pose according to reference keyframe
    KeyFrame *pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr * pRef->GetPose());

    if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR ||
        !mbOnlyTracking) {
        return;
    }

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo
    // sensor
    std::vector<std::pair<float, int>> vDepthIdx;
    vDepthIdx.reserve(static_cast<unsigned long>(mLastFrame.numSemanticKeys));
    for (int i = 0; i < mLastFrame.numSemanticKeys; i++) {
        float z = mLastFrame.mvDepth[i];
        if (z > 0) {
            vDepthIdx.emplace_back(std::make_pair(z, i));
        }
    }

    if (vDepthIdx.empty()) {
        return;
    }

    std::sort(vDepthIdx.begin(), vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for (const auto &depth_idx : vDepthIdx) {
        int i = depth_idx.second;

        bool bCreateNew = false;

        MapPoint *pMP = mLastFrame.mvpMapPoints[i];
        if (!pMP) {
            bCreateNew = true;
        } else if (pMP->Observations() < 1) {
            bCreateNew = true;
        }

        if (bCreateNew) {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            auto *pNewMP = new MapPoint(x3D, mpMap, &mLastFrame, i);

            mLastFrame.mvpMapPoints[i] = pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        } else {
            nPoints++;
        }

        if (depth_idx.first > mThDepth && nPoints > 100) {
            break;
        }
    }
}

bool Tracking::TrackWithMotionModel() {
    ORBmatcher matcher(0.9, true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    // Update pose and covariance estimate.
    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

    // Convert mVelocity to Eigen matrix.
    Eigen::Matrix4d temp;
    cv::cv2eigen(mVelocity, temp);
    Eigen::Affine3d motion_model;
    motion_model.matrix() = temp.matrix();

    StateCovarianceType current_covariance =
      SIVO::updateStateCovarianceMotion(mLastFrame.mSigmacw, motion_model);
    mCurrentFrame.SetCovariance(current_covariance);

    std::fill(mCurrentFrame.mvpMapPoints.begin(),
              mCurrentFrame.mvpMapPoints.end(),
              static_cast<MapPoint *>(nullptr));

    // Project points seen in previous frame
    int th;
    if (mSensor != System::STEREO) {
        th = 15;
    } else {
        th = 7;
    }

    int nmatches = matcher.SearchByProjection(
      mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR);

    // If few matches, uses a wider window search
    if (nmatches < 20) {
        fill(mCurrentFrame.mvpMapPoints.begin(),
             mCurrentFrame.mvpMapPoints.end(),
             static_cast<MapPoint *>(nullptr));
        nmatches = matcher.SearchByProjection(
          mCurrentFrame, mLastFrame, 2 * th, mSensor == System::MONOCULAR);
    }

    if (nmatches < 20) {
        return false;
    }

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.numSemanticKeys; i++) {
        if (mCurrentFrame.mvpMapPoints[i]) {
            if (mCurrentFrame.mvbOutlier[i]) {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i] =
                  static_cast<MapPoint *>(nullptr);
                mCurrentFrame.mvbOutlier[i] = false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++;
        }
    }

    if (mbOnlyTracking) {
        mbVO = nmatchesMap < 10;
        return nmatches > 20;
    }

    return nmatchesMap >= 10;
}

bool Tracking::TrackLocalMap() {
    // We have an estimation of the camera pose and some map points tracked in
    // the frame.
    // We retrieve the local map and try to find matches to points in the local
    // map.

    UpdateLocalMap();

    SearchLocalPoints();

    // Optimize Pose
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for (int i = 0; i < mCurrentFrame.numSemanticKeys; i++) {
        if (mCurrentFrame.mvpMapPoints[i]) {
            if (!mCurrentFrame.mvbOutlier[i]) {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if (!mbOnlyTracking) {
                    if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                        mnMatchesInliers++;
                } else
                    mnMatchesInliers++;
            } else if (mSensor == System::STEREO)
                mCurrentFrame.mvpMapPoints[i] =
                  static_cast<MapPoint *>(nullptr);
        }
    }

    // Decide if the tracking was successful
    // More restrictive if there was a relocalization recently
    if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames &&
        mnMatchesInliers < 50) {
        return false;
    }

    if (mnMatchesInliers < 30) {
        return false;
    } else {
        return true;
    }
}

bool Tracking::NeedNewKeyFrame() {
    if (mbOnlyTracking) {
        return false;
    }

    // If Local Mapping is frozen by a Loop Closure do not insert keyframes
    if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()) {
        return false;
    }

    const auto nKFs = static_cast<int>(mpMap->KeyFramesInMap());

    // Do not insert keyframes if not enough frames have passed from last
    // relocalization
    if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames &&
        nKFs > mMaxFrames) {
        return false;
    }

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;

    if (nKFs <= 2) {
        nMinObs = 2;
    }

    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be
    // potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose = 0;

    for (int i = 0; i < mCurrentFrame.numSemanticKeys; i++) {
        if (mCurrentFrame.mvDepth[i] > 0 &&
            mCurrentFrame.mvDepth[i] < mThDepth) {
            if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i]) {
                nTrackedClose++;
            } else {
                nNonTrackedClose++;
            }
        }
    }

    // Modify nTrackedClose Threshold to 30 instead of 100, as SIVO uses
    // less close points.
    bool bNeedToInsertClose = (nTrackedClose < 30) && (nNonTrackedClose > 70);

    // Thresholds
    float thRefRatio = 0.75f;

    if (nKFs < 2) {
        thRefRatio = 0.25f;
    }

    // Condition 1a: More than "MaxFrames" have passed from last keyframe
    // insertion
    const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames &&
                      bLocalMappingIdle);
    // Condition 1c: tracking is weak
    const bool c1c =
      mSensor != System::MONOCULAR &&
      (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose);
    // Condition 2: Few tracked points compared to reference keyframe. Lots of
    // visual odometry compared to map matches.
    const bool c2 =
      ((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose) &&
       mnMatchesInliers > 15);

    if ((c1a || c1b || c1c) && c2) {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if (bLocalMappingIdle) {
            return true;
        } else {
            mpLocalMapper->InterruptBA();
            if (mSensor != System::MONOCULAR) {
                if (mpLocalMapper->KeyframesInQueue() < 3) {
                    return true;
                } else {
                    return false;
                }
            } else {
                return false;
            }
        }
    } else {
        return false;
    }
}

void Tracking::CreateNewKeyFrame() {
    if (!mpLocalMapper->SetNotStop(true)) {
        return;
    }

    auto *pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    mCurrentFrame.UpdatePoseMatrices();

    // Points are created via the SIVO information criteria.
    for (unsigned long i = 0;
         i < static_cast<unsigned long>(mCurrentFrame.numSemanticKeys);
         ++i) {
        // Get keypoint coordinates
        auto col = static_cast<int>(mCurrentFrame.mvKeysSemantic.at(i).pt.x);
        auto row = static_cast<int>(mCurrentFrame.mvKeysSemantic.at(i).pt.y);
        float z = mCurrentFrame.mvDepth.at(i);

        // Extract entropy
        double classification_entropy = mCurrentFrame.mEntropy(row, col);

        // Several conditions must be met for map point insertion
        // 1. Ensure depth is > 0
        if (!(z > 0)) {
            continue;
        }

        bool bCreateNew = false;

        MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
        if (!pMP) {
            bCreateNew = true;
        } else if (pMP->Observations() < 1) {
            bCreateNew = true;
            mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(nullptr);
        }

        if (bCreateNew) {
            // Extract 3D point location through unprojection
            cv::Mat wP = mCurrentFrame.UnprojectStereo(i);

            // Extract coordinates
            double wX = wP.at<float>(0, 0);
            double wY = wP.at<float>(1, 0);
            double wZ = wP.at<float>(2, 0);

            // Calculate stereo jacobian for this point.
            StereoProjectionPoseJacobianType stereo_jacobian =
              SIVO::computeStereoJacobianPose(mCurrentFrame.fx,
                                              mCurrentFrame.fy,
                                              mCurrentFrame.mb,
                                              wX,
                                              wY,
                                              wZ);

            // Extract measurement noise for this point.
            // This is done in the same manner as in the optimization.
            double Sigma2 =
              mCurrentFrame
                .mvLevelSigma2[mCurrentFrame.mvKeysSemantic.at(i).octave];
            Eigen::Matrix3d measurement_noise =
              Eigen::Matrix3d::Identity() * Sigma2;

            // Compute the full covariance for the state and this point.
            StereoCovarianceType stereo_covariance =
              SIVO::computeStereoCovariance(
                mCurrentFrame.mSigmacw, stereo_jacobian, measurement_noise);

            // Extract the mutual information between the state and
            // point.
            double mutual_information =
              SIVO::computeStereoMutualInformation(stereo_covariance);

            // The entropy reduction is the joint entropy between this
            // added feature and the classification.
            double entropy_reduction =
              mutual_information - classification_entropy;

            if (entropy_reduction > mThEntropyReduction) {
                // Create new map point
                auto *pNewMP = new MapPoint(wP, pKF, mpMap);

                auto detected_class =
                  static_cast<Classes>(mCurrentFrame.mClasses(row, col));
                pNewMP->SetSemanticInfo(detected_class);

                if (!pNewMP->isBad()) {
                    pNewMP->AddObservation(pKF, static_cast<size_t>(i));

                    // Add map point to Keyframe and map.
                    pKF->AddMapPoint(pNewMP, static_cast<size_t>(i));
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints.at(i) = pNewMP;
                }
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = static_cast<unsigned int>(mCurrentFrame.mnId);
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints() {
    // Do not search map points already matched
    for (auto vit = mCurrentFrame.mvpMapPoints.begin(),
              vend = mCurrentFrame.mvpMapPoints.end();
         vit != vend;
         vit++) {
        MapPoint *pMP = *vit;
        if (pMP) {
            if (pMP->isBad()) {
                *vit = static_cast<MapPoint *>(nullptr);
            } else {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch = 0;

    // Project points in frame and check its visibility
    for (auto vit = mvpLocalMapPoints.begin(); vit != mvpLocalMapPoints.end();
         vit++) {
        MapPoint *pMP = *vit;
        if (pMP->mnLastFrameSeen == mCurrentFrame.mnId) {
            continue;
        }

        if (pMP->isBad()) {
            continue;
        }

        // Project (this fills MapPoint variables for matching)
        if (mCurrentFrame.isInFrustum(pMP, 0.5)) {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if (nToMatch > 0) {
        ORBmatcher matcher(0.8);
        int th = 1;
        if (mSensor == System::RGBD) {
            th = 3;
        }
        // If the camera has been relocalized recently, perform a coarser
        // search
        if (mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
            th = 5;
        }
        matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
    }
}

void Tracking::UpdateLocalMap() {
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints() {
    mvpLocalMapPoints.clear();

    for (std::vector<KeyFrame *>::const_iterator
           itKF = mvpLocalKeyFrames.begin(),
           itEndKF = mvpLocalKeyFrames.end();
         itKF != itEndKF;
         itKF++) {
        KeyFrame *pKF = *itKF;
        const std::vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

        for (auto itMP = vpMPs.begin(); itMP != vpMPs.end(); itMP++) {
            MapPoint *pMP = *itMP;

            if (!pMP) {
                continue;
            }

            if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId) {
                continue;
            }

            if (!pMP->isBad()) {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
            }
        }
    }
}

void Tracking::UpdateLocalKeyFrames() {
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame *, int> keyframeCounter;
    for (int i = 0; i < mCurrentFrame.numSemanticKeys; i++) {
        if (mCurrentFrame.mvpMapPoints[i]) {
            MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
            if (!pMP->isBad()) {
                const map<KeyFrame *, size_t> observations =
                  pMP->GetObservations();
                for (auto it = observations.begin(); it != observations.end();
                     it++)
                    keyframeCounter[it->first]++;
            } else {
                mCurrentFrame.mvpMapPoints[i] = nullptr;
            }
        }
    }

    if (keyframeCounter.empty())
        return;

    int max = 0;
    KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map.
    // Also
    // check which keyframe shares most points
    for (map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(),
                                              itEnd = keyframeCounter.end();
         it != itEnd;
         it++) {
        KeyFrame *pKF = it->first;

        if (pKF->isBad())
            continue;

        if (it->second > max) {
            max = it->second;
            pKFmax = pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors
    // to
    // already-included keyframes
    for (std::vector<KeyFrame *>::const_iterator
           itKF = mvpLocalKeyFrames.begin(),
           itEndKF = mvpLocalKeyFrames.end();
         itKF != itEndKF;
         itKF++) {
        // Limit the number of keyframes
        if (mvpLocalKeyFrames.size() > 80)
            break;

        KeyFrame *pKF = *itKF;

        const std::vector<KeyFrame *> vNeighs =
          pKF->GetBestCovisibilityKeyFrames(10);

        for (std::vector<KeyFrame *>::const_iterator
               itNeighKF = vNeighs.begin(),
               itEndNeighKF = vNeighs.end();
             itNeighKF != itEndNeighKF;
             itNeighKF++) {
            KeyFrame *pNeighKF = *itNeighKF;
            if (!pNeighKF->isBad()) {
                if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame *> spChilds = pKF->GetChildren();
        for (set<KeyFrame *>::const_iterator sit = spChilds.begin(),
                                             send = spChilds.end();
             sit != send;
             sit++) {
            KeyFrame *pChildKF = *sit;
            if (!pChildKF->isBad()) {
                if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame *pParent = pKF->GetParent();
        if (pParent) {
            if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                break;
            }
        }
    }

    if (pKFmax) {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization() {
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for
    // relocalisation
    std::vector<KeyFrame *> vpCandidateKFs =
      mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if (vpCandidateKFs.empty()) {
        return false;
    }

    const int nKFs = static_cast<int>(vpCandidateKFs.size());

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75, true);

    std::vector<PnPsolver *> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    std::vector<std::vector<MapPoint *>> vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    std::vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates = 0;

    for (int i = 0; i < nKFs; i++) {
        KeyFrame *pKF = vpCandidateKFs[i];
        if (pKF->isBad())
            vbDiscarded[i] = true;
        else {
            int nmatches =
              matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
            if (nmatches < 15) {
                vbDiscarded[i] = true;
                continue;
            } else {
                PnPsolver *pSolver =
                  new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9, true);

    while (nCandidates > 0 && !bMatch) {
        for (int i = 0; i < nKFs; i++) {
            if (vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            std::vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver *pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if (bNoMore) {
                vbDiscarded[i] = true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if (!Tcw.empty()) {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint *> sFound;

                const int np = vbInliers.size();

                for (int j = 0; j < np; j++) {
                    if (vbInliers[j]) {
                        mCurrentFrame.mvpMapPoints[j] =
                          vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    } else
                        mCurrentFrame.mvpMapPoints[j] = NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if (nGood < 10)
                    continue;

                for (int io = 0; io < mCurrentFrame.numSemanticKeys; io++)
                    if (mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io] =
                          static_cast<MapPoint *>(nullptr);

                // If few inliers, search by projection in a coarse window
                // and
                // optimize again
                if (nGood < 50) {
                    int nadditional = matcher2.SearchByProjection(
                      mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);

                    if (nadditional + nGood >= 50) {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by
                        // projection again in a narrower window
                        // the camera has been already optimized with many
                        // points
                        if (nGood > 30 && nGood < 50) {
                            sFound.clear();
                            for (int ip = 0; ip < mCurrentFrame.numSemanticKeys;
                                 ip++)
                                if (mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(
                                      mCurrentFrame.mvpMapPoints[ip]);
                            nadditional = matcher2.SearchByProjection(
                              mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);

                            // Final optimization
                            if (nGood + nadditional >= 50) {
                                nGood =
                                  Optimizer::PoseOptimization(&mCurrentFrame);

                                for (int io = 0;
                                     io < mCurrentFrame.numSemanticKeys;
                                     io++)
                                    if (mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io] =
                                          nullptr;
                            }
                        }
                    }
                }

                // If the pose is supported by enough inliers stop ransac
                // and
                // continue
                if (nGood >= 50) {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if (!bMatch) {
        return false;
    } else {
        mnLastRelocFrameId = static_cast<unsigned int>(mCurrentFrame.mnId);
        return true;
    }
}

void Tracking::Reset() {
    std::cout << "System Resetting" << std::endl;
    if (mpViewer) {
        mpViewer->RequestStop();
        while (!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    std::cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    std::cout << " done" << std::endl;

    // Reset Loop Closing
    std::cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    std::cout << " done" << std::endl;

    // Clear BoW Database
    std::cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    std::cout << " done" << std::endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if (mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath) {
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if (k3 != 0) {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag) {
    mbOnlyTracking = flag;
}

}  // namespace ORB_SLAM
