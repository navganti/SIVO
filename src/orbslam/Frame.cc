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

#include "include/orbslam/Frame.h"
#include "include/orbslam/Converter.h"
#include "include/orbslam/ORBmatcher.h"
#include <thread>

namespace SIVO {

long unsigned int Frame::nNextId = 0;
bool Frame::mbInitialComputations = true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame() = default;

// Copy Constructor
Frame::Frame(const Frame &frame)
    : mpORBvocabulary(frame.mpORBvocabulary),
      mpORBextractorLeft(frame.mpORBextractorLeft),
      mpORBextractorRight(frame.mpORBextractorRight),
      mpBayesianSegNet(frame.mpBayesianSegNet),
      mTimeStamp(frame.mTimeStamp),
      mK(frame.mK.clone()),
      mDistCoef(frame.mDistCoef.clone()),
      mbf(frame.mbf),
      mb(frame.mb),
      mThDepth(frame.mThDepth),
      mThConfidence(frame.mThConfidence),
      mThEntropyReduction(frame.mThEntropyReduction),
      numSemanticKeys(frame.numSemanticKeys),
      mvKeysLeft(frame.mvKeysLeft),
      mvKeysSemantic(frame.mvKeysSemantic),
      mvKeysRight(frame.mvKeysRight),
      mvRight(frame.mvRight),
      mvDepth(frame.mvDepth),
      mBowVec(frame.mBowVec),
      mFeatVec(frame.mFeatVec),
      mDescriptorsLeft(frame.mDescriptorsLeft.clone()),
      mDescriptorsRight(frame.mDescriptorsRight.clone()),
      mDescriptorsSemantic(frame.mDescriptorsSemantic.clone()),
      mvpMapPoints(frame.mvpMapPoints),
      mvbOutlier(frame.mvbOutlier),
      mnId(frame.mnId),
      mpReferenceKF(frame.mpReferenceKF),
      mnScaleLevels(frame.mnScaleLevels),
      mfScaleFactor(frame.mfScaleFactor),
      mfLogScaleFactor(frame.mfLogScaleFactor),
      mvScaleFactors(frame.mvScaleFactors),
      mvInvScaleFactors(frame.mvInvScaleFactors),
      mvLevelSigma2(frame.mvLevelSigma2),
      mvInvLevelSigma2(frame.mvInvLevelSigma2) {
    for (int i = 0; i < FRAME_GRID_COLS; i++)
        for (int j = 0; j < FRAME_GRID_ROWS; j++)
            mGrid[i][j] = frame.mGrid[i][j];

    if (!frame.mTcw.empty()) {
        SetPose(frame.mTcw);
    }

    if (!frame.mSigmacw.isZero(0)) {
        mSigmacw = frame.mSigmacw;
    }
}

Frame::Frame(const cv::Mat &imLeftGrey,
             const cv::Mat &imLeftColour,
             const cv::Mat &imRight,
             const double &timeStamp,
             ORBextractor *pORBextractorLeft,
             ORBextractor *pORBextractorRight,
             ORBVocabulary *voc,
             BayesianSegNet *pBayesianSegNet,
             cv::Mat &K,
             cv::Mat &distCoef,
             const float &bf,
             const float &thDepth,
             const float &thConfidence,
             const float &thEntropyReduction)
    : mpORBvocabulary(voc),
      mpORBextractorLeft(pORBextractorLeft),
      mpORBextractorRight(pORBextractorRight),
      mpBayesianSegNet(pBayesianSegNet),
      mTimeStamp(timeStamp),
      mK(K.clone()),
      mDistCoef(distCoef.clone()),
      mbf(bf),
      mThDepth(thDepth),
      mThConfidence(thConfidence),
      mThEntropyReduction(thEntropyReduction),
      mpReferenceKF(static_cast<KeyFrame *>(nullptr)) {
    // Frame ID
    mnId = nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = static_cast<float>(mpORBextractorLeft->GetScaleFactor());
    mfLogScaleFactor = std::log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // Semantic segmentation and ORB extraction
    SegmentImage(imLeftColour);
    std::thread threadLeft(&Frame::ExtractORB, this, 0, imLeftGrey);
    std::thread threadRight(&Frame::ExtractORB, this, 1, imRight);
    threadLeft.join();
    threadRight.join();

    if (mvKeysLeft.empty()) {
        return;
    } else {
        SelectSemanticKeys();

        if (mvKeysSemantic.empty()) {
            return;
        }
    }

    // Get the number of total keypoints, and compute stereo matches
    numSemanticKeys = static_cast<int>(mvKeysSemantic.size());

    ComputeStereoMatches();

    mvpMapPoints =
      std::vector<MapPoint *>(static_cast<unsigned long>(numSemanticKeys),
                              static_cast<MapPoint *>(nullptr));
    mvbOutlier =
      std::vector<bool>(static_cast<unsigned long>(numSemanticKeys), false);

    // This is done only for the first Frame (or after a change in the
    // calibration)
    if (mbInitialComputations) {
        ComputeImageBounds(imLeftGrey);

        mfGridElementWidthInv =
          static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);
        mfGridElementHeightInv =
          static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);

        fx = K.at<float>(0, 0);
        fy = K.at<float>(1, 1);
        cx = K.at<float>(0, 2);
        cy = K.at<float>(1, 2);
        invfx = 1.0f / fx;
        invfy = 1.0f / fy;

        mbInitialComputations = false;
    }

    mb = mbf / fx;

    AssignFeaturesToGrid();
}

void Frame::SelectSemanticKeys() {
    // ORB_SLAM2's tracking looks at the total number of detected keypoints.
    // Instead, it should look at the number of potential keys that are from the
    // accepted static classes.
    for (size_t i = 0; i < mvKeysLeft.size(); ++i) {
        // Get keypoint coordinate
        auto col = static_cast<int>(mvKeysLeft.at(i).pt.x);
        auto row = static_cast<int>(mvKeysLeft.at(i).pt.y);

        // Determine class
        auto detection = static_cast<Classes>(mClasses(row, col));

        // If detection is deemed to be a static object
        if (detection <= Classes::TERRAIN) {
            // Save keypoints and descriptors
            mvKeysSemantic.emplace_back(mvKeysLeft.at(i));

            if (mDescriptorsSemantic.empty()) {
                mDescriptorsSemantic =
                  mDescriptorsLeft.row(static_cast<int>(i));
            } else {
                mDescriptorsSemantic.push_back(
                  mDescriptorsLeft.row(static_cast<int>(i)));
            }
        }
    }
}

void Frame::AssignFeaturesToGrid() {
    int nReserve = 0.5f * numSemanticKeys / (FRAME_GRID_COLS * FRAME_GRID_ROWS);

    for (size_t i = 0; i < FRAME_GRID_COLS; i++) {
        for (size_t j = 0; j < FRAME_GRID_ROWS; j++) {
            mGrid[i][j].reserve(static_cast<size_t>(nReserve));
        }
    }

    for (int i = 0; i < numSemanticKeys; i++) {
        const cv::KeyPoint &kp = mvKeysSemantic[i];

        int nGridPosX, nGridPosY;
        if (PosInGrid(kp, nGridPosX, nGridPosY)) {
            mGrid[nGridPosX][nGridPosY].push_back(
              static_cast<unsigned long>(i));
        }
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im) {
    if (flag == 0)
        (*mpORBextractorLeft)(im, cv::Mat(), mvKeysLeft, mDescriptorsLeft);
    else
        (*mpORBextractorRight)(im, cv::Mat(), mvKeysRight, mDescriptorsRight);
}

void Frame::SegmentImage(const cv::Mat &im) {
    MatXu classes;
    MatXd confidence;
    MatXd entropy;

    mpBayesianSegNet->segmentImage(im, classes, confidence, entropy);

    mClasses = classes;
    mConfidence = confidence;
    mEntropy = entropy;

    // Generate overlaid semantic image
    mImSemantic = mpBayesianSegNet->generateSegmentedImage(classes, im);
}

cv::Mat Frame::getSegmentedImage() {
    return mImSemantic.clone();
}

void Frame::SetPose(cv::Mat Tcw) {
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices() {
    mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0, 3).col(3);
    mOw = -mRcw.t() * mtcw;
}

void Frame::SetCovariance(const StateCovarianceType &Sigmacw) {
    mSigmacw = Sigmacw;
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit) {
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos();

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw * P + mtcw;
    const auto &PcX = Pc.at<float>(0);
    const auto &PcY = Pc.at<float>(1);
    const auto &PcZ = Pc.at<float>(2);

    // Check positive depth
    if (PcZ < 0.0f)
        return false;

    // Project in image and check it is not outside
    const float invz = 1.0f / PcZ;
    const float u = fx * PcX * invz + cx;
    const float v = fy * PcY * invz + cy;

    if (u < mnMinX || u > mnMaxX)
        return false;
    if (v < mnMinY || v > mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P - mOw;
    const auto dist = static_cast<float>(cv::norm(PO));

    if (dist < minDistance || dist > maxDistance) {
        return false;
    }

    // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = static_cast<float>(PO.dot(Pn) / dist);

    if (viewCos < viewingCosLimit) {
        return false;
    }

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist, this);

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf * invz;
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel = nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}

std::vector<size_t> Frame::GetFeaturesInArea(const float &x,
                                             const float &y,
                                             const float &r,
                                             const int minLevel,
                                             const int maxLevel) const {
    std::vector<size_t> vIndices;
    vIndices.reserve(static_cast<unsigned long>(numSemanticKeys));

    const int nMinCellX =
      max(0, (int) floor((x - mnMinX - r) * mfGridElementWidthInv));
    if (nMinCellX >= FRAME_GRID_COLS) {
        return vIndices;
    }

    const int nMaxCellX =
      min((int) FRAME_GRID_COLS - 1,
          (int) ceil((x - mnMinX + r) * mfGridElementWidthInv));
    if (nMaxCellX < 0) {
        return vIndices;
    }

    const int nMinCellY =
      max(0, (int) floor((y - mnMinY - r) * mfGridElementHeightInv));
    if (nMinCellY >= FRAME_GRID_ROWS) {
        return vIndices;
    }

    const int nMaxCellY =
      min((int) FRAME_GRID_ROWS - 1,
          (int) ceil((y - mnMinY + r) * mfGridElementHeightInv));
    if (nMaxCellY < 0) {
        return vIndices;
    }

    const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

    for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
        for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
            const vector<size_t> vCell = mGrid[ix][iy];
            if (vCell.empty()) {
                continue;
            }

            for (size_t j = 0; j < vCell.size(); j++) {
                const cv::KeyPoint &kp = mvKeysSemantic[vCell[j]];
                if (bCheckLevels) {
                    if (kp.octave < minLevel)
                        continue;
                    if (maxLevel >= 0)
                        if (kp.octave > maxLevel)
                            continue;
                }

                const float distx = kp.pt.x - x;
                const float disty = kp.pt.y - y;

                if (fabs(distx) < r && fabs(disty) < r) {
                    vIndices.push_back(vCell[j]);
                }
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY) {
    posX = static_cast<int>(round((kp.pt.x - mnMinX) * mfGridElementWidthInv));
    posY = static_cast<int>(round((kp.pt.y - mnMinY) * mfGridElementHeightInv));

    // Keypoint's coordinates are undistorted, which could cause to go out of
    // the image
    if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 ||
        posY >= FRAME_GRID_ROWS) {
        return false;
    }

    return true;
}

void Frame::ComputeBoW() {
    if (mBowVec.empty()) {
        vector<cv::Mat> vCurrentDesc =
          Converter::toDescriptorVector(mDescriptorsSemantic);
        mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft) {
    if (mDistCoef.at<float>(0) != 0.0) {
        cv::Mat mat(4, 2, CV_32F);
        mat.at<float>(0, 0) = 0.0;
        mat.at<float>(0, 1) = 0.0;
        mat.at<float>(1, 0) = imLeft.cols;
        mat.at<float>(1, 1) = 0.0;
        mat.at<float>(2, 0) = 0.0;
        mat.at<float>(2, 1) = imLeft.rows;
        mat.at<float>(3, 0) = imLeft.cols;
        mat.at<float>(3, 1) = imLeft.rows;

        // Undistort corners
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
        mat = mat.reshape(1);

        mnMinX = min(mat.at<float>(0, 0), mat.at<float>(2, 0));
        mnMaxX = max(mat.at<float>(1, 0), mat.at<float>(3, 0));
        mnMinY = min(mat.at<float>(0, 1), mat.at<float>(1, 1));
        mnMaxY = max(mat.at<float>(2, 1), mat.at<float>(3, 1));

    } else {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches() {
    mvRight = std::vector<float>(static_cast<size_t>(numSemanticKeys), -1.0f);
    mvDepth = std::vector<float>(static_cast<size_t>(numSemanticKeys), -1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH + ORBmatcher::TH_LOW) / 2;

    const auto nRows =
      static_cast<unsigned long>(mpORBextractorLeft->mvImagePyramid[0].rows);

    // Assign keypoints to row table
    std::vector<std::vector<size_t>> vRowIndices(nRows, std::vector<size_t>());

    for (unsigned long i = 0; i < nRows; i++) {
        // Reserve up to 200 potential keypoints available at each row.
        vRowIndices[i].reserve(200);
    }

    const auto Nr = static_cast<int>(mvKeysRight.size());

    for (int iR = 0; iR < Nr; iR++) {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float row = 2.0f * mvScaleFactors[mvKeysRight[iR].octave];
        const auto maxr = static_cast<int>(ceil(kpY + row));
        const auto minr = static_cast<int>(floor(kpY - row));

        // Maintain keypoints potentially at the same height in the right image
        // factoring in scale factor.
        for (int yi = minr; yi <= maxr; yi++) {
            vRowIndices[yi].push_back(static_cast<unsigned long>(iR));
        }
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf / minZ;

    // For each left keypoint search a match in the right image
    std::vector<std::pair<int, int>> vDistIdx;
    vDistIdx.reserve(static_cast<unsigned long>(numSemanticKeys));

    for (int iL = 0; iL < numSemanticKeys; iL++) {
        // Extract information about left keypoint
        const cv::KeyPoint &kpL = mvKeysSemantic[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        // Extract features found at this image row
        const std::vector<size_t> &vCandidates = vRowIndices[vL];

        if (vCandidates.empty()) {
            continue;
        }

        const float minU = uL - maxD;
        const float maxU = uL - minD;

        if (maxU < 0) {
            continue;
        }

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptorsSemantic.row(iL);

        // Compare descriptor to right keypoints
        for (size_t iC = 0; iC < vCandidates.size(); iC++) {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if (kpR.octave < levelL - 1 || kpR.octave > levelL + 1) {
                continue;
            }

            const float &uR = kpR.pt.x;

            if (uR >= minU && uR <= maxU) {
                // Extract right image descriptor, and compare
                const cv::Mat &dR = mDescriptorsRight.row(static_cast<int>(iR));
                const int dist = ORBmatcher::DescriptorDistance(dL, dR);

                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if (bestDist < thOrbDist) {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x * scaleFactor);
            const float scaledvL = round(kpL.pt.y * scaleFactor);
            const float scaleduR0 = round(uR0 * scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave]
                           .rowRange(scaledvL - w, scaledvL + w + 1)
                           .colRange(scaleduL - w, scaleduL + w + 1);
            IL.convertTo(IL, CV_32F);
            IL =
              IL - IL.at<float>(w, w) * cv::Mat::ones(IL.rows, IL.cols, CV_32F);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            std::vector<float> vDists;
            vDists.resize(2 * L + 1);

            const float iniu = scaleduR0 + L - w;
            const float endu = scaleduR0 + L + w + 1;
            if (iniu < 0 ||
                endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols) {
                continue;
            }

            for (int incR = -L; incR <= +L; incR++) {
                cv::Mat IR =
                  mpORBextractorRight->mvImagePyramid[kpL.octave]
                    .rowRange(scaledvL - w, scaledvL + w + 1)
                    .colRange(scaleduR0 + incR - w, scaleduR0 + incR + w + 1);
                IR.convertTo(IR, CV_32F);
                IR =
                  IR -
                  IR.at<float>(w, w) * cv::Mat::ones(IR.rows, IR.cols, CV_32F);

                float dist = static_cast<float>(cv::norm(IL, IR, cv::NORM_L1));
                if (dist < bestDist) {
                    bestDist = static_cast<int>(dist);
                    bestincR = incR;
                }

                vDists[L + incR] = dist;
            }

            if (bestincR == -L || bestincR == L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L + bestincR - 1];
            const float dist2 = vDists[L + bestincR];
            const float dist3 = vDists[L + bestincR + 1];

            const float deltaR =
              (dist1 - dist3) / (2.0f * (dist1 + dist3 - 2.0f * dist2));

            if (deltaR < -1 || deltaR > 1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave] *
                           ((float) scaleduR0 + (float) bestincR + deltaR);

            float disparity = (uL - bestuR);

            if (disparity >= minD && disparity < maxD) {
                if (disparity <= 0) {
                    disparity = 0.01;
                    bestuR = static_cast<float>(uL - 0.01);
                }
                mvDepth[iL] = mbf / disparity;
                mvRight[iL] = bestuR;
                vDistIdx.emplace_back(std::pair<int, int>(bestDist, iL));
            }
        }
    }

    std::sort(vDistIdx.begin(), vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size() / 2].first;
    const float thDist = 1.5f * 1.4f * median;

    for (int i = static_cast<int>(vDistIdx.size() - 1); i >= 0; i--) {
        if (vDistIdx[i].first < thDist) {
            break;
        } else {
            mvRight[vDistIdx[i].second] = -1;
            mvDepth[vDistIdx[i].second] = -1;
        }
    }
}

cv::Mat Frame::UnprojectStereo(const unsigned long &i) {
    const float z = mvDepth.at(i);
    if (z > 0) {
        const float u = mvKeysSemantic.at(i).pt.x;
        const float v = mvKeysSemantic.at(i).pt.y;
        const float x = (u - cx) * z * invfx;
        const float y = (v - cy) * z * invfy;

        cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);

        return mRwc * x3Dc + mOw;
    } else {
        return cv::Mat();
    }
}

}  // namespace SIVO
