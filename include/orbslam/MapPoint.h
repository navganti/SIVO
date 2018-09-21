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

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "KeyFrame.h"
#include "Frame.h"
#include "Map.h"
#include "bayesian_segnet/bayesian_segnet.hpp"
#include "sivo_helpers/sivo_helpers.hpp"

#include <opencv2/core/core.hpp>
#include <mutex>
#include <map>

namespace SIVO {

class KeyFrame;
class Map;
class Frame;

/** Structure containing keypoint information as well as class detection,
 * confidence, and variance.
 */
struct SemanticKeypoint {
    SemanticKeypoint(const cv::KeyPoint &kp,
                     const Classes &detected_class,
                     const double confidence,
                     const double entropy)
        : kp(kp),
          detected_class(detected_class),
          confidence(confidence),
          entropy(entropy) {}

    /// The point feature location determined by the ORBextractor.
    cv::KeyPoint kp;

    /// The detected class from BayesianSegNet
    Classes detected_class = Classes::VOID;

    /// The confidence of the class detection
    double confidence = 0.0;

    /// The entropy of the detected class confidence.
    double entropy = 0.0;

    /// The net entropy reduction for this point.
    double entropy_reduction = 0.0;
};

class MapPoint {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map *pMap);
    MapPoint(const cv::Mat &Pos, Map *pMap, Frame *pFrame, const int &idxF);

    void SetSemanticInfo(const Classes &detected_class);
    Classes GetSemanticInfo();

    void SetWorldPos(const cv::Mat &Pos);
    cv::Mat GetWorldPos();

    void SetCovariance(const StateCovarianceType &Sigma);
    StateCovarianceType GetCovariance();

    cv::Mat GetNormal();
    KeyFrame *GetReferenceKeyFrame();

    std::map<KeyFrame *, size_t> GetObservations();
    int Observations();

    void AddObservation(KeyFrame *pKF, size_t idx);
    void EraseObservation(KeyFrame *pKF);

    int GetIndexInKeyFrame(KeyFrame *pKF);
    bool IsInKeyFrame(KeyFrame *pKF);

    void SetBadFlag();
    bool isBad();

    void Replace(MapPoint *pMP);
    MapPoint *GetReplaced();

    void IncreaseVisible(int n = 1);
    void IncreaseFound(int n = 1);
    float GetFoundRatio();
    inline int GetFound() {
        return mnFound;
    }

    void ComputeDistinctiveDescriptors();

    cv::Mat GetDescriptor();

    void UpdateNormalAndDepth();

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();
    int PredictScale(const float &currentDist, KeyFrame *pKF);
    int PredictScale(const float &currentDist, Frame *pF);

 public:
    long unsigned int mnId;
    static long unsigned int nNextId;
    long int mnFirstKFid;
    long int mnFirstFrame;
    int nObs;

    // Variables used by the tracking
    float mTrackProjX;
    float mTrackProjY;
    float mTrackProjXR;
    bool mbTrackInView;
    int mnTrackScaleLevel;
    float mTrackViewCos;
    long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnLastFrameSeen;

    // Variables used by local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;

    // Variables used by loop closing
    long unsigned int mnLoopPointForKF;
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;
    cv::Mat mPosGBA;
    long unsigned int mnBAGlobalForKF;

    static std::mutex mGlobalMutex;

 protected:
    /// Position in absolute coordinates
    cv::Mat mWorldPos;

    /// Covariance of position
    StateCovarianceType mSigma;

    /// Keyframes observing the point and associated index in keyframe
    std::map<KeyFrame *, size_t> mObservations;

    /// Mean viewing direction
    cv::Mat mNormalVector;

    /// Best descriptor to fast matching
    cv::Mat mDescriptor;

    /// Reference KeyFrame
    KeyFrame *mpRefKF;

    /// Tracking counters
    int mnVisible;
    int mnFound;

    /// Bad flag (we do not currently erase MapPoint from memory)
    bool mbBad;
    MapPoint *mpReplaced;

    /// Scale invariance distances
    float mfMinDistance;
    float mfMaxDistance;

    Map *mpMap;

    /// Mutexes
    std::mutex mMutexPos;
    std::mutex mMutexFeatures;
    std::mutex mMutexCovariance;

    // Semantic Information
    Classes mClass = Classes::VOID;
};
}  // namespace SIVO

#endif  // MAPPOINT_H
