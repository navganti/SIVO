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

#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include "KeyFrame.h"
#include "Map.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"

#include <mutex>

namespace SIVO {

class Tracking;
class LoopClosing;
class Map;

class LocalMapping {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    LocalMapping(Map *pMap, const float bMonocular);

    void SetLoopCloser(LoopClosing *pLoopCloser);

    void SetTracker(Tracking *pTracker);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame *pKF);

    // Thread Synch
    void RequestStop();
    void RequestReset();
    bool Stop();
    void Release();
    bool isStopped();
    bool stopRequested();
    bool AcceptKeyFrames();

    /** Locks mutex, and the tracking thread will see that local mapping is
     * busy.
     *
     * @param flag The flag to accept keyframes, whether it is true or false.
     */
    void SetAcceptKeyFrames(bool flag);

    bool SetNotStop(bool flag);

    void InterruptBA();

    void RequestFinish();
    bool isFinished();

    int KeyframesInQueue() {
        std::unique_lock<std::mutex> lock(mMutexNewKFs);
        return static_cast<int>(mlNewKeyFrames.size());
    }

 protected:
    /** Checks to see if there are new keyframes in the queue.
     *
     * @return Boolean indicating whether there are new keyframes in the queue.
     */
    bool CheckNewKeyFrames();

    /// Bag of Words conversion, and keyframe insertion in the map.
    void ProcessNewKeyFrame();

    /// Checks map points, determines whether any need to be removed.
    void MapPointCulling();

    /// Triangulates new mapPoints
    void CreateNewMapPoints();

    /// If no new keyframes, looks for more matches in neighbour keyframes
    void SearchInNeighbors();

    /** Verifies the SIVO semantic checking criteria. Point must have depth, must be one of the static classes,
     * and the confidence must be above the threshold. For the current frame point we evaluate this information, but
     * for the matched feature we only return the true class.
     *
     * @param pKF The pointer to the keyframe.
     * @param idx The current index from the CreateNewMapPoints loop.
     * @param wP The position of the 3D point with respect to the world frame,
     * expressed in the world frame. wP = wPwp.
     * @param compute_information A boolean indicating whether to actually verify that the feature meets our information
     * criteria, or if we are just looking to get the semantic class information.
     * @return The detected class. If the feature does not match our criteria, this is returned as VOID.
     */
    Classes CheckSemantics(const KeyFrame* pKF, const int idx, const cv::Mat &wP, bool compute_information);

    /** Removes redundant local keyframes. A keyframe is considered redundant
     * if the 90% of the MapPoints it sees are seen in at least 3 other
     * keyframes (in the same or finer scale). Only close stereo points are
     * considered.
     */
    void KeyFrameCulling();

    /** Computes the Fundamental matrix between two keyframes.
     *
     * @param pKF1 The pointer to the first keyframe.
     * @param pKF2 The pointer to the second keyframe.
     * @return The fundamental matrix between the two frames.
     */
    cv::Mat ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2);

    /** Converts a 3D vector into its skew-symmetrical form.
     *
     * @param v The vector of points.
     * @return The skew-symmetric matrix.
     */
    cv::Mat SkewSymmetricMatrix(const cv::Mat &v);

    bool mbMonocular;

    void ResetIfRequested();
    bool mbResetRequested;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    Map *mpMap;

    LoopClosing *mpLoopCloser;
    Tracking *mpTracker;

    std::list<KeyFrame *> mlNewKeyFrames;

    KeyFrame *mpCurrentKeyFrame;

    std::list<MapPoint *> mlpRecentAddedMapPoints;

    std::mutex mMutexNewKFs;

    bool mbAbortBA;

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;

    bool mbAcceptKeyFrames;
    std::mutex mMutexAccept;
};
}  // namespace SIVO

#endif  // LOCALMAPPING_H
