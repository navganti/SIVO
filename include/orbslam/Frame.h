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

#ifndef FRAME_H
#define FRAME_H

#include <vector>

#include "bayesian_segnet/bayesian_segnet.hpp"
#include "sivo_helpers/sivo_helpers.hpp"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "ORBextractor.h"
#include "ORBVocabulary.h"

#include "dependencies/DBoW2/DBoW2/BowVector.h"
#include "dependencies/DBoW2/DBoW2/FeatureVector.h"

#include <opencv2/opencv.hpp>

namespace SIVO {

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class MapPoint;
class KeyFrame;

class Frame {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Frame();

    // Copy constructor.
    Frame(const Frame &frame);

    // Constructor for stereo cameras.
    Frame(const cv::Mat &imLeftGrey,
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
          const float &thEntropyReduction);

    // Extract ORB on the image. 0 for left image and 1 for right image.
    void ExtractORB(int flag, const cv::Mat &im);

    // Perform segmentation inference, and classify the pixels.
    void SegmentImage(const cv::Mat &im);

    /// Retrieve the overlaid segmented image.
    cv::Mat getSegmentedImage();

    /// Compute Bag of Words representation.
    void ComputeBoW();

    /// Set the camera pose.
    void SetPose(cv::Mat Tcw);

    /// Set the camera covariance
    void SetCovariance(const StateCovarianceType &Sigmacw);

    // Computes rotation, translation and camera center matrices from the camera
    // pose.
    void UpdatePoseMatrices();

    // Returns the camera center.
    inline cv::Mat GetCameraCenter() {
        return mOw.clone();
    }

    // Returns inverse of rotation
    inline cv::Mat GetRotationInverse() {
        return mRwc.clone();
    }

    // Check if a MapPoint is in the frustum of the camera
    // and fill variables of the MapPoint to be used by the tracking
    bool isInFrustum(MapPoint *pMP, float viewingCosLimit);

    // Compute the cell of a keypoint (return false if outside the grid)
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    std::vector<size_t> GetFeaturesInArea(const float &x,
                                          const float &y,
                                          const float &r,
                                          const int minLevel = -1,
                                          const int maxLevel = -1) const;

    // Search a match for each keypoint in the left image to a keypoint in the
    // right image.
    // If there is a match, depth is computed and the right coordinate
    // associated to the left keypoint is stored.
    void ComputeStereoMatches();

    /// Backprojects a keypoint (with depth) into 3D world coordinates.
    cv::Mat UnprojectStereo(const unsigned long &i);

 public:
    /// Vocabulary used for relocalization.
    ORBVocabulary *mpORBvocabulary;

    /// Feature extractor. The right is used only in the stereo case.
    ORBextractor *mpORBextractorLeft, *mpORBextractorRight;

    /// Bayesian SegNet for semantic segmentation
    BayesianSegNet *mpBayesianSegNet;

    /// Frame timestamp.
    double mTimeStamp;

    /// Calibration matrix and OpenCV distortion parameters.
    cv::Mat mK;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    cv::Mat mDistCoef;

    /// The overlaid semantically segmented image.
    cv::Mat mImSemantic;

    /// Stereo baseline multiplied by fx.
    float mbf;

    /// Stereo baseline in meters.
    float mb;

    /** Threshold close/far points. Close points are inserted from 1 view.
     * Far points are inserted as in the monocular case from 2 views.
     */
    float mThDepth;

    /** Semantic segmentation detection threshold. If the confidence is below
     * this value, the detection is discarded.
     */
    float mThConfidence;

    /** Entropy reduction threshold for feature selection. A feature will only
     * be selected if the entropy reduction is greater than this threshold.
     */
    float mThEntropyReduction;

    /** Number of Semantic KeyPoints - keypoints which are one of the approved
     * static classes.
     */
    int numSemanticKeys;

    /// Vector of all keypoints detected in the left image
    std::vector<cv::KeyPoint> mvKeysLeft;

    /// Vector of keypoints which are one of the approved static classes.
    std::vector<cv::KeyPoint> mvKeysSemantic;

    /// Vector of keypoints detected in the right image.
    std::vector<cv::KeyPoint> mvKeysRight;

    // Corresponding stereo coordinate and depth for each keypoint.
    // "Monocular" keypoints have a negative value.
    std::vector<float> mvRight;
    std::vector<float> mvDepth;

    // Matrices for class detection, confidence, and entropy
    MatXu mClasses;
    MatXd mConfidence;
    MatXd mEntropy;

    // Bag of Words Vector structures.
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // ORB descriptor, each row associated to a keypoint.
    cv::Mat mDescriptorsLeft, mDescriptorsRight, mDescriptorsSemantic;

    // MapPoints associated to keypoints, NULL pointer if no association.
    std::vector<MapPoint *> mvpMapPoints;

    // Flag to identify outlier associations.
    std::vector<bool> mvbOutlier;

    // Keypoints are assigned to cells in a grid to reduce matching complexity
    // when projecting MapPoints.
    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    /// Camera pose.
    cv::Mat mTcw;

    /// Covariance of Camera pose.
    StateCovarianceType mSigmacw;

    // Current and Next Frame id.
    static long unsigned int nNextId;
    long unsigned int mnId;

    /// Reference Keyframe.
    KeyFrame *mpReferenceKF;

    // Scale pyramid info.
    int mnScaleLevels;
    float mfScaleFactor;
    float mfLogScaleFactor;
    std::vector<float> mvScaleFactors;
    std::vector<float> mvInvScaleFactors;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    // Undistorted Image Bounds (computed once).
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    static bool mbInitialComputations;

 private:
    /// Determines keypoints which are static
    void SelectSemanticKeys();

    /** Computes image bounds for the undistorted image (called in the
     * constructor)
     *
     * @param imLeft The undistorted image.
     */
    void ComputeImageBounds(const cv::Mat &imLeft);

    /** Assign keypoints to the grid for faster feature matching (called in the
     * constructor)
     */
    void AssignFeaturesToGrid();

    /// Camera rotation.
    cv::Mat mRcw;

    /// Inverse camera rotation
    cv::Mat mRwc;

    /// Camera translation
    cv::Mat mtcw;

    /// Location of camera center.
    cv::Mat mOw;
};

}  // namespace SIVO

#endif  // FRAME_H
