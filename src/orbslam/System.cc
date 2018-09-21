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

#include "include/orbslam/Converter.h"
#include "include/orbslam/System.h"
#include <iomanip>
#include <pangolin/pangolin.h>
#include <thread>

namespace SIVO {

System::System(const string &strVocFile,
               const string &strSettingsFile,
               const string &strPrototxtFile,
               const string &strWeightsFile,
               const eSensor sensor,
               const bool bUseViewer)
    : mSensor(sensor),
      mpViewer(static_cast<Viewer *>(NULL)),
      mbReset(false),
      mbActivateLocalizationMode(false),
      mbDeactivateLocalizationMode(false) {
    // Output welcome message
    std::cout
      << std::endl
      << "ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of "
         "Zaragoza."
      << std::endl
      << "This program comes with ABSOLUTELY NO WARRANTY;" << std::endl
      << "This is free software, and you are welcome to redistribute it"
      << std::endl
      << "under certain conditions. See LICENSE.txt." << std::endl
      << std::endl;

    if (mSensor != STEREO) {
        std::cerr << "Sensor mode must be Stereo!" << std::endl;
        exit(-1);
    }

    // Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        std::cerr << "Failed to open settings file at: " << strSettingsFile
                  << std::endl;
        exit(-1);
    }

    // Load ORB Vocabulary
    std::cout << std::endl
              << "Loading ORB Vocabulary. This could take a while..."
              << std::endl;

    mpVocabulary = new ORBVocabulary();
    bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
    if (!bVocLoad) {
        std::cerr << "Wrong path to vocabulary. " << std::endl;
        std::cerr << "Failed to open at: " << strVocFile << std::endl;
        exit(-1);
    }
    std::cout << "Vocabulary loaded!" << std::endl << std::endl;

    // Create KeyFrame Database
    mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

    // Create the Map
    mpMap = new Map();

    // Create Drawers. These are used by the Viewer
    mpFrameDrawer = new FrameDrawer(mpMap);
    mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

    BayesianSegNetParams params{strPrototxtFile, strWeightsFile};
    mpBayesianSegNet = new BayesianSegNet(params);

    // Initialize the Tracking thread
    //(it will live in the main thread of execution, the one that called this
    // constructor)
    mpTracker = new Tracking(this,
                             mpVocabulary,
                             mpFrameDrawer,
                             mpMapDrawer,
                             mpMap,
                             mpKeyFrameDatabase,
                             mpBayesianSegNet,
                             strSettingsFile,
                             mSensor);

    std::cout << "Tracker thread created!" << std::endl;

    // Initialize the Local Mapping thread and launch
    mpLocalMapper = new LocalMapping(mpMap, mSensor == MONOCULAR);
    mptLocalMapping = new thread(&LocalMapping::Run, mpLocalMapper);

    std::cout << "Local mapping thread created!" << std::endl;

    // Initialize the Loop Closing thread and launch
    mpLoopCloser = new LoopClosing(
      mpMap, mpKeyFrameDatabase, mpVocabulary, mSensor != MONOCULAR);
    mptLoopClosing = new thread(&LoopClosing::Run, mpLoopCloser);

    std::cout << "Loop closing thread created!" << std::endl;

    // Initialize the Viewer thread and launch
    if (bUseViewer) {
        cv::Size input_geometry = mpBayesianSegNet->getInputGeometry();
        mpViewer = new Viewer(this,
                              mpFrameDrawer,
                              mpMapDrawer,
                              mpTracker,
                              strSettingsFile,
                              input_geometry);
        mptViewer = new thread(&Viewer::Run, mpViewer);
        mpTracker->SetViewer(mpViewer);

        std::cout << "Viewer thread created!" << std::endl;
    }

    // Set pointers between threads
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);

    std::cout << "Thread references set!" << std::endl;
}


std::pair<cv::Mat, cv::Mat> System::resizeImages(const cv::Mat &imLeft,
                                                 const cv::Mat &imRight) {
    std::pair<cv::Mat, cv::Mat> resizedImages;

    cv::Size input_geometry = mpBayesianSegNet->getInputGeometry();

    // Resize image to use with bayesian segnet
    int x_tl = imLeft.cols / 2 - input_geometry.width / 2;
    int y_tl = imLeft.rows / 2 - input_geometry.height / 2;
    cv::Rect roi{x_tl, y_tl, input_geometry.width, input_geometry.height};

    // Crop image, and clone to ensure contiguous data.
    imLeft(roi).copyTo(resizedImages.first);
    imRight(roi).copyTo(resizedImages.second);

    return resizedImages;
}

cv::Mat System::TrackStereo(const cv::Mat &imLeft,
                            const cv::Mat &imRight,
                            const double &timestamp) {
    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if (mbActivateLocalizationMode) {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while (!mpLocalMapper->isStopped()) {
                usleep(1000);
            }

            // Deactivate local mapping, only localize the camera.
            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }

        if (mbDeactivateLocalizationMode) {
            // Re-enable local mapping
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
        unique_lock<mutex> lock(mMutexReset);
        if (mbReset) {
            mpTracker->Reset();
            mbReset = false;
        }
    }

    std::pair<cv::Mat, cv::Mat> resizedImages = resizeImages(imLeft, imRight);
    cv::Mat Tcw = mpTracker->GrabImageStereo(
      resizedImages.first, resizedImages.second, timestamp);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPoints = mpTracker->mCurrentFrame.mvKeysSemantic;

    return Tcw;
}

void System::ActivateLocalizationMode() {
    unique_lock<mutex> lock(mMutexMode);
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode() {
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}

bool System::MapChanged() {
    static int n = 0;
    int curn = mpMap->GetLastBigChangeIdx();
    if (n < curn) {
        n = curn;
        return true;
    } else
        return false;
}

void System::Reset() {
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

void System::Shutdown() {
    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();
    if (mpViewer) {
        mpViewer->RequestFinish();
        while (!mpViewer->isFinished())
            usleep(5000);
    }

    // Wait until all thread have effectively stopped
    while (!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() ||
           mpLoopCloser->isRunningGBA()) {
        usleep(5000);
    }

    if (mpViewer)
        pangolin::BindToContext("ORB-SLAM2: Map Viewer");

    // Save number of KFs and map points to file.
    std::ofstream f;
    f.open("keyframes_points.txt", std::ofstream::out | std::ofstream::trunc);


    f << "Final number of keyframes: " << mpMap->KeyFramesInMap() << std::endl;
    f << "Final number of map points: " << mpMap->MapPointsInMap() << std::endl;

    f.close();
}

void System::SaveTrajectoryKITTI(const string &filename) {
    std::cout << std::endl
              << "Saving camera trajectory to " << filename << " ..."
              << std::endl;
    if (mSensor == MONOCULAR) {
        std::cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular."
                  << std::endl;
        return;
    }

    std::vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
    std::sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is
    // optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative
    // transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT)
    // and a flag which is true when tracking failed (lbL).
    std::list<KeyFrame *>::iterator lRit = mpTracker->mlpReferences.begin();
    std::list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    for (std::list<cv::Mat>::iterator
           lit = mpTracker->mlRelativeFramePoses.begin(),
           lend = mpTracker->mlRelativeFramePoses.end();
         lit != lend;
         lit++, lRit++, lT++) {
        KeyFrame *pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

        while (pKF->isBad()) {
            //  std::cout << "bad parent" << std::endl;
            Trw = Trw * pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw * pKF->GetPose() * Two;

        cv::Mat Tcw = (*lit) * Trw;
        cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
        cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);

        f << setprecision(9) << Rwc.at<float>(0, 0) << " "
          << Rwc.at<float>(0, 1) << " " << Rwc.at<float>(0, 2) << " "
          << twc.at<float>(0) << " " << Rwc.at<float>(1, 0) << " "
          << Rwc.at<float>(1, 1) << " " << Rwc.at<float>(1, 2) << " "
          << twc.at<float>(1) << " " << Rwc.at<float>(2, 0) << " "
          << Rwc.at<float>(2, 1) << " " << Rwc.at<float>(2, 2) << " "
          << twc.at<float>(2) << std::endl;
    }
    f.close();
    std::cout << std::endl << "trajectory saved!" << std::endl;
}

int System::GetTrackingState() {
    unique_lock<mutex> lock(mMutexState);
    return mTrackingState;
}

std::vector<MapPoint *> System::GetTrackedMapPoints() {
    unique_lock<mutex> lock(mMutexState);
    return mTrackedMapPoints;
}

std::vector<cv::KeyPoint> System::GetTrackedKeyPointsUn() {
    unique_lock<mutex> lock(mMutexState);
    return mTrackedKeyPoints;
}
}  // namespace SIVO
