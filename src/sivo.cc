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

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <opencv2/core/core.hpp>

#include <include/orbslam/System.h>

void loadImages(const std::string &strPathToSequence,
                std::vector<std::string> &vstrImageLeft,
                std::vector<std::string> &vstrImageRight,
                std::vector<double> &vTimestamps);

int main(int argc, char **argv) {
    if (argc != 6) {
        std::cerr
          << std::endl
          << "Usage: ./stereo_kitti path_to_vocabulary path_to_settings "
             "path_to_model_prototxt path_to_model_weights path_to_sequence"
          << std::endl;
        return 1;
    }

    // Retrieve paths to images
    std::vector<std::string> vstrImageLeft;
    std::vector<std::string> vstrImageRight;
    std::vector<double> vTimestamps;

    loadImages(
      std::string(argv[5]), vstrImageLeft, vstrImageRight, vTimestamps);

    const int nImages = static_cast<int>(vstrImageLeft.size());

    // Create SLAM system. It initializes all system threads and gets ready to
    // process frames.
    SIVO::System SLAM(
      argv[1], argv[2], argv[3], argv[4], SIVO::System::STEREO, true);

    // Vector for tracking time statistics
    std::vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    std::cout << std::endl << "-------" << std::endl;
    std::cout << "Start processing sequence ..." << std::endl;
    std::cout << "Images in the sequence: " << nImages << std::endl
              << std::endl;

    // Main loop
    cv::Mat imLeft, imRight;
    for (int ni = 0; ni < nImages; ni++) {
        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni], CV_LOAD_IMAGE_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni], CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if (imLeft.empty()) {
            std::cerr << std::endl
                      << "Failed to load image at: "
                      << std::string(vstrImageLeft[ni]) << std::endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 =
          std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 =
          std::chrono::monotonic_clock::now();
#endif

        // Pass the images to the SLAM system
        SLAM.TrackStereo(imLeft, imRight, tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 =
          std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 =
          std::chrono::monotonic_clock::now();
#endif

        double ttrack =
          std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count();

        vTimesTrack[ni] = static_cast<float>(ttrack);

        // Wait to load the next frame
        double T = 0;
        if (ni < nImages - 1)
            T = vTimestamps[ni + 1] - tframe;
        else if (ni > 0)
            T = tframe - vTimestamps[ni - 1];

        if (ttrack < T)
            usleep(static_cast<useconds_t>((T - ttrack) * 1e6));
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for (int ni = 0; ni < nImages; ni++) {
        totaltime += vTimesTrack[ni];
    }
    cout << "-------" << std::endl << std::endl;
    cout << "median tracking time: " << vTimesTrack[nImages / 2] << std::endl;
    cout << "mean tracking time: " << totaltime / nImages << std::endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryKITTI("CameraTrajectory.txt");

    return 0;
}

void loadImages(const std::string &strPathToSequence,
                std::vector<std::string> &vstrImageLeft,
                std::vector<std::string> &vstrImageRight,
                std::vector<double> &vTimestamps) {
    std::ifstream fTimes;
    std::string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while (!fTimes.eof()) {
        std::string s;
        getline(fTimes, s);
        if (!s.empty()) {
            std::stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    std::string strPrefixLeft = strPathToSequence + "/image_2/";
    std::string strPrefixRight = strPathToSequence + "/image_3/";

    const int nTimes = static_cast<int>(vTimestamps.size());
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for (int i = 0; i < nTimes; i++) {
        std::stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
    }
}
