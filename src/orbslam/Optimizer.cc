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

#include "include/orbslam/Optimizer.h"

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>

#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include "include/orbslam/Converter.h"

#include <mutex>


namespace SIVO {

void Optimizer::GlobalBundleAdjustment(Map *pMap,
                                       int nIterations,
                                       bool *pbStopFlag,
                                       const unsigned long nLoopKF,
                                       const bool bRobust) {
    std::vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
    std::vector<MapPoint *> vpMP = pMap->GetAllMapPoints();
    BundleAdjustment(vpKFs, vpMP, nIterations, pbStopFlag, nLoopKF, bRobust);
}

void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs,
                                 const vector<MapPoint *> &vpMP,
                                 int nIterations,
                                 bool *pbStopFlag,
                                 const unsigned long nLoopKF,
                                 const bool bRobust) {
    std::vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    // Set up optimizer.
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

    linearSolver =
      new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    auto *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    auto *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;


    // Set KeyFrame vertices
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame *pKF = vpKFs[i];
        if (pKF->isBad()) {
            continue;
        }

        // Create pose estimates and add to the graph.
        auto *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
        vSE3->setId(static_cast<int>(pKF->mnId));

        // Keep the first frame fixed.
        vSE3->setFixed(pKF->mnId == 0);

        // Add edge, update current max KF Id.
        optimizer.addVertex(vSE3);
        if (pKF->mnId > maxKFid) {
            maxKFid = pKF->mnId;
        }
    }

    const float thHuber2D = std::sqrt(5.991f);
    const float thHuber3D = std::sqrt(7.815f);

    // Set MapPoint vertices
    for (size_t i = 0; i < vpMP.size(); i++) {
        MapPoint *pMP = vpMP[i];

        if (pMP->isBad()) {
            continue;
        }

        // Create point vertex and add to graph.
        auto *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        const auto id = static_cast<int>(pMP->mnId + maxKFid) + 1;
        vPoint->setId(id);

        // Marginalize points in order to speed up matrix inversion.
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const std::map<KeyFrame *, size_t> observations =
          pMP->GetObservations();

        int nEdges = 0;
        // SET EDGES
        for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin();
             mit != observations.end();
             mit++) {
            KeyFrame *pKF = mit->first;
            if (pKF->isBad() || pKF->mnId > maxKFid) {
                continue;
            }

            nEdges++;

            const cv::KeyPoint &kp = pKF->mvKeysSemantic[mit->second];

            // Add monocular observations.
            if (pKF->mvRight[mit->second] < 0) {
                // Observation for monocular is only x and y in left image.
                Eigen::Matrix<double, 2, 1> obs;
                obs << kp.pt.x, kp.pt.y;

                // Create and set edge between KF and map point.
                auto *e = new g2o::EdgeSE3ProjectXYZ();

                e->setVertex(0,
                             dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                               optimizer.vertex(id)));
                e->setVertex(1,
                             dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                               optimizer.vertex(static_cast<int>(pKF->mnId))));

                e->setMeasurement(obs);

                // Set measurement uncertainty
                const float &invSigma2 = pKF->mvInvLevelSigma2[kp.octave];
                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                // Set up robust kernel
                if (bRobust) {
                    auto *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                // Add camera intrinsics to the edge.
                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;

                optimizer.addEdge(e);
            } else {
                // Stereo observation, measurement is x, y pixel in left image
                // and the horzontal pixel in the right image.
                Eigen::Matrix<double, 3, 1> obs;
                const float kp_r = pKF->mvRight[mit->second];
                obs << kp.pt.x, kp.pt.y, kp_r;

                // Create edge between KF and map point
                auto *e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0,
                             dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                               optimizer.vertex(id)));
                e->setVertex(1,
                             dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                               optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);

                // Set uncertainty for measurement.
                const float &invSigma2 = pKF->mvInvLevelSigma2[kp.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                e->setInformation(Info);

                // Set up robust kernel if desired.
                if (bRobust) {
                    auto *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                // Set camera intrinsics and baseline.
                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);
            }
        }

        // If no edges, remove the map point vertex.
        if (nEdges == 0) {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i] = true;
        } else {
            vbNotIncludedMP[i] = false;
        }
    }

    // Optimize!
    std::cout << "Global BA Optimization!" << std::endl;
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // Keyframes
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame *pKF = vpKFs[i];
        if (pKF->isBad()) {
            continue;
        }

        // Recover pose.
        auto *vSE3 = dynamic_cast<g2o::VertexSE3Expmap *>(
          optimizer.vertex(static_cast<int>(pKF->mnId)));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        if (nLoopKF == 0) {
            pKF->SetPose(Converter::toCvMat(SE3quat));
        } else {
            pKF->mTcwGBA.create(4, 4, CV_32F);
            Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
            pKF->mnBAGlobalForKF = nLoopKF;
        }
    }

    // Points
    for (size_t i = 0; i < vpMP.size(); i++) {
        if (vbNotIncludedMP[i]) {
            continue;
        }

        MapPoint *pMP = vpMP[i];

        if (pMP->isBad()) {
            continue;
        }

        auto *vPoint = dynamic_cast<g2o::VertexSBAPointXYZ *>(
          optimizer.vertex(static_cast<int>(pMP->mnId + maxKFid) + 1));

        if (nLoopKF == 0) {
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        } else {
            pMP->mPosGBA.create(3, 1, CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }
}

int Optimizer::PoseOptimization(Frame *pFrame) {
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

    linearSolver =
      new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>();

    auto *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    auto *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences = 0;

    // Set Frame vertex
    auto *vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->numSemanticKeys;

    std::vector<g2o::EdgeSE3ProjectXYZOnlyPose *> vpEdgesMono;
    std::vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(static_cast<unsigned long>(N));
    vnIndexEdgeMono.reserve(static_cast<unsigned long>(N));

    std::vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> vpEdgesStereo;
    std::vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(static_cast<unsigned long>(N));
    vnIndexEdgeStereo.reserve(static_cast<unsigned long>(N));

    const float deltaMono = std::sqrt(5.991f);
    const float deltaStereo = std::sqrt(7.815f);

    {
        // Prevent modification to map points.
        std::unique_lock<std::mutex> lock(MapPoint::mGlobalMutex);

        // Add map points to the optimizer.
        for (int i = 0; i < N; i++) {
            MapPoint *pMP = pFrame->mvpMapPoints[i];
            if (pMP) {
                if (pFrame->mvRight.at(i) < 0) {
                    // Monocular observation of feature
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    // Observation is pixel coordinate in left image.
                    Eigen::Matrix<double, 2, 1> obs;
                    const cv::KeyPoint &kp = pFrame->mvKeysSemantic[i];
                    obs << kp.pt.x, kp.pt.y;

                    auto *e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                    // Connect vertex with edge, and set measurement
                    e->setVertex(0,
                                 dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                   optimizer.vertex(0)));
                    e->setMeasurement(obs);

                    // Set uncertainty for measurement
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kp.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    // Create Huber Kernel for robust cost function
                    auto *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    // Insert camera intrinsic parameters
                    e->fx = pFrame->fx;
                    e->fy = pFrame->fy;
                    e->cx = pFrame->cx;
                    e->cy = pFrame->cy;

                    // Insert map point world position.
                    cv::Mat Xw = pMP->GetWorldPos();
                    e->Xw[0] = Xw.at<float>(0);
                    e->Xw[1] = Xw.at<float>(1);
                    e->Xw[2] = Xw.at<float>(2);

                    // Add edge to optimizer and maintenance variables
                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                } else {
                    // Stereo observation of feature
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    // Observation is pixel in the first image, and right
                    // image horizontal coordinate.
                    Eigen::Matrix<double, 3, 1> obs;
                    const cv::KeyPoint &kp = pFrame->mvKeysSemantic[i];
                    const float &kp_r = pFrame->mvRight[i];
                    obs << kp.pt.x, kp.pt.y, kp_r;

                    auto *e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                    // Connect vertices with edge, and set measurement
                    e->setVertex(0,
                                 dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                   optimizer.vertex(0)));
                    e->setMeasurement(obs);

                    // Set uncertainty for measurement.
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kp.octave];
                    Eigen::Matrix3d Info =
                      Eigen::Matrix3d::Identity() * invSigma2;
                    e->setInformation(Info);

                    // Create Huber Kernel for robust cost function
                    auto *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaStereo);

                    // Insert camera intrinsic parameters and baseline
                    e->fx = pFrame->fx;
                    e->fy = pFrame->fy;
                    e->cx = pFrame->cx;
                    e->cy = pFrame->cy;
                    e->bf = pFrame->mbf;

                    // Insert map point world position.
                    cv::Mat Xw = pMP->GetWorldPos();
                    e->Xw[0] = Xw.at<float>(0);
                    e->Xw[1] = Xw.at<float>(1);
                    e->Xw[2] = Xw.at<float>(2);

                    // Add edge to optimizer and maintenance variables
                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vnIndexEdgeStereo.push_back(static_cast<unsigned long>(i));
                }
            }
        }
    }

    // Not enough correspondences, do not include.
    if (nInitialCorrespondences < 3) {
        return 0;
    }

    // We perform 4 optimizations, after each optimization we classify
    // observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they
    // can be classified as inliers again.
    const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
    const int its[4] = {10, 10, 10, 10};

    int nBad = 0;
    for (size_t it = 0; it < 4; it++) {
        // Reset estimates
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        optimizer.setVerbose(false);
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad = 0;

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if (pFrame->mvbOutlier[idx]) {
                e->computeError();
            }

            // Perform chi2 test to determine whether point is an outlier
            // 95% confidence from the chi2 test.
            const auto chi2 = static_cast<float>(e->chi2());

            if (chi2 > chi2Stereo[it]) {
                pFrame->mvbOutlier[idx] = true;

                // Optimizer will not include edges >= level 1.
                // Effectively only using a subset of the edges.
                e->setLevel(1);
                nBad++;
            } else {
                e->setLevel(0);
                pFrame->mvbOutlier[idx] = false;
            }

            // After two iterations, we may not need the robust kernel?
            // Not really sure what the point of this is.
            if (it == 2) {
                e->setRobustKernel(nullptr);
            }
        }

        if (optimizer.edges().size() < 10) {
            break;
        }
    }

    // Recover optimized pose
    auto *vSE3_recov =
      dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose);

    // Recover optimized covariance value and store
    // Unlike the local mapping and global BA, we can extract covariances each time.
    g2o::SparseBlockMatrixXd spinv;
    if (optimizer.computeMarginals(spinv, vSE3_recov)) {
        // Extract block and convert to cv::Mat
        Eigen::MatrixXd *Sigmacw = spinv.block(0, 0);
        pFrame->SetCovariance(*Sigmacw);
    }

    // Return number of inliers
    return nInitialCorrespondences - nBad;
}

void Optimizer::LocalBundleAdjustment(KeyFrame *pKF,
                                      bool *pbStopFlag,
                                      Map *pMap) {
    // Local KeyFrames: First Breath Search from Current Keyframe
    std::list<KeyFrame *> lLocalKeyFrames;

    lLocalKeyFrames.emplace_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    // Extract all covisible keyframes, and keep for BA if frame is good.
    const std::vector<KeyFrame *> vCovisibleKFs =
      pKF->GetVectorCovisibleKeyFrames();

    for (const auto &covisible_kf : vCovisibleKFs) {
        KeyFrame *pKFi = covisible_kf;
        pKFi->mnBALocalForKF = pKF->mnId;

        if (!pKFi->isBad()) {
            lLocalKeyFrames.push_back(pKFi);
        }
    }

    // Local MapPoints seen in Local KeyFrames
    std::list<MapPoint *> lLocalMapPoints;
    for (auto lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end();
         lit != lend;
         lit++) {
        // Extract map point matches
        std::vector<MapPoint *> vpMPs = (*lit)->GetMapPointMatches();
        for (auto vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++) {
            MapPoint *pMP = *vit;
            if (pMP) {
                if (!pMP->isBad()) {
                    if (pMP->mnBALocalForKF != pKF->mnId) {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF = pKF->mnId;
                    }
                }
            }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not
    // Local Keyframes
    std::list<KeyFrame *> lFixedKFs;
    for (auto lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end();
         lit != lend;
         lit++) {
        // Extract all keyframes the map point was seen in.
        std::map<KeyFrame *, size_t> observations = (*lit)->GetObservations();

        // Add to fixed keyframes if point was not seen in the local frame,
        // and if the fixed ID is set, but not the current frame.
        for (auto mit = observations.begin(), mend = observations.end();
             mit != mend;
             mit++) {
            KeyFrame *pKFi = mit->first;

            if (pKFi->mnBALocalForKF != pKF->mnId &&
                pKFi->mnBAFixedForKF != pKF->mnId) {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if (!pKFi->isBad()) {
                    lFixedKFs.push_back(pKFi);
                }
            }
        }
    }

    // Set up optimizer
    g2o::SparseOptimizer optimizer;

    // Use cholmod solver in order to extract marginal covariances.
    auto *linearSolver =
      new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>();

    auto *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    auto *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if (pbStopFlag) {
        optimizer.setForceStopFlag(pbStopFlag);
    }

    unsigned long maxKFid = 0;

    // If the first frame is present, set this flag true.
    // Required for marginal covariance recovery, due to a mismatch between
    // the size of the local frames and
    bool first_frame = false;

    // Set Local KeyFrame vertices
    for (auto lit = lLocalKeyFrames.begin(); lit != lLocalKeyFrames.end();
         lit++) {
        KeyFrame *pKFi = *lit;

        if (pKFi->mnId == 0) {
            first_frame = true;
        }

        // Create new pose vertex
        auto *vSE3 = new g2o::VertexSE3Expmap();

        // Set estimate and add it to the graph.
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(static_cast<int>(pKFi->mnId));
        vSE3->setFixed(pKFi->mnId == 0);
        optimizer.addVertex(vSE3);

        if (pKFi->mnId > maxKFid) {
            maxKFid = pKFi->mnId;
        }
    }

    // Set Fixed KeyFrame vertices
    for (auto lit = lFixedKFs.begin(); lit != lFixedKFs.end(); lit++) {
        KeyFrame *pKFi = *lit;

        // Create new pose vertex
        auto *vSE3 = new g2o::VertexSE3Expmap();

        // Set estimate and add it to the graph.
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(static_cast<int>(pKFi->mnId));
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > maxKFid) {
            maxKFid = pKFi->mnId;
        }
    }

    // Set MapPoint vertices
    const unsigned long nExpectedSize =
      (lLocalKeyFrames.size() + lFixedKFs.size()) * lLocalMapPoints.size();

    // Monocular edges, keyframes and map points
    std::vector<g2o::EdgeSE3ProjectXYZ *> vpEdgesMono;
    std::vector<KeyFrame *> vpEdgeKFMono;
    std::vector<MapPoint *> vpMapPointEdgeMono;

    // Stereo edges, keyframes and map points
    std::vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo;
    std::vector<KeyFrame *> vpEdgeKFStereo;
    std::vector<MapPoint *> vpMapPointEdgeStereo;

    // Preallocate vectors for mono and stereo
    vpEdgesMono.reserve(nExpectedSize);
    vpEdgeKFMono.reserve(nExpectedSize);
    vpMapPointEdgeMono.reserve(nExpectedSize);
    vpEdgesStereo.reserve(nExpectedSize);
    vpEdgeKFStereo.reserve(nExpectedSize);
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    // Set chi2 thresholds for 95% confidence
    const float thHuberMono = sqrt(5.991f);
    const float thHuberStereo = sqrt(7.815f);

    // Add local map point edges.
    for (auto lit = lLocalMapPoints.begin(); lit != lLocalMapPoints.end();
         lit++) {
        MapPoint *pMP = *lit;

        auto *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        int id = static_cast<int>(pMP->mnId + maxKFid) + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const std::map<KeyFrame *, size_t> observations =
          pMP->GetObservations();

        // Set edges
        for (auto mit = observations.begin(); mit != observations.end();
             mit++) {
            KeyFrame *pKFi = mit->first;

            if (!pKFi->isBad()) {
                const cv::KeyPoint &kp = pKFi->mvKeysSemantic[mit->second];

                // Monocular observation
                if (pKFi->mvRight[mit->second] < 0) {
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kp.pt.x, kp.pt.y;

                    auto *e = new g2o::EdgeSE3ProjectXYZ();

                    // Connect edge from keyframe to map point
                    e->setVertex(0,
                                 dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                   optimizer.vertex(id)));
                    e->setVertex(
                      1,
                      dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                        optimizer.vertex(static_cast<int>(pKFi->mnId))));

                    // Set measurement and uncertainty
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kp.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    // Create kernel for robust cost function
                    auto *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    // Set camera intrinsics
                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;

                    // Add edge to optimizer and maintenance variables
                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                } else {
                    // Stereo observation is the pixel location in the left
                    // image, and the horizontal coordinate in the right
                    Eigen::Matrix<double, 3, 1> obs;
                    const float kp_r = pKFi->mvRight[mit->second];
                    obs << kp.pt.x, kp.pt.y, kp_r;

                    auto *e = new g2o::EdgeStereoSE3ProjectXYZ();

                    // Connect edge from keyframe to map point
                    e->setVertex(0,
                                 dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                   optimizer.vertex(id)));
                    e->setVertex(
                      1,
                      dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                        optimizer.vertex(static_cast<int>(pKFi->mnId))));

                    // Set edge and uncertainty
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kp.octave];
                    Eigen::Matrix3d Info =
                      Eigen::Matrix3d::Identity() * invSigma2;
                    e->setInformation(Info);

                    // Create kernel for robust cost function
                    auto *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    // Set camera intrinsics and baseline
                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    // Add edge to optimizer and maintenance variables
                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
            }
        }
    }

    if (pbStopFlag) {
        if (*pbStopFlag) {
            return;
        }
    }

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    bool bDoMore = true;

    if (pbStopFlag) {
        if (*pbStopFlag) {
            bDoMore = false;
        }
    }

    if (bDoMore) {
        // Check inlier observations
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
            // Check monocular observations, see if within 95% confidence
            g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
            MapPoint *pMP = vpMapPointEdgeMono[i];

            if (pMP->isBad()) {
                continue;
            }

            // If outlier
            if (e->chi2() > 5.991 || !e->isDepthPositive()) {
                // Optimizer will not include edges >= level 1.
                // Effectively only using a subset of the edges.
                e->setLevel(1);
            }

            // If an outlier, it's been removed.
            // Else, it's an inlier, and we no longer need the robust kernel.
            e->setRobustKernel(nullptr);
        }

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
            // Check stereo observations, see if within 95% confidence
            g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
            MapPoint *pMP = vpMapPointEdgeStereo[i];

            if (pMP->isBad()) {
                continue;
            }

            // If outlier
            if (e->chi2() > 7.815 || !e->isDepthPositive()) {
                // Optimizer will not include edges >= level 1.
                // Effectively only using a subset of the edges.
                e->setLevel(1);
            }

            // If an outlier, it's been removed.
            // Else, it's an inlier, and we no longer need the robust kernel.
            e->setRobustKernel(nullptr);
        }

        // Optimize again without the outliers
        optimizer.initializeOptimization(0);
        optimizer.optimize(10);
    }

    // After optimization, check outliers again. This time delete.
    std::vector<std::pair<KeyFrame *, MapPoint *>> vToErase;
    vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());

    // Check to see if observations are outliers. Same as above.
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
        // Check monocular
        g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
        MapPoint *pMP = vpMapPointEdgeMono[i];

        if (pMP->isBad()) {
            continue;
        }

        if (e->chi2() > 5.991 || !e->isDepthPositive()) {
            // If outlier, mark for erase.
            KeyFrame *pKFi = vpEdgeKFMono[i];
            vToErase.emplace_back(std::make_pair(pKFi, pMP));
        }
    }

    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
        // Check stereo observations
        g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
        MapPoint *pMP = vpMapPointEdgeStereo[i];

        if (pMP->isBad()) {
            continue;
        }

        if (e->chi2() > 7.815 || !e->isDepthPositive()) {
            // If outlier, mark to erase.
            KeyFrame *pKFi = vpEdgeKFStereo[i];
            vToErase.emplace_back(std::make_pair(pKFi, pMP));
        }
    }

    // Get Map Mutex
    std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);

    // Erase map point observations from the keyframe.
    if (!vToErase.empty()) {
        for (const auto &erase : vToErase) {
            KeyFrame *pKFi = erase.first;
            MapPoint *pMPi = erase.second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data

    // Block to extract marginal covariance from.
    int block_id = lLocalKeyFrames.size() - 1;

    // If first frame is present, we need to remove one more block.
    // This is because it is set fixed, and is not present in the
    // marginal covariance recovery, even though it is in the local frames.
    if (first_frame) {
        --block_id;
    }

    // Extract each local keyframe
    // We only need the latest marginal block in order to select features for the next state.
    for (auto lit = lLocalKeyFrames.begin(); lit != lLocalKeyFrames.end();
         lit++) {
        // Recover keyframes
        KeyFrame *pKFi = *lit;

        // Extract pose.
        auto *vSE3 = dynamic_cast<g2o::VertexSE3Expmap *>(
          optimizer.vertex(static_cast<int>(pKFi->mnId)));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKFi->SetPose(Converter::toCvMat(SE3quat));

        if (pKFi->mnId == pKF->mnId)  {
            // Extract pose covariance
            g2o::SparseBlockMatrixXd spinv;

            if (optimizer.computeMarginals(spinv, vSE3)) {
                // Block ID in the marginal recovery depends on the ID of the
                // vertex in the graph
                Eigen::MatrixXd *Sigmacw = spinv.block(block_id, block_id);
                pKF->SetCovariance(*Sigmacw);
            }
        } else {
            --block_id;
        }

    }

    // Points
    for (auto lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end();
         lit != lend;
         lit++) {
        MapPoint *pMP = *lit;

        // Extract point information
        auto *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(
          optimizer.vertex(static_cast<int>(pMP->mnId + maxKFid) + 1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }
}

void Optimizer::OptimizeEssentialGraph(
  Map *pMap,
  KeyFrame *pLoopKF,
  KeyFrame *pCurKF,

  const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
  const LoopClosing::KeyFrameAndPose &CorrectedSim3,

  const std::map<KeyFrame *, std::set<KeyFrame *>> &LoopConnections,

  const bool &bFixScale) {
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);

    g2o::BlockSolver_7_3::LinearSolverType *linearSolver =
      new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();

    auto *solver_ptr = new g2o::BlockSolver_7_3(linearSolver);
    auto *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    const std::vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
    const std::vector<MapPoint *> vpMPs = pMap->GetAllMapPoints();

    const auto nMaxKFid = static_cast<unsigned int>(pMap->GetMaxKFid());

    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw(nMaxKFid + 1);
    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vCorrectedSwc(
      nMaxKFid + 1);
    vector<g2o::VertexSim3Expmap *> vpVertices(nMaxKFid + 1);

    const int minFeat = 100;

    // Set KeyFrame vertices
    for (size_t i = 0, iend = vpKFs.size(); i < iend; i++) {
        KeyFrame *pKF = vpKFs[i];
        if (pKF->isBad()) {
            continue;
        }

        auto *VSim3 = new g2o::VertexSim3Expmap();

        const auto nIDi = static_cast<int>(pKF->mnId);

        LoopClosing::KeyFrameAndPose::const_iterator it =
          CorrectedSim3.find(pKF);

        if (it != CorrectedSim3.end()) {
            vScw[nIDi] = it->second;
            VSim3->setEstimate(it->second);
        } else {
            Eigen::Matrix<double, 3, 3> Rcw =
              Converter::toMatrix3d(pKF->GetRotation());
            Eigen::Matrix<double, 3, 1> tcw =
              Converter::toVector3d(pKF->GetTranslation());
            g2o::Sim3 Siw(Rcw, tcw, 1.0);
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }

        if (pKF == pLoopKF) {
            VSim3->setFixed(true);
        }


        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = bFixScale;

        optimizer.addVertex(VSim3);

        vpVertices[nIDi] = VSim3;
    }


    std::set<std::pair<long unsigned int, long unsigned int>> sInsertedEdges;

    const Eigen::Matrix<double, 7, 7> matLambda =
      Eigen::Matrix<double, 7, 7>::Identity();

    // Set Loop edges
    for (map<KeyFrame *, std::set<KeyFrame *>>::const_iterator
           mit = LoopConnections.begin(),
           mend = LoopConnections.end();
         mit != mend;
         mit++) {
        KeyFrame *pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        const std::set<KeyFrame *> &spConnections = mit->second;
        const g2o::Sim3 Siw = vScw[nIDi];
        const g2o::Sim3 Swi = Siw.inverse();

        for (std::set<KeyFrame *>::const_iterator sit = spConnections.begin(),
                                                  send = spConnections.end();
             sit != send;
             sit++) {
            const long unsigned int nIDj = (*sit)->mnId;
            if ((nIDi != pCurKF->mnId || nIDj != pLoopKF->mnId) &&
                pKF->GetWeight(*sit) < minFeat)
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];
            const g2o::Sim3 Sji = Sjw * Swi;

            auto *e = new g2o::EdgeSim3();
            e->setVertex(1,
                         dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                           optimizer.vertex(nIDj)));
            e->setVertex(0,
                         dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                           optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;

            optimizer.addEdge(e);

            sInsertedEdges.insert(make_pair(min(nIDi, nIDj), max(nIDi, nIDj)));
        }
    }

    // Set normal edges
    for (size_t i = 0, iend = vpKFs.size(); i < iend; i++) {
        KeyFrame *pKF = vpKFs[i];

        const auto nIDi = static_cast<int>(pKF->mnId);

        g2o::Sim3 Swi;

        LoopClosing::KeyFrameAndPose::const_iterator iti =
          NonCorrectedSim3.find(pKF);

        if (iti != NonCorrectedSim3.end()) {
            Swi = (iti->second).inverse();
        } else {
            Swi = vScw[nIDi].inverse();
        }

        KeyFrame *pParentKF = pKF->GetParent();

        // Spanning tree edge
        if (pParentKF) {
            auto nIDj = static_cast<int>(pParentKF->mnId);

            g2o::Sim3 Sjw;

            LoopClosing::KeyFrameAndPose::const_iterator itj =
              NonCorrectedSim3.find(pParentKF);

            if (itj != NonCorrectedSim3.end())
                Sjw = itj->second;
            else
                Sjw = vScw[nIDj];

            g2o::Sim3 Sji = Sjw * Swi;

            auto *e = new g2o::EdgeSim3();
            e->setVertex(1,
                         dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                           optimizer.vertex(nIDj)));
            e->setVertex(0,
                         dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                           optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // Loop edges
        const std::set<KeyFrame *> sLoopEdges = pKF->GetLoopEdges();
        for (std::set<KeyFrame *>::const_iterator sit = sLoopEdges.begin(),
                                                  send = sLoopEdges.end();
             sit != send;
             sit++) {
            KeyFrame *pLKF = *sit;
            if (pLKF->mnId < pKF->mnId) {
                g2o::Sim3 Slw;

                LoopClosing::KeyFrameAndPose::const_iterator itl =
                  NonCorrectedSim3.find(pLKF);

                if (itl != NonCorrectedSim3.end())
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->mnId];

                g2o::Sim3 Sli = Slw * Swi;
                auto *el = new g2o::EdgeSim3();
                el->setVertex(
                  1,
                  dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                    optimizer.vertex(static_cast<int>(pLKF->mnId))));
                el->setVertex(0,
                              dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        // Covisibility graph edges
        const std::vector<KeyFrame *> vpConnectedKFs =
          pKF->GetCovisiblesByWeight(minFeat);
        for (std::vector<KeyFrame *>::const_iterator vit =
               vpConnectedKFs.begin();
             vit != vpConnectedKFs.end();
             vit++) {
            KeyFrame *pKFn = *vit;
            if (pKFn && pKFn != pParentKF && !pKF->hasChild(pKFn) &&
                !sLoopEdges.count(pKFn)) {
                if (!pKFn->isBad() && pKFn->mnId < pKF->mnId) {
                    if (sInsertedEdges.count(
                          std::make_pair(min(pKF->mnId, pKFn->mnId),
                                         max(pKF->mnId, pKFn->mnId)))) {
                        continue;
                    }

                    g2o::Sim3 Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn =
                      NonCorrectedSim3.find(pKFn);

                    if (itn != NonCorrectedSim3.end())
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->mnId];

                    g2o::Sim3 Sni = Snw * Swi;

                    auto *en = new g2o::EdgeSim3();
                    en->setVertex(1,
                                  dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                    optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0,
                                  dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                    optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }
    }

    // Optimize!
    std::cout << "Optimize Essential Graph!" << std::endl;
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame *pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        auto *VSim3 =
          dynamic_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw = VSim3->estimate();
        vCorrectedSwc[nIDi] = CorrectedSiw.inverse();
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        double s = CorrectedSiw.scale();

        eigt *= (1. / s);  //[R t/s;0 1]

        cv::Mat Tiw = Converter::toCvSE3(eigR, eigt);

        pKFi->SetPose(Tiw);
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and
    // transform back with optimized pose
    for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
        MapPoint *pMP = vpMPs[i];

        if (pMP->isBad())
            continue;

        int nIDr;
        if (pMP->mnCorrectedByKF == pCurKF->mnId) {
            nIDr = pMP->mnCorrectedReference;
        } else {
            KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }

        g2o::Sim3 Srw = vScw[nIDr];
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        cv::Mat P3Dw = pMP->GetWorldPos();
        Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw =
          correctedSwr.map(Srw.map(eigP3Dw));

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMP->SetWorldPos(cvCorrectedP3Dw);

        pMP->UpdateNormalAndDepth();
    }
}

int Optimizer::OptimizeSim3(KeyFrame *pKF1,
                            KeyFrame *pKF2,
                            vector<MapPoint *> &vpMatches1,
                            g2o::Sim3 &g2oS12,
                            const float th2,
                            const bool bFixScale) {
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;

    linearSolver =
      new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    auto *solver_ptr = new g2o::BlockSolverX(linearSolver);

    auto *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Calibration
    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    // Camera poses
    const cv::Mat R1w = pKF1->GetRotation();
    const cv::Mat t1w = pKF1->GetTranslation();
    const cv::Mat R2w = pKF2->GetRotation();
    const cv::Mat t2w = pKF2->GetTranslation();

    // Set Sim3 vertex
    auto *vSim3 = new g2o::VertexSim3Expmap();
    vSim3->_fix_scale = bFixScale;
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->_principle_point1[0] = K1.at<float>(0, 2);
    vSim3->_principle_point1[1] = K1.at<float>(1, 2);
    vSim3->_focal_length1[0] = K1.at<float>(0, 0);
    vSim3->_focal_length1[1] = K1.at<float>(1, 1);
    vSim3->_principle_point2[0] = K2.at<float>(0, 2);
    vSim3->_principle_point2[1] = K2.at<float>(1, 2);
    vSim3->_focal_length2[0] = K2.at<float>(0, 0);
    vSim3->_focal_length2[1] = K2.at<float>(1, 1);
    optimizer.addVertex(vSim3);

    // Set MapPoint vertices
    const auto N = static_cast<int>(vpMatches1.size());
    const std::vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
    std::vector<g2o::EdgeSim3ProjectXYZ *> vpEdges12;
    std::vector<g2o::EdgeInverseSim3ProjectXYZ *> vpEdges21;
    std::vector<size_t> vnIndexEdge;

    vnIndexEdge.reserve(static_cast<unsigned long>(2 * N));
    vpEdges12.reserve(static_cast<unsigned long>(2 * N));
    vpEdges21.reserve(static_cast<unsigned long>(2 * N));

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;

    for (int i = 0; i < N; i++) {
        if (!vpMatches1[i]) {
            continue;
        }

        MapPoint *pMP1 = vpMapPoints1[i];
        MapPoint *pMP2 = vpMatches1[i];

        const int id1 = 2 * i + 1;
        const int id2 = 2 * (i + 1);

        const int i2 = pMP2->GetIndexInKeyFrame(pKF2);

        if (pMP1 && pMP2) {
            if (!pMP1->isBad() && !pMP2->isBad() && i2 >= 0) {
                auto *vPoint1 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D1w = pMP1->GetWorldPos();
                cv::Mat P3D1c = R1w * P3D1w + t1w;
                vPoint1->setEstimate(Converter::toVector3d(P3D1c));
                vPoint1->setId(id1);
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                auto *vPoint2 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D2w = pMP2->GetWorldPos();
                cv::Mat P3D2c = R2w * P3D2w + t2w;
                vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            } else {
                continue;
            }
        } else {
            continue;
        }

        nCorrespondences++;

        // Set edge x1 = S12*X2
        Eigen::Matrix<double, 2, 1> obs1;
        const cv::KeyPoint &kp1 = pKF1->mvKeysSemantic[i];
        obs1 << kp1.pt.x, kp1.pt.y;

        // Create edge
        auto *e12 = new g2o::EdgeSim3ProjectXYZ();
        e12->setVertex(
          0,
          dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id2)));
        e12->setVertex(
          1,
          dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
        e12->setMeasurement(obs1);
        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kp1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare1);

        // Set up kernel for robust cost
        auto *rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);

        // Add edge to optimizer.
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        Eigen::Matrix<double, 2, 1> obs2;
        const cv::KeyPoint &kp2 = pKF2->mvKeysSemantic[i2];
        obs2 << kp2.pt.x, kp2.pt.y;

        auto *e21 = new g2o::EdgeInverseSim3ProjectXYZ();

        e21->setVertex(
          0,
          dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id1)));
        e21->setVertex(
          1,
          dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kp2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare2);

        // Set up kernel for robust cost.
        auto *rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);

        // Add edge to optimizer and maintenance variables
        optimizer.addEdge(e21);
        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(static_cast<unsigned long>(i));
    }

    // Optimize!
    std::cout << "Optimize sim3!" << std::endl;
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inliers
    int nBad = 0;
    for (size_t i = 0; i < vpEdges12.size(); i++) {
        g2o::EdgeSim3ProjectXYZ *e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ *e21 = vpEdges21[i];
        if (!e12 || !e21) {
            continue;
        }

        if (e12->chi2() > th2 || e21->chi2() > th2) {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx] = static_cast<MapPoint *>(nullptr);
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i] = static_cast<g2o::EdgeSim3ProjectXYZ *>(nullptr);
            vpEdges21[i] =
              static_cast<g2o::EdgeInverseSim3ProjectXYZ *>(nullptr);
            nBad++;
        }
    }

    int nMoreIterations;
    if (nBad > 0) {
        nMoreIterations = 10;
    } else {
        nMoreIterations = 5;
    }

    if (nCorrespondences - nBad < 10) {
        return 0;
    }

    // Optimize again only with inliers

    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    for (size_t i = 0; i < vpEdges12.size(); i++) {
        g2o::EdgeSim3ProjectXYZ *e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ *e21 = vpEdges21[i];
        if (!e12 || !e21)
            continue;

        if (e12->chi2() > th2 || e21->chi2() > th2) {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx] = static_cast<MapPoint *>(nullptr);
        } else
            nIn++;
    }

    // Recover optimized Sim3
    auto *vSim3_recov =
      dynamic_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(0));
    g2oS12 = vSim3_recov->estimate();

    return nIn;
}


}  // namespace ORB_SLAM
