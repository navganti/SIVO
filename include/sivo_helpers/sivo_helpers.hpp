/* Copyright (c) 2018, Waterloo Autonomous Vehicles Laboratory (WAVELab),
 * University of Waterloo. All Rights Reserved.
 *
 * ############################################################################
 *                   ____
 *                  /    \
 *     ____         \____/
 *    /    \________//      ______
 *    \____/--/      \_____/     \\
 *            \______/----/      // __        __  __  __      __  ______
 *            //          \_____//  \ \  /\  / / /  \ \ \    / / / ____/
 *       ____//                      \ \/  \/ / / /\ \ \ \  / / / /__
 *      /      \\                     \  /\  / / /  \ \ \ \/ / / /____
 *     /       //                      \/  \/ /_/    \_\ \__/ /______/
 *     \______//                     LABORATORY
 *
 * ############################################################################
 *
 * File: sivo_helpers.hpp
 * Desc: Header file containing utilities for SIVO's feature selection criteria.
 * Auth: Pranav Ganti
 *
 * Copyright (c) 2018, Waterloo Autonomous Vehicles Laboratory (WAVELab),
 * University of Waterloo. All Rights Reserved.
 *
 * ###########################################################################
 */

#ifndef ORBSLAM_SI_SIVO_HELPERS_CPP_HPP
#define ORBSLAM_SI_SIVO_HELPERS_CPP_HPP

#include <Eigen/Core>
#include <Eigen/Dense>

namespace Eigen {
using Matrix6d = Eigen::Matrix<double, 6, 6>;
}

namespace SIVO {

/// The matrix type for a state covariance only. Dim: 6x6.
using StateCovarianceType = Eigen::Matrix6d;

/// The matrix type for a state + monocular feature covariance. Dim: 8x8.
using MonoCovarianceType = Eigen::Matrix<double, 8, 8>;

/// The matrix type for a state + stereo feature covariance. Dim: 9x9.
using StereoCovarianceType = Eigen::Matrix<double, 9, 9>;

/** The matrix type for the Jacobian of the monocular projection function wrt
 * the robot pose. Dim: 2x6.
 */
using MonoProjectionPoseJacobianType = Eigen::Matrix<double, 2, 6>;

/** The matrix type for the Jacobian of the stereo projection function wrt
 * the robot pose. Dim: 3x6.
 */
using StereoProjectionPoseJacobianType = Eigen::Matrix<double, 3, 6>;

/** The matrix type for the Jacobian of the monocular projection function wrt
 * the world point. Dim: 2x3.
 */
using MonoProjectionPointJacobianType = Eigen::Matrix<double, 2, 3>;

/** The matrix type for the jacobian of the stereo projection function wrt the
 * world point. Dim: 3x3.
 */
using StereoProjectionPointJacobianType = Eigen::Matrix<double, 3, 3>;

Eigen::Matrix3d createSkewSymmetricMatrixFromVector(const Eigen::Vector3d vector);

class SIVO {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    /** Computes the jacobian of the monocular projection function with respect
   * to the pose perturbation, evaluated at the current state and for a particular
   * point.
   *
   * @param fx The x-direction focal length.
   * @param fy The y-direction focal length.
   * @param cXcp The x-coordinate of the map point (p) with respect to the
   * camera
   * frame (c), expressed in the camera frame (c).
   * @param cYcp The y-coordinate of the map point (p) with respect to the
   * camera
   * frame (c), expressed in the camera frame (c).
   * @param cZcp The z-coordinate of the map point (p) with respect to the
   * camera
   * frame (c), expressed in the camera frame (c). Cannot equal zero.
   * @return The jacobian of the monocular projection function with respect to
   * the pose perturbation.
   */
    static MonoProjectionPoseJacobianType computeMonocularJacobianPose(
      const double fx,
      const double fy,
      const double cXcp,
      const double cYcp,
      const double cZcp);

    /** Computes the jacobian of the stereo projection function with respect to
     * the pose perturbation, evaluated at the current state and for a
     * particular point.
     *
     * @param fx The x-direction focal length.
     * @param fy The y-direction focal length.
     * @param bl The baseline between the stereo cameras (m).
     * @param cXcp The x-coordinate of the map point (p) with respect to the
     * camera
     * frame (c), expressed in the camera frame (c).
     * @param cYcp The y-coordinate of the map point (p) with respect to the
     * camera
     * frame (c), expressed in the camera frame (c).
     * @param cZcp The z-coordinate of the map point (p) with respect to the
     * camera
     * frame (c), expressed in the camera frame (c). Cannot equal zero.
     * @return The jacobian of the stereo projection function with respect to
     * the pose perturbation.
     */
    static StereoProjectionPoseJacobianType computeStereoJacobianPose(
      const double fx,
      const double fy,
      const double bl,
      const double cXcp,
      const double cYcp,
      const double cZcp);

    /** Computes the jacobian of the monocular projection function with respect
     * to the world point, evaluated at the world point estimate.
     *
     * @param fx The x-direction focal length.
     * @param fy The y-direction focal length.
     * @param cXcp The x-coordinate of the map point (p) with respect to the
     * camera
     * frame (c), expressed in the camera frame (c).
     * @param cYcp The y-coordinate of the map point (p) with respect to the
     * camera
     * frame (c), expressed in the camera frame (c).
     * @param cZcp The z-coordinate of the map point (p) with respect to the
     * camera
     * frame (c), expressed in the camera frame (c).
     * @param Ccw The orientation of the world frame with respect to the camera
     * frame.
     * @return The jacobian of the monocular projection function with respect to
     * the
     * point.
     */
    static MonoProjectionPointJacobianType computeMonocularJacobianPoint(
      const double fx,
      const double fy,
      const double cXcp,
      const double cYcp,
      const double cZcp,
      const Eigen::Matrix3d Ccw);

    /** Computes the jacobian of the monocular projection function with respect
     * to the world point, evaluated at the world point estimate.
     *
     * @param fx The x-direction focal length.
     * @param fy The y-direction focal length.
     * @param bl The stereo baseline.
     * @param cXcp The x-coordinate of the map point (p) with respect to the
     * camera
     * frame (c), expressed in the camera frame (c).
     * @param cYcp The y-coordinate of the map point (p) with respect to the
     * camera
     * frame (c), expressed in the camera frame (c).
     * @param cZcp The z-coordinate of the map point (p) with respect to the
     * camera
     * frame (c), expressed in the camera frame (c).
     * @param Ccw The orientation of the world frame with respect to the camera
     * frame.
     * @return The jacobian of the stereo projection function with respect to
     * the world point.
     */
    static StereoProjectionPointJacobianType computeStereoJacobianPoint(
      const double fx,
      const double fy,
      const double bl,
      const double cXcp,
      const double cYcp,
      const double cZcp,
      const Eigen::Matrix3d Ccw);

    /** Compute the full covariance matrix for the robot state and a new
     * monocular feature.
     *
     * @param state_covariance The state covariance.
     * @param mono_jacobian The jacobian of the monocular projection function
     * wrt the robot pose.
     * @return The full covariance matrix.
     */
    static MonoCovarianceType computeMonocularCovariance(
      const StateCovarianceType &state_covariance,
      const MonoProjectionPoseJacobianType &mono_jacobian,
      const Eigen::Matrix2d &measurement_noise);

    /** Compute the full covariance matrix for the robot state and a new stereo
     * feature.
     *
     * @param state_covariance The state covariance.
     * @param stereo_jacobian The jacobian of the stereo projection function wrt
     * the robot pose.
     * @return The full covariance matrix.
     */
    static StereoCovarianceType computeStereoCovariance(
      const StateCovarianceType &state_covariance,
      const StereoProjectionPoseJacobianType &stereo_jacobian,
      const Eigen::Matrix3d &measurement_noise);

    /** Compute the mutual information between the robot state and new monocular
     * feature.
     *
     * @param mono_covariance The full covariance matrix for the state and
     * monocular feature.
     * @return The mutual information; the amount of information added to the
     * state by measuring this new monocular feature, in bits.
     */
    static double computeMonocularMutualInformation(
      const MonoCovarianceType &mono_covariance);

    /** Compute the mutual information between the robot state and new stereo
     * feature.
     *
     * @param stereo_covariance The full covariance matrix for the state and
     * stereo feature.
     * @return The mutual information; the amount of information added to the
     * state by measuring this new stereo feature, in bits.
     */
    static double computeStereoMutualInformation(
      const StereoCovarianceType &stereo_covariance);

    /** Update the previous state covariance for the addition of a new stereo
     * measurement using a Kalman gain style update.
     *
     * @param prev_covariance The previous state covariance.
     * @param stereo_jacobian The jacobian of the stereo reprojection function
     * with respect to the pose.
     * @param measurement_noise The noise associated with the stereo
     * measurement.
     * @return The propagated state covariance with this new measurement.
     */
    static StateCovarianceType updateStateCovarianceStereo(
      const StateCovarianceType &prev_covariance,
      const StereoProjectionPoseJacobianType &stereo_jacobian,
      const Eigen::Matrix3d &measurement_noise);

    /** Updates the previous state covariance through a constant velocity motion model propagation. The noise added to
     * the motion model propagation is an arbitrary value of 0.001.
     *
     * @param prev_covariance The previous state covariance.
     * @param motion_model The ORBSLAM "velocity matrix" (relative transform between the previous pose, and the pose
     * before it).
     * @return The updated state covariance matrix.
     */
    static StateCovarianceType updateStateCovarianceMotion(
      const StateCovarianceType &prev_covariance,
      const Eigen::Affine3d &motion_model);
};
}  // namespace SIVO

#endif  // ORBSLAM_SI_SIVO_HELPERS_CPP_HPP
