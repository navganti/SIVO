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
 * Desc: Source file containing utilities for SIVO's feature selection criteria.
 * Auth: Pranav Ganti
 *
 * Copyright (c) 2018, Waterloo Autonomous Vehicles Laboratory (WAVELab),
 * University of Waterloo. All Rights Reserved.
 *
 * ###########################################################################
 */

#include <include/sivo_helpers/sivo_helpers.hpp>

namespace SIVO {

Eigen::Matrix3d createSkewSymmetricMatrixFromVector(
  const Eigen::Vector3d vector) {
    Eigen::Matrix3d skew_sym;

    skew_sym << 0, -vector(2), vector(1), vector(2), 0, -vector(0), -vector(1),
      vector(0), 0;

    return skew_sym;
}

MonoProjectionPoseJacobianType SIVO::computeMonocularJacobianPose(
  const double fx,
  const double fy,
  const double cXcp,
  const double cYcp,
  const double cZcp) {
    MonoProjectionPoseJacobianType mono_jacobian =
      MonoProjectionPoseJacobianType::Zero();

    if (cZcp != 0) {
        mono_jacobian << (fx / cZcp), 0.0, (-fx * cXcp / (cZcp * cZcp)),
          (-fx * cXcp * cYcp / (cZcp * cZcp)),
          fx * (1.0 + (cXcp * cXcp) / (cZcp * cZcp)), (-fx * cYcp / cZcp), 0.0,
          fy / cZcp, (-fy * cYcp / (cZcp * cZcp)),
          -fy * (1 + (cYcp * cYcp) / (cZcp * cZcp)),
          (fy * cXcp * cYcp / (cZcp * cZcp)), (fy * cXcp / cZcp);
    }

    return mono_jacobian;
}

StereoProjectionPoseJacobianType SIVO::computeStereoJacobianPose(
  const double fx,
  const double fy,
  const double bl,
  const double cXcp,
  const double cYcp,
  const double cZcp) {
    StereoProjectionPoseJacobianType stereo_jacobian =
      StereoProjectionPoseJacobianType::Zero();

    if (cZcp != 0) {
        stereo_jacobian << (fx / cZcp), 0.0, (-fx * cXcp / (cZcp * cZcp)),
          (-fx * cXcp * cYcp / (cZcp * cZcp)),
          fx * (1.0 + (cXcp * cXcp) / (cZcp * cZcp)), (-fx * cYcp / cZcp), 0.0,
          fy / cZcp, (-fy * cYcp / (cZcp * cZcp)),
          -fy * (1 + (cYcp * cYcp) / (cZcp * cZcp)),
          (fy * cXcp * cYcp / (cZcp * cZcp)), (fy * cXcp / cZcp), (fx / cZcp),
          0.0, (-fx * (cXcp - bl) / (cZcp * cZcp)),
          (-fx * (cXcp - bl) * cYcp / (cZcp * cZcp)),
          fx * (1.0 + (cXcp * (cXcp - bl)) / (cZcp * cZcp)),
          (-fx * cYcp / cZcp);
    }

    return stereo_jacobian;
}

MonoProjectionPointJacobianType SIVO::computeMonocularJacobianPoint(
  const double fx,
  const double fy,
  const double cXcp,
  const double cYcp,
  const double cZcp,
  const Eigen::Matrix3d Ccw) {
    MonoProjectionPointJacobianType mono_jacobian =
      MonoProjectionPointJacobianType::Zero();

    // The derivative of the projection function wrt the camera point.
    MonoProjectionPointJacobianType projection_jacobian;

    if (cZcp != 0) {
        projection_jacobian << (fx / cZcp), 0.0, (-fx * cXcp / (cZcp * cZcp)),
          0.0, (fy / cZcp), (-fy * cYcp / (cZcp * cZcp));

        mono_jacobian = projection_jacobian * Ccw;
    }

    return mono_jacobian;
}

StereoProjectionPointJacobianType SIVO::computeStereoJacobianPoint(
  const double fx,
  const double fy,
  const double bl,
  const double cXcp,
  const double cYcp,
  const double cZcp,
  const Eigen::Matrix3d Ccw) {
    StereoProjectionPointJacobianType stereo_jacobian =
      StereoProjectionPointJacobianType::Zero();

    // The derivative of the projection function wrt the camera point.
    StereoProjectionPointJacobianType projection_jacobian;

    if (cZcp != 0) {
        projection_jacobian << (fx / cZcp), 0.0, (-fx * cXcp / (cZcp * cZcp)),
          0.0, (fy / cZcp), (-fy * cYcp / (cZcp * cZcp)), (fx / cZcp), 0.0,
          (-fx * (cXcp - bl) / (cZcp * cZcp));

        stereo_jacobian = projection_jacobian * Ccw;
    }

    return stereo_jacobian;
}

MonoCovarianceType SIVO::computeMonocularCovariance(
  const StateCovarianceType &state_covariance,
  const MonoProjectionPoseJacobianType &mono_jacobian,
  const Eigen::Matrix2d &measurement_noise) {
    MonoCovarianceType mono_covariance;

    // Copy state covariance
    mono_covariance.block<6, 6>(0, 0) = state_covariance;

    // Compute propagated measurement covariance.
    mono_covariance.block<2, 2>(6, 6) =
      mono_jacobian * state_covariance * mono_jacobian.transpose() +
      measurement_noise;

    // Compute cross-correlations
    mono_covariance.block<6, 2>(0, 6) =
      state_covariance * mono_jacobian.transpose();
    mono_covariance.block<2, 6>(6, 0) = mono_jacobian * state_covariance;

    return mono_covariance;
}

StereoCovarianceType SIVO::computeStereoCovariance(
  const StateCovarianceType &state_covariance,
  const StereoProjectionPoseJacobianType &stereo_jacobian,
  const Eigen::Matrix3d &measurement_noise) {
    StereoCovarianceType stereo_covariance;

    // Copy state covariance
    stereo_covariance.block<6, 6>(0, 0) = state_covariance;

    // Compute propagated masurement covariance.
    stereo_covariance.block<3, 3>(6, 6) =
      stereo_jacobian * state_covariance * stereo_jacobian.transpose() +
      measurement_noise;

    // Compute cross-correlations
    stereo_covariance.block<6, 3>(0, 6) =
      state_covariance * stereo_jacobian.transpose();
    stereo_covariance.block<3, 6>(6, 0) = stereo_jacobian * state_covariance;

    return stereo_covariance;
}

double SIVO::computeMonocularMutualInformation(
  const MonoCovarianceType &mono_covariance) {
    double mutual_information;

    // Extract state and measurement covariance blocks
    StateCovarianceType state_covariance = mono_covariance.block<6, 6>(0, 0);
    Eigen::Matrix2d measurement_covariance = mono_covariance.block<2, 2>(6, 6);

    // Compute the determinants of all covariances.
    double state_cov_det = state_covariance.determinant();
    double meas_cov_det = measurement_covariance.determinant();
    double cov_det = mono_covariance.determinant();

    mutual_information =
      0.5 * std::log2(state_cov_det * meas_cov_det / cov_det);

    return mutual_information;
}

double SIVO::computeStereoMutualInformation(
  const StereoCovarianceType &stereo_covariance) {
    double mutual_information;

    // Extract state and measurement covariance blocks
    StateCovarianceType state_covariance = stereo_covariance.block<6, 6>(0, 0);
    Eigen::Matrix3d measurement_covariance =
      stereo_covariance.block<3, 3>(6, 6);

    // Compute the determinants of all covariances.
    double state_cov_det = state_covariance.determinant();
    double meas_cov_det = measurement_covariance.determinant();
    double cov_det = stereo_covariance.determinant();

    mutual_information =
      0.5 * std::log2(state_cov_det * meas_cov_det / cov_det);

    return mutual_information;
}

StateCovarianceType SIVO::updateStateCovarianceStereo(
  const StateCovarianceType &prev_covariance,
  const StereoProjectionPoseJacobianType &stereo_jacobian,
  const Eigen::Matrix3d &measurement_noise) {
    StateCovarianceType updated_covariance;

    Eigen::MatrixXd temp =
      stereo_jacobian * prev_covariance * stereo_jacobian.transpose() +
      measurement_noise;

    Eigen::Matrix<double, 6, 3> kalman_gain;
    kalman_gain =
      prev_covariance * stereo_jacobian.transpose() * temp.inverse();

    updated_covariance =
      (StateCovarianceType::Identity() - kalman_gain * stereo_jacobian) *
      prev_covariance;

    return updated_covariance;
}

StateCovarianceType SIVO::updateStateCovarianceMotion(
  const StateCovarianceType &prev_covariance,
  const Eigen::Affine3d &motion_model) {
    StateCovarianceType current_covariance;

    // Create skew symmetric matrix from translation.
    Eigen::Vector3d t = motion_model.translation();
    Eigen::Matrix3d t_x = createSkewSymmetricMatrixFromVector(t);

    // Create adjoint.
    Eigen::Matrix6d adjoint = Eigen::Matrix6d::Zero();
    adjoint.block<3, 3>(0, 0) = motion_model.rotation();
    adjoint.block<3, 3>(0, 3) = t_x * motion_model.rotation();
    adjoint.block<3, 3>(3, 3) = motion_model.rotation();

    // Create Jacobian.
    Eigen::Matrix6d motion_jacobian = Eigen::Matrix6d::Identity() + adjoint;

    // Propagate new state covariance, motion model noise is tunable.
    current_covariance =
      motion_jacobian * prev_covariance * motion_jacobian.transpose() +
      Eigen::Matrix6d::Identity() * 0.01;

    return current_covariance;
}

}  // namespace SIVO
