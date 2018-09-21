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
 * File: bayesian_segnet.hpp
 * Desc: Header file for C++ implementation of Bayesian SegNet.
 * Auth: Pranav Ganti
 *
 * Copyright (c) 2018, Waterloo Autonomous Vehicles Laboratory (WAVELab),
 * University of Waterloo. All Rights Reserved.
 *
 * ###########################################################################
 */

#ifndef BAYESIAN_SEGNET_BAYESIAN_SEGNET_HPP
#define BAYESIAN_SEGNET_BAYESIAN_SEGNET_HPP

#include <caffe/caffe.hpp>

#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cmath>
#include <memory>

namespace SIVO {

using MatXd =
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using MatXu =
  Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using Tensor1d = Eigen::Tensor<double, 1, Eigen::RowMajor>;

using Tensor2d = Eigen::Tensor<double, 2, Eigen::RowMajor>;

using Tensor3d = Eigen::Tensor<double, 3, Eigen::RowMajor>;

using Tensor4d = Eigen::Tensor<double, 4, Eigen::RowMajor>;

using Tensor2u = Eigen::Tensor<uint8_t, 2, Eigen::RowMajor>;

using Tensor4f = Eigen::Tensor<float, 4, Eigen::RowMajor>;

double computeEntropy(const double probability);

/// Classes that can be detected by our trained Bayesian SegNet model.
enum Classes {
    ROAD,
    SIDEWALK,
    BUILDING,
    WALL,
    POLE,
    TRAFFIC_LIGHT,
    TRAFFIC_SIGN,
    VEGETATION,
    TERRAIN,
    SKY,
    PERSON,
    CAR,
    COMMERCIAL_VEHICLE,
    BIKE,
    VOID = 255
};

struct BayesianSegNetParams {
    /** Default constructor. Requires the path to the model files, but all other
     * parameters are default.
     *
     * @param model_filepath The path to the model's .prototxt file.
     * @param weights_filepath The path to the model's .caffemodel file.
     * \endparblock
     */
    BayesianSegNetParams(const std::string model_filepath,
                         const std::string weights_filepath)
        : model_file(model_filepath), weights_file(weights_filepath) {}

    /// Boolean to determine hardware. If true, will use GPU for inference.
    bool use_gpu = true;

    /// The filepath to the model's architecture file (.prototxt).
    std::string model_file;

    /// The filepath to the trained weights to use for inference (.caffemodel)
    std::string weights_file;
};

/// Class to perform semantic segmentation
class BayesianSegNet {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /// Constructor
    explicit BayesianSegNet(const BayesianSegNetParams &params);

    /// Destructor
    ~BayesianSegNet() = default;

    /** Run inference on the image using Bayesian SegNet.
     *
     * @param[in] image The image to segment.
     * @param[out] classes The pixel-wise classes for the resized image.
     * @param[out] confidence The pixel-wise confidence for the detected
     * classes.
     * @param[out] variance The variance of the detected confidence values.
     * @param[out] entropy The classification entropy for each pixel.
     */
    void segmentImage(const cv::Mat &image,
                      MatXu &classes,
                      MatXd &confidence,
                      MatXd &entropy);

    /** Create an OpenCV matrix of the pixel-wise confidence for display.
     *
     * @param confidence The pixel-wise confidence values for the detected
     * classes.
     * @return The matrix in grayscale OpenCV format.
     */
    cv::Mat generateConfidenceImage(const MatXd &confidence);

    /** Create an OpenCV matrix of the variance values for display. This
     * method normalizes the variance matrix, such that the largest value is
     * 1. This normalization is required for the sake of visualization.
     *
     * @param variance The variance of the detected confidence values.
     * @return The normalized matrix in grayscale OpenCV format.
     */
    cv::Mat generateVarianceImage(MatXd &variance);

    /** Create an OpenCV matrix of the entropy values for display. This method
     * normalizes the entropy matrix, such that the largest value is 1. This
     * normalization is required for the sake of visualization.
     *
     * @param entropy The pixelwise classification entropy.
     * @return The normalized matrix in grayscale OpenCV format.
     */
    cv::Mat generateEntropyImage(MatXd &entropy);

    /** Create an overlaid segmented image, illustrating the segmented classes.
     *
     * @param classes The detected classes in the image.
     * @param test_image The original image to overlay the segmentations on.
     * @return The segmented image.
     */
    cv::Mat generateSegmentedImage(const MatXu &classes,
                                   const cv::Mat &test_image);

    /// Return input geometry
    cv::Size getInputGeometry() {
        return this->input_geometry;
    }

 private:
    /// Verifies that the .prototxt and .caffemodel paths are not empty.
    void checkConfig();

    /// Generates the OpenCV Lookup Table to map class colours.
    void generateSegmentationColours();

    /** Wraps the input layer of the network with a vector of images.
     *
     * This method creates placeholder images, setting the pointers to the
     * allocated data of the input layer. Caffe stores data in C-contiguous
     * format, and so we can wrap the data beforehand. This method also allows
     * us to avoid a memcpy.
     *
     * @param input_batch The batch of images for inference. The first vector
     * indicates the different images, while the second vector is the split
     * B, G, and R channels.
     */
    void wrapInputLayer(std::vector<std::vector<cv::Mat>> &input_batch);

    /** Crop the image to the input dimensions required by the network.
     *
     * @param image The image to resize.
     * @return The resized image.
     */
    cv::Mat resizeImage(const cv::Mat &image);

    /** Crop the image, and populate the data wrapped in the wrapInputLayer
     * function.
     *
     * @param image The image to be inferred for segmentation.
     * @param input_batch The batch of images for inference.
     */
    void preprocessImage(const cv::Mat &image,
                         std::vector<std::vector<cv::Mat>> &input_batch);

    /** Performs an argmax over the confidence tensor to determine the class
     * detection.
     *
     * The confidence tensor is 3 dimensions (num_classes, width, height). The
     * argmax is taken over the 0th dimension to determine which pixel had the
     * highest softmax value.
     *
     * @param mean_confidence The averaged confidence values from each batch.
     * @return The pixel-wise class detections.
     */
    MatXu computeClasses(const Tensor3d &mean_confidence);

    /** Determines the maximum pixel-wise confidence over the confidence tensor.
     *
     * @param mean_confidence The averaged confidence values from each batch.
     * @return The pixel-wise confidence values.
     */
    MatXd computeMaxConfidence(const Tensor3d &mean_confidence);

    /** Computes the sample variance associated with the confidence outputs for
     * each pixel.
     *
     * @return The pixel-wise variance values.
     */
    MatXd computeVariance(const MatXu &classes);

    /** Computes the entropy of the classification output, following the
     * formulation of Gal.
     *
     * The entropy equals the negative sum over each class of the confidence
     * multiplied by the logarithm of each value.
     *
     * @return The entropy values for each pixel.
     */
    MatXd computeClassificationEntropy(const Tensor3d &mean_confidence);

    /** Extracts the averaged confidence values across the Monte Carlo
     * simulations.
     *
     * @return The averaged confidence values from each batch, dimensions
     * (num_classes, width, height).
     */
    Tensor3d extractMeanConfidence();

 private:
    /// The neural network loaded from the .prototxt and .caffemodel files.
    std::shared_ptr<caffe::Net<float>> network;

    /// The pointer to the network's input layer data.
    boost::shared_ptr<caffe::Blob<float>> input_layer;

    /// The dimensions for the input image data.
    cv::Size input_geometry;

    /// Name of the input blob for the Bayesian SegNet network.
    std::string input_blob_name{"data"};

    /// Name of the output blob for the Bayesian SegNet network.
    std::string output_blob_name{"prob"};

    /// Lookup table for class colour mapping
    cv::Mat class_colours = cv::Mat::zeros(256, 1, CV_8UC3);

    /// Usage parameters
    BayesianSegNetParams params;
};
}  // namespace SIVO

#endif  // BAYESIAN_SEGNET_BAYESIAN_SEGNET_HPP
