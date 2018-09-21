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
 * File: bayesian_segnet.cpp
 * Desc: Source code for the C++ implementation of Bayesian SegNet.
 * Auth: Pranav Ganti
 *
 * Copyright (c) 2018, Waterloo Autonomous Vehicles Laboratory (WAVELab),
 * University of Waterloo. All Rights Reserved.
 *
 * ###########################################################################
 */

#include "bayesian_segnet/bayesian_segnet.hpp"

#include <opencv2/highgui/highgui.hpp>

#include <chrono>


namespace SIVO {

double computeEntropy(const double probability) {
    if (probability == 0) {
        return 0;
    } else {
        return -1.0 * probability * std::log2(probability);
    }
}

BayesianSegNet::BayesianSegNet(const BayesianSegNetParams &params)
    : params(params) {
    // Verify parameter values
    this->checkConfig();

    // Select hardware specification
    if (this->params.use_gpu) {
        caffe::Caffe::set_mode(caffe::Caffe::GPU);
    } else {
        caffe::Caffe::set_mode(caffe::Caffe::CPU);
    }

    // Load the network in testing phase, and load the weights
    this->network.reset(
      new caffe::Net<float>(this->params.model_file, caffe::TEST));
    this->network->CopyTrainedLayersFrom(this->params.weights_file);

    // Extract first layer, and ensure network is expecting a 3 channel image.
    this->input_layer = this->network->blob_by_name(this->input_blob_name);
    if (this->input_layer->shape(1) != 3) {
        throw std::invalid_argument("Input layer must have 3 channels!");
    } else if (this->input_layer->shape(0) <= 1) {
        throw std::invalid_argument(
          "Input layer must have a batch size greater than 1!");
    }

    // Set input image data dimensions.
    this->input_geometry =
      cv::Size{this->input_layer->shape(3), this->input_layer->shape(2)};

    // Generate class colours
    this->generateSegmentationColours();
}

void BayesianSegNet::checkConfig() {
    if (this->params.model_file.empty()) {
        throw std::invalid_argument("model_file (.prototxt file) is empty!");
    }

    if (this->params.weights_file.empty()) {
        throw std::invalid_argument(
          "weights_file (.caffemodel file) is empty!");
    }
}

void BayesianSegNet::generateSegmentationColours() {
    // Set each colour.
    this->class_colours.at<cv::Vec3b>(Classes::ROAD) = cv::Vec3b(128, 64, 128);
    this->class_colours.at<cv::Vec3b>(Classes::SIDEWALK) =
      cv::Vec3b(232, 35, 244);
    this->class_colours.at<cv::Vec3b>(Classes::BUILDING) =
      cv::Vec3b(69, 69, 69);
    this->class_colours.at<cv::Vec3b>(Classes::WALL) = cv::Vec3b(156, 102, 102);
    this->class_colours.at<cv::Vec3b>(Classes::POLE) = cv::Vec3b(153, 153, 153);
    this->class_colours.at<cv::Vec3b>(Classes::TRAFFIC_LIGHT) =
      cv::Vec3b(30, 170, 250);
    this->class_colours.at<cv::Vec3b>(Classes::TRAFFIC_SIGN) =
      cv::Vec3b(0, 220, 220);
    this->class_colours.at<cv::Vec3b>(Classes::VEGETATION) =
      cv::Vec3b(35, 142, 107);
    this->class_colours.at<cv::Vec3b>(Classes::TERRAIN) =
      cv::Vec3b(152, 251, 152);
    this->class_colours.at<cv::Vec3b>(Classes::SKY) = cv::Vec3b(180, 130, 70);
    this->class_colours.at<cv::Vec3b>(Classes::PERSON) = cv::Vec3b(60, 20, 220);
    this->class_colours.at<cv::Vec3b>(Classes::CAR) = cv::Vec3b(142, 0, 0);
    this->class_colours.at<cv::Vec3b>(Classes::COMMERCIAL_VEHICLE) =
      cv::Vec3b(70, 0, 0);
    this->class_colours.at<cv::Vec3b>(Classes::BIKE) = cv::Vec3b(32, 11, 119);
    this->class_colours.at<cv::Vec3b>(Classes::VOID) = cv::Vec3b(0, 0, 0);

    std::cout << "Class colours loaded!" << std::endl;
}

void BayesianSegNet::wrapInputLayer(
  std::vector<std::vector<cv::Mat>> &input_batch) {
    int batch = this->input_layer->shape(0);
    int channels = this->input_layer->shape(1);
    int height = this->input_layer->shape(2);
    int width = this->input_layer->shape(3);

    // Access the starting location of the input layer's data.
    float *input_data = this->input_layer->mutable_cpu_data();

    for (int i = 0; i < batch; ++i) {
        std::vector<cv::Mat> img_channels;
        for (int j = 0; j < channels; ++j) {
            // Create a new image
            cv::Mat channel(height, width, CV_32FC1, input_data);
            img_channels.push_back(channel);
            input_data += width * height;
        }

        input_batch.push_back(img_channels);
    }
}

cv::Mat BayesianSegNet::resizeImage(const cv::Mat &image) {
    // Resize image to network requirement
    cv::Mat resized_image;

    if (image.size() == this->input_geometry) {
        return image;
    } else if (image.rows >= this->input_geometry.height &&
               image.cols >= this->input_geometry.width) {
        // Only can resize image if it is larger than the requested size.
        int x_tl = image.cols / 2 - this->input_geometry.width / 2;
        int y_tl = image.rows / 2 - this->input_geometry.height / 2;

        cv::Rect roi{
          x_tl, y_tl, this->input_geometry.width, this->input_geometry.height};

        // Crop image, and clone to ensure contiguous data.
        resized_image = image(roi).clone();
    }

    return resized_image;
}

void BayesianSegNet::preprocessImage(
  const cv::Mat &image, std::vector<std::vector<cv::Mat>> &input_batch) {
    // Resize image to network specifications
    cv::Mat resized_image = this->resizeImage(image);

    // Cast from uint8 to floats.
    cv::Mat resized_float;
    resized_image.convertTo(resized_float, CV_32FC3);

    // Split image into its 3 channels, and populate data.
    for (int i = 0; i < this->input_layer->shape(0); ++i) {
        cv::split(resized_float.clone(),
                  input_batch.at(static_cast<size_t>(i)));
    }
}

MatXu BayesianSegNet::computeClasses(const Tensor3d &mean_confidence) {
    // Extract class values through argmax, over dimension 0.
    Tensor2u classes_tensor = mean_confidence.argmax(0).cast<uint8_t>();

    // Convert classes tensor to matrix
    Eigen::Map<MatXu> classes(classes_tensor.data(),
                              this->input_geometry.height,
                              this->input_geometry.width);

    return classes;
}

MatXd BayesianSegNet::computeMaxConfidence(const Tensor3d &mean_confidence) {
    // Extract maximum across the classes.
    Tensor2d confidence_tensor =
      mean_confidence.maximum(Eigen::array<int, 1>({0}));

    // Convert from Tensor to MatXd
    Eigen::Map<MatXd> confidence(confidence_tensor.data(),
                                 this->input_geometry.height,
                                 this->input_geometry.width);

    return confidence;
}

MatXd BayesianSegNet::computeVariance(const MatXu &classes) {
    // Extract output layer information (Softmax probablitiies)
    boost::shared_ptr<caffe::Blob<float>> output_layer =
      this->network->blob_by_name(this->output_blob_name);

    MatXd variance(output_layer->shape(2), output_layer->shape(3));

    // Extract output from blob.
    auto output_data =
      Eigen::TensorMap<Tensor4f>(output_layer->mutable_cpu_data(),
                                 output_layer->shape(0),
                                 output_layer->shape(1),
                                 output_layer->shape(2),
                                 output_layer->shape(3));
    Tensor4d output = output_data.cast<double>();

    // Need to look at how each Monte Carlo simulation predicted a pixel
    for (int i = 0; i < output_layer->shape(1); ++i) {
        // Extract the class-wise confidence for each pixel
        Tensor3d class_output_tensor = output.chip(i, 1);

        for (int j = 0; j < this->input_geometry.height; ++j) {
            // Extract row slice from the full image
            Tensor2d output_row = class_output_tensor.chip(j, 1);

            for (int k = 0; k < this->input_geometry.width; ++k) {
                auto detected_class = static_cast<int>(classes(j, k));

                // We only care about the variance of the properly detected
                // class.
                if (i == detected_class) {
                    // Extract the pixel vector from the row slice.
                    Tensor1d output_vector = output_row.chip(k, 1);

                    // Calculate temp variables needed for variance calculation
                    Eigen::Tensor<double, 0, Eigen::RowMajor> mean_confidence =
                      output_vector.mean();
                    double avg = mean_confidence(0);
                    auto size = static_cast<int>(output_vector.size());

                    // Calculate sum
                    double sum = 0;

                    for (int l = 0; l < size; ++l) {
                        sum += std::pow(output_vector(l) - avg, 2);
                    }

                    // Store SAMPLE variance, by dividing by size - 1.
                    variance(j, k) = sum / static_cast<double>(size - 1);
                }
            }
        }
    }

    return variance;
}

MatXd BayesianSegNet::computeClassificationEntropy(
  const Tensor3d &mean_confidence) {
    // Evaluate the elementwise entropy
    Tensor3d elementwise_entropy;
    elementwise_entropy = mean_confidence.unaryExpr(&computeEntropy);

    // Sum entropy values over all classes, and map to a matrix.
    Tensor2d entropy_tensor =
      elementwise_entropy.sum(Eigen::array<int, 1>({0}));
    Eigen::Map<MatXd> entropy(entropy_tensor.data(),
                              this->input_geometry.height,
                              this->input_geometry.width);

    return entropy;
}

Tensor3d BayesianSegNet::extractMeanConfidence() {
    // Extract output layer information (Softmax probabilities)
    boost::shared_ptr<caffe::Blob<float>> output_layer =
      this->network->blob_by_name(this->output_blob_name);

    // Copy the data to an Eigen::Tensor for postprocessing, and cast to
    // double
    auto output_data =
      Eigen::TensorMap<Tensor4f>(output_layer->mutable_cpu_data(),
                                 output_layer->shape(0),
                                 output_layer->shape(1),
                                 output_layer->shape(2),
                                 output_layer->shape(3));
    Tensor4d output = output_data.cast<double>();

    // Extract mean confidence over the Monte Carlo trials
    Tensor3d mean_confidence = output.mean(Eigen::array<int, 1>({0}));

    return mean_confidence;
}

void BayesianSegNet::segmentImage(const cv::Mat &image,
                                  MatXu &classes,
                                  MatXd &confidence,
                                  MatXd &entropy) {
    std::vector<std::vector<cv::Mat>> input_channels;

    // Preprocess input data
    this->wrapInputLayer(input_channels);
    this->preprocessImage(image, input_channels);

    // Run inference
    this->network->Forward();

    // Extract the mean confidence over the Monte Carlo simulations.
    Tensor3d mean_confidence = this->extractMeanConfidence();

    classes = this->computeClasses(mean_confidence);
    confidence = this->computeMaxConfidence(mean_confidence);
    entropy = this->computeClassificationEntropy(mean_confidence);
}

cv::Mat BayesianSegNet::generateConfidenceImage(const MatXd &confidence) {
    cv::Mat confidence_image(static_cast<int>(confidence.rows()),
                             static_cast<int>(confidence.cols()),
                             CV_64FC1);

    cv::eigen2cv(confidence, confidence_image);

    return confidence_image;
}

cv::Mat BayesianSegNet::generateVarianceImage(MatXd &variance) {
    cv::Mat variance_image(static_cast<int>(variance.rows()),
                           static_cast<int>(variance.cols()),
                           CV_64FC1);
    cv::eigen2cv(variance, variance_image);

    cv::Mat normalized_variance(static_cast<int>(variance.rows()),
                                static_cast<int>(variance.cols()),
                                CV_64FC1);

    cv::normalize(
      variance_image, normalized_variance, 0.0, 1.0, cv::NORM_MINMAX, CV_64FC1);

    return normalized_variance;
}

cv::Mat BayesianSegNet::generateEntropyImage(MatXd &entropy) {
    cv::Mat entropy_image(static_cast<int>(entropy.rows()),
                          static_cast<int>(entropy.cols()),
                          CV_64FC1);
    cv::eigen2cv(entropy, entropy_image);

    cv::Mat normalized_entropy(static_cast<int>(entropy.rows()),
                               static_cast<int>(entropy.cols()),
                               CV_64FC1);

    cv::normalize(
      entropy_image, normalized_entropy, 0.0, 1.0, cv::NORM_MINMAX, CV_64FC1);

    return normalized_entropy;
}

cv::Mat BayesianSegNet::generateSegmentedImage(const MatXu &classes,
                                               const cv::Mat &test_image) {
    cv::Mat segmented_image(static_cast<int>(classes.rows()),
                            static_cast<int>(classes.cols()),
                            CV_8UC3);
    cv::Mat classes_image(static_cast<int>(classes.rows()),
                          static_cast<int>(classes.cols()),
                          CV_8UC1);
    cv::Mat classes_image_3ch(static_cast<int>(classes.rows()),
                              static_cast<int>(classes.cols()),
                              CV_8UC3);

    // Convert classes to a cv::Mat and convert to 3ch BGR
    cv::eigen2cv(classes, classes_image);
    cv::cvtColor(classes_image, classes_image_3ch, CV_GRAY2BGR);

    // Use custom color map to convert colours.
    cv::LUT(classes_image_3ch, this->class_colours, segmented_image);

    // Resize input image (if necessary)
    cv::Mat resized_image = this->resizeImage(test_image);

    // Create an overlaid image of the segmentation and the resized image.
    cv::addWeighted(
      segmented_image, 0.5, resized_image, 0.5, 0, segmented_image);

    return segmented_image;
}
}  // namespace SIVO
