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
 * File: test_map_builder.cpp
 * Desc: Library-level testing for the map building module.
 * Auth: Pranav Ganti
 *
 * Copyright (c) 2018, Waterloo Autonomous Vehicles Laboratory (WAVELab),
 * University of Waterloo. All Rights Reserved.
 *
 * ###########################################################################
 */

#include "bayesian_segnet/bayesian_segnet.hpp"

#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

#include <opencv2/highgui/highgui.hpp>

#include <gtest/gtest.h>

namespace SIVO {
const auto TEST_MODEL = "config/test_model.prototxt";
const auto TEST_WEIGHTS = "config/test_weights.caffemodel";
const auto TEST_IMAGE = "data/test_image.png";

TEST(EigenTests, ArgmaxTest) {
    Eigen::Tensor<float, 4, Eigen::RowMajor> tensor(2, 3, 5, 7);
    tensor.setRandom();
    tensor = (tensor + tensor.constant(0.5)).log();
    tensor(0, 0, 0, 0) = 10.0;

    Eigen::Tensor<Eigen::DenseIndex, 0, Eigen::RowMajor> tensor_argmax =
      tensor.argmax();
    ASSERT_EQ(tensor_argmax(0), 0);

    tensor(1, 2, 4, 6) = 20.0;

    tensor_argmax = tensor.argmax();
    ASSERT_EQ(tensor_argmax(0), 2 * 3 * 5 * 7 - 1);

    // Eigen's test_argmax_dim() test.
    Eigen::Tensor<float, 4, Eigen::RowMajor> tensor2(2, 3, 5, 7);
    std::vector<int> dims{2, 3, 5, 7};

    for (int dim = 0; dim < 4; ++dim) {
        tensor2.setRandom();
        tensor2 = (tensor2 + tensor2.constant(0.5)).log();

        Eigen::Tensor<Eigen::DenseIndex, 3, Eigen::RowMajor> tensor2_argmax;
        Eigen::array<Eigen::DenseIndex, 4> ix;

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 5; ++k) {
                    for (int l = 0; l < 7; ++l) {
                        ix[0] = i;
                        ix[1] = j;
                        ix[2] = k;
                        ix[3] = l;
                        if (ix[dim] != 0)
                            continue;
                        // suppose dim == 1, then for all i, k, l, set tensor(i,
                        // 0, k, l) = 10.0
                        tensor2(ix) = 10.0;
                    }
                }
            }
        }

        tensor2_argmax = tensor2.argmax(dim);

        ASSERT_EQ(tensor2_argmax.size(),
                  std::ptrdiff_t(2 * 3 * 5 * 7 / tensor2.dimension(dim)));

        for (std::ptrdiff_t n = 0; n < tensor2_argmax.size(); ++n) {
            // Expect max to be in the last index of the reduced dimension
            ASSERT_EQ(tensor2_argmax.data()[n], 0);
        }

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 5; ++k) {
                    for (int l = 0; l < 7; ++l) {
                        ix[0] = i;
                        ix[1] = j;
                        ix[2] = k;
                        ix[3] = l;
                        if (ix[dim] != tensor2.dimension(dim) - 1)
                            continue;
                        // suppose dim == 1, then for all i, k, l, set tensor(i,
                        // 2, k, l) = 20.0
                        tensor2(ix) = 20.0;
                    }
                }
            }
        }

        tensor2_argmax = tensor2.argmax(dim);

        ASSERT_EQ(tensor2_argmax.size(),
                  ptrdiff_t(2 * 3 * 5 * 7 / tensor2.dimension(dim)));
        for (ptrdiff_t n = 0; n < tensor2_argmax.size(); ++n) {
            // Expect max to be in the last index of the reduced dimension
            ASSERT_EQ(tensor2_argmax.data()[n], tensor2.dimension(dim) - 1);
        }
    }

    // Dimensionality checks for argmax.
    Eigen::Tensor<double, 3, Eigen::RowMajor> tensor3(15, 352, 1024);
    Eigen::Tensor<Eigen::DenseIndex, 2, Eigen::RowMajor> tensor3_argmax;
    int dim = 0;

    tensor3.setRandom();
    tensor3 = (tensor3 + tensor3.constant(0.5)).log();

    tensor3_argmax = tensor3.argmax(dim);

    ASSERT_EQ(tensor3_argmax.size(), 352 * 1024);
}

TEST(BayesianSegNetTests, InitializationTest) {
    std::string test_model_bad, test_weights_bad;

    BayesianSegNetParams params0(TEST_MODEL, test_weights_bad);
    BayesianSegNetParams params1(test_model_bad, TEST_WEIGHTS);
    BayesianSegNetParams params2(test_model_bad, test_weights_bad);
    BayesianSegNetParams params3(TEST_MODEL, TEST_WEIGHTS);

    EXPECT_THROW(BayesianSegNet model0(params0), std::invalid_argument);
    EXPECT_THROW(BayesianSegNet model1(params1), std::invalid_argument);
    EXPECT_THROW(BayesianSegNet model2(params2), std::invalid_argument);
    EXPECT_NO_THROW(BayesianSegNet model3(params3));
}

TEST(BayesianSegNetTests, SegmentationTest) {
    cv::Mat test_image = cv::imread(TEST_IMAGE);
    BayesianSegNet model{
      BayesianSegNetParams{TEST_MODEL, TEST_WEIGHTS}};

    cv::Size input_geometry = model.getInputGeometry();

    MatXu classes;
    MatXd confidence;
    MatXd entropy;

    model.segmentImage(test_image, classes, confidence, entropy);

    ASSERT_EQ(classes.size(), input_geometry.height * input_geometry.width);
    ASSERT_EQ(confidence.size(), input_geometry.height * input_geometry.width);
    ASSERT_EQ(entropy.size(), input_geometry.height * input_geometry.width);
}

TEST(BayesianSegNetTests, ConfidenceImageTest) {
    cv::Mat test_image = cv::imread(TEST_IMAGE);
    BayesianSegNet model{
      BayesianSegNetParams{TEST_MODEL, TEST_WEIGHTS}};

    MatXu classes;
    MatXd confidence;
    MatXd entropy;

    model.segmentImage(test_image, classes, confidence, entropy);

    auto confidence_image = model.generateConfidenceImage(confidence);

    cv::imshow("confidence_image", confidence_image);
    cv::waitKey(0);
}

TEST(BayesianSegNetTests, SegmentedImageTest) {
    cv::Mat test_image = cv::imread(TEST_IMAGE);
    BayesianSegNet model{
      BayesianSegNetParams{TEST_MODEL, TEST_WEIGHTS}};

    MatXu classes;
    MatXd confidence;
    MatXd entropy;

    model.segmentImage(test_image, classes, confidence, entropy);

    auto segmented_image = model.generateSegmentedImage(classes, test_image);

    cv::imshow("segmentation", segmented_image);
    cv::waitKey(0);
}

TEST(BayesianSegNetTests, EntropyImageTest) {
    cv::Mat test_image = cv::imread(TEST_IMAGE);
    BayesianSegNet model{
            BayesianSegNetParams{TEST_MODEL, TEST_WEIGHTS}};

    MatXu classes;
    MatXd confidence;
    MatXd entropy;

    model.segmentImage(test_image, classes, confidence, entropy);

    auto entropy_image = model.generateEntropyImage(entropy);

    cv::imshow("entropy", entropy_image);
    cv::waitKey(0);
}
}  // namespace SIVO

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}
