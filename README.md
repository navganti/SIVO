# SIVO: Semantically Informed Visual Odometry and Mapping

SIVO is a novel feature selection method for visual SLAM which facilitates long-term localization. This algorithm enhances traditional feature detectors with deep learning based scene understanding using a Bayesian neural network, which provides context for visual SLAM while accounting for neural network uncertainty.

Our method selects features which provide the highest reduction in Shannon entropy between the entropy of the current state and the joint entropy of the state, given the addition of the new feature with the classification entropy of the feature from a Bayesian neural network. This strategy generates a sparse map suitable for long-term localization, as each selected feature significantly reduces the uncertainty of the vehicle state and has been detected to be a static object (building, traffic sign, etc.) repeatedly with a high confidence.

The paper can be found [here](https://arxiv.org/pdf/1811.11946.pdf). If you find this code useful, please cite the paper:

```
@inproceedings{ganti2019network,
  title={Network Uncertainty Informed Semantic Feature Selection for Visual SLAM},
  author={Ganti, Pranav and Waslander, Steven},
  booktitle={16th Conference on Computer and Robot Vision (CRV)},
  pages={121--128},
  year={2019},
  organization={IEEE}
}
```

If you'd like to deep dive further into the theory, background, or methodology, please refer to my [thesis](https://uwspace.uwaterloo.ca/handle/10012/14111). If you use refer to this document in your work, please cite it:

```
@mastersthesis{ganti2018SIVO,
author={{Ganti, Pranav}},
title={SIVO: Semantically Informed Visual Odometry and Mapping},
year={2018},
publisher="UWSpace",
url={http://hdl.handle.net/10012/14111}
}

```


This method builds on the work of Bayesian SegNet and ORB\_SLAM2. Detailed background information can be found [below](#background-and-related-publications).

## Prerequisites

### Hardware and Operating System

This implementation has been tested with **Ubuntu 16.04**.

A powerful CPU (e.g. Intel i7), and a powerful GPU (e.g. NVIDIA TitanX) are required to provide more stable and accurate results. Due to the technique of approximating a Bayesian Neural Network by passing an image through the network several times, this network does not quite run in real time.

### C++11 Compiler
The thread and chrono functionalities of C++11 are required

### Caffe-SegNet (Included in the dependencies folder)

A modified version of Caffe is required to use Bayesian SegNet. Please see the [`caffe-segnet-cudnn7`](https://github.com/navganti/caffe-segnet-cudnn7/tree/b37d681223c15cb7a65181ad675fca54f7b02e9d) submodule within this repository, and follow the installation instructions.

If you wish to test or train weights for the Bayesian SegNet architecture, please see our modified [SegNet](https://www.github.com/navganti/SegNet) repository for information and a tutorial.

### Pangolin
Pangolin is used for visualization and user interface. Download and install instructions can be found [here](https://github.com/stevenlovegrove/Pangolin).

### OpenCV
OpenCV is used to manipulate images and features. Download and install instructions can be found [here](http://opencv.org). **Required version > OpenCV 3.2**.

### Eigen3
Eigen3 is used for linear algebra, specifically matrix and tensor manipulation. Required by g2o and Bayesian SegNet. Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required version > 3.2.0 (for Tensors)**.

### DBoW2 (Included in the dependencies folder)
We use modified versions of the [DBoW2](https://github.com/dorian3d/DBoW2) library to perform place recognition.

### g2o (Included in the dependencies folder)

We use a modified version of the [g2o](https://github.com/navganti/g2o) library to perform non-linear optimization for the SLAM backend. The original repository can be found [here](https://github.com/rainerkuemmerle/g2o).

## Install

Clone the repository:

`git clone --recursive https://github.com/navganti/SIVO.git`

or

`git clone --recursive git@github.com:navganti/SIVO.git`

Ensure you use the recursive flag to initialize the submodule.

### Downloading Trained Weights

The network weights (`*.caffemodel`) are stored using Git LFS. If you already have this installed, the `git clone` command above should download all necessary files. If not, perform the following.

1. Install Git LFS. The steps can be found [here](https://help.github.com/articles/installing-git-large-file-storage/).
2. Run the command: `git lfs install`
3. Navigate to the location of the SIVO home folder.
4. Run the command: `git lfs pull`. This should download the weights and place them in the appropriate subfolders within `config/bayesian_segnet/`.

There are 2 separate weights files. There are files for both _Standard_ and _Basic_ Bayesian SegNet trained on the [KITTI Semantic Dataset](http://www.cvlibs.net/datasets/kitti/eval_semantics.php); these weights were first trained using the [Cityscapes Dataset](https://www.cityscapes-dataset.com), and were then fine tuned. All weights have the batch normalization layer merged with the preceding convolutional layer in order to speed up inference.

## Build

1. Build [`caffe-segnet-cudnn7`](https://github.com/navganti/caffe-segnet-cudnn7/tree/cd322a1205c9536b3d74416c17d04e8dc857c053). Navigate to `dependencies/caffe-segnet-cudnn7`, and follow the instructions listed in the README, specifically the __CMake__ installation steps. This process is a little involved - please follow the instructions carefully.
2. Ensure all other [prerequisites](#prerequisites) are installed.
3. We provide a script, `build.sh`, to build DBoW2, g2o, as well as the SIVO repository. Within the main repository folder, run:

```
chmod +x build.sh
./build.sh
```

This will create **liborbslam_so**, **libbayesian_segnet.so**, and **libsivo_helpers.so** in the _lib_ folder and the executable **SIVO** in the _bin_ folder.

## Usage

The program can be run with the following:

```bash
./bin/SIVO config/Vocabulary/ORBvoc.txt config/CONFIGURATION_FILE config/bayesian_segnet/PATH_TO_PROTOTXT config/bayesian_segnet/PATH_TO_CAFFEMODEL PATH_TO_DATASET_FOLDER/dataset/sequences/SEQUENCE_NUMBER
```
The parameters `CONFIGURATION_FILE`, `PATH_TO_PROTOTXT`, `PATH_TO_CAFFEMODEL`, and `PATH_TO_DATASET_FOLDER` must be modified.

### KITTI Dataset

To use SIVO with the KITTI dataset, perform the following

1. Download the dataset (colour images) from [here](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).

2. Modify CONFIGURATION_FILE to be `/kitti/KITTIX.yaml`, where `KITTIX.yaml` is one of `KITTI00-02.yaml`, `KITTI03.yaml`, or `KITTI04-12.yaml` for sequence 0 to 2, 3, and 4 to 12 respectively. Parameters can be modified within these files (such as entropy threshold, number of base ORB features, etc.)

3. Change `PATH_TO_PROTOTXT` to the full path to the desired Bayesian SegNet .prototxt file (e.g. basic/kitti/bayesian_segnet_basic_kitti.prototxt). __Modify the `input_dim` value to be a batch size which fits on your GPU__. Keep the image size in mind - for KITTI, we are resizing the images to be 352 x 1024 such that they work with the network dimensions. This resizing happens within the `src/orbslam/System.cc` file, and the resizing dimensions are taken from the `.prototxt` file. 

4. Change `PATH_TO_CAFFEMODEL` to the full path to the desired Bayesian SegNet .caffemodel file (e.g. basic/kitti/bayesian_segnet_basic_kitti.caffemodel).

5. Change `PATH_TO_DATASET_FOLDER` to the full path to the downloaded dataset folder. Change `SEQUENCE_NUMBER` to 00, 01, 02,.., 10.

6. Launch the program using the above command.

### SLAM and Localization Modes
You can change between the *SLAM* and *Localization* modes using the GUI of the map viewer.

__SLAM Mode__: This is the default mode. The system runs three threads in parallel : Tracking, Local Mapping and Loop Closing. The system localizes the camera, builds new map and tries to close loops.

__Localization Mode__: This mode can be used when you have a good map of your working area. In this mode the Local Mapping and Loop Closing are deactivated. The system localizes the camera in the map (which is no longer updated), using relocalization if needed.

# License

SIVO is a modification of ORB_SLAM2, which is released under the [GPLv3 license](https://github.com/navganti/SIVO/blob/master/LICENSE.txt). Therefore, this code is also released under GPLv3. The original ORB_SLAM2 license can be found [here](https://github.com/raulmur/ORB_SLAM2/blob/master/License-gpl.txt).

For a detailed list of all code/library dependencies and associated licenses, please see [Dependencies.md](https://github.com/navganti/SIVO/blob/master/Dependencies.md).

# Background and Related Publications

## Bayesian SegNet

This work uses the [Bayesian SegNet](http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html) architecture for semantic segmentation, created by [Alex Kendall](https://alexgkendall.com/), [Vijay Badrinarayanan](https://sites.google.com/site/vijaybacademichomepage/home), and [Roberto Cipolla](http://mi.eng.cam.ac.uk/~cipolla/).

Bayesian SegNet is an extension of [SegNet](http://mi.eng.cam.ac.uk/projects/segnet/), a deep convolutional encoder-decoder architecture for image segmentation. This network extends SegNet by incorporating model uncertainty, implemented through dropout layers.

### Publications

For more information about the SegNet architecture:

Alex Kendall, Vijay Badrinarayanan and Roberto Cipolla __"Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding."__ arXiv preprint arXiv:1511.02680, 2015. [PDF](http://arxiv.org/abs/1511.02680).

Vijay Badrinarayanan, Alex Kendall and Roberto Cipolla __"SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation."__ PAMI, 2017. [PDF](http://arxiv.org/abs/1511.00561).

## ORB_SLAM2

SIVO's localization functionality builds upon ORB_SLAM2. ORB_SLAM2 is a real-time SLAM library for **Monocular**, **Stereo** and **RGB-D** cameras that computes the camera trajectory and a sparse 3D reconstruction (in the stereo and RGB-D case with true scale). It is able to detect loops and relocalize the camera in real time. We provide examples to run the SLAM system in the [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) as stereo or monocular, in the [TUM dataset](http://vision.in.tum.de/data/datasets/rgbd-dataset) as RGB-D or monocular, and in the [EuRoC dataset](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) as stereo or monocular. ORB_SLAM2 also contains a ROS node to process live monocular, stereo or RGB-D streams. **The library can be compiled without ROS**. ORB-SLAM2 provides a GUI to change between a *SLAM Mode* and *Localization Mode*, see [here](#slam-and-localization-modes).

### Example Videos
<a href="https://www.youtube.com/embed/ufvPS5wJAx0" target="_blank"><img src="http://img.youtube.com/vi/ufvPS5wJAx0/0.jpg"
alt="ORB-SLAM2" width="240" height="180" border="10" /></a>
<a href="https://www.youtube.com/embed/T-9PYCKhDLM" target="_blank"><img src="http://img.youtube.com/vi/T-9PYCKhDLM/0.jpg"
alt="ORB-SLAM2" width="240" height="180" border="10" /></a>
<a href="https://www.youtube.com/embed/kPwy8yA4CKM" target="_blank"><img src="http://img.youtube.com/vi/kPwy8yA4CKM/0.jpg"
alt="ORB-SLAM2" width="240" height="180" border="10" /></a>

### Publications:

[Monocular] Raúl Mur-Artal, J. M. M. Montiel and Juan D. Tardós. **ORB-SLAM: A Versatile and Accurate Monocular SLAM System**. *IEEE Transactions on Robotics,* vol. 31, no. 5, pp. 1147-1163, 2015. (**2015 IEEE Transactions on Robotics Best Paper Award**). **[PDF](http://webdiis.unizar.es/~raulmur/MurMontielTardosTRO15.pdf)**.

[Stereo and RGB-D] Raúl Mur-Artal and Juan D. Tardós. **ORB-SLAM2: an Open-Source SLAM System for Monocular, Stereo and RGB-D Cameras**. *IEEE Transactions on Robotics,* vol. 33, no. 5, pp. 1255-1262, 2017. **[PDF](https://128.84.21.199/pdf/1610.06475.pdf)**.

[DBoW2 Place Recognizer] Dorian Gálvez-López and Juan D. Tardós. **Bags of Binary Words for Fast Place Recognition in Image Sequences**. *IEEE Transactions on Robotics,* vol. 28, no. 5, pp.  1188-1197, 2012. **[PDF](http://doriangalvez.com/php/dl.php?dlp=GalvezTRO12.pdf)**

If you use ORB_SLAM2 (Monocular) in an academic work, please cite:

    @article{murTRO2015,
      title={{ORB-SLAM}: a Versatile and Accurate Monocular {SLAM} System},
      author={Mur-Artal, Ra\'ul, Montiel, J. M. M. and Tard\'os, Juan D.},
      journal={IEEE Transactions on Robotics},
      volume={31},
      number={5},
      pages={1147--1163},
      doi = {10.1109/TRO.2015.2463671},
      year={2015}
     }

if you use ORB_SLAM2 (Stereo or RGB-D) in an academic work, please cite:

    @article{murORB2,
      title={{ORB-SLAM2}: an Open-Source {SLAM} System for Monocular, Stereo and {RGB-D} Cameras},
      author={Mur-Artal, Ra\'ul and Tard\'os, Juan D.},
      journal={IEEE Transactions on Robotics},
      volume={33},
      number={5},
      pages={1255--1262},
      doi = {10.1109/TRO.2017.2705103},
      year={2017}
     }
