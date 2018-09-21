# Dependencies and Licenses for SIVO version 1.0

In this document we list all the pieces of code included by SIVO and linked libraries which are not property of the authors of SIVO.

## Code in the **src** and **include** folders

- All code in the `src/orbslam` and `include/orbslam` folders is original or modified source code from [`ORB_SLAM2`](https://github.com/raulmur/ORB_SLAM2/), which is released under the GPLv3 license. As SIVO is a modification of ORB_SLAM2, this code is also released under the [GPLv3 license](https://github.com/navganti/SIVO/blob/master/LICENSE.txt).

- `src/orbslam/ORBextractor.cc` is a modified version of `orb.cpp` from the OpenCV library. The original code is BSD licensed.

- `PnPsolver.h`, `PnPsolver.cc`: are modified versions of the epnp.h and epnp.cc of Vincent Lepetit. This code can be found in popular BSD licensed computer vision libraries as [OpenCV](https://github.com/Itseez/opencv/blob/master/modules/calib3d/src/epnp.cpp) and [OpenGV](https://github.com/laurentkneip/opengv/blob/master/src/absolute_pose/modules/Epnp.cpp). The original code is FreeBSD.

- The function `ORBmatcher::DescriptorDistance` in `src/orbslam/ORBmatcher.cc` can be found [here](http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel). The code is in the public domain.

## Code in the __config__ folder

- The `.prototxt` and `.caffemodel` files within the `config/bayesian_segnet` folder are the Caffe model and weights files for Bayesian SegNet ([forked repository](https://github.com/navganti/SegNet), [original](https://github.com/alexgkendall/SegNet-Tutorial)). These files are released under the Creative Commons License.

## Code in the __dependencies__ folder

- All code in `dependencies/DBoW2` is a modified version of the [DBoW2](https://github.com/dorian3d/DBoW2) and [DLib](https://github.com/dorian3d/DLib) libraries. All files included are released under the BSD license.

- All code in `dependencies/g2o` is a modified version of g2o ([forked repository](https://github.com/navganti/g2o), [original](https://github.com/RainerKuemmerle/g2o)). All files included are released under the BSD license.

- All code in `dependencies/caffe-segnet-cudnn7` is a modified version of [Caffe](https://github.com/BVLC/caffe). All files are released under the FreeBSD license.

## External dependencies

- __Pangolin (visualization and user interface)__ is released under the MIT license.

- __OpenCV (computer vision)__ is released under the BSD license.

- __Eigen3 (linear algebra)__ is released under MPL2 for versions greater than 3.1.1. Earlier versions are released under LGPLv3.
