cmake_minimum_required(VERSION 2.8)
project(SIVO)

if(NOT CMAKE_BUILD_TYPE)
   MESSAGE(STATUS "Using 'Release' build type as CMAKE_BUILD_TYPE is not set")
   set(CMAKE_BUILD_TYPE Release)
endif()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

if (CMAKE_BUILD_TYPE MATCHES Release)
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -Wno-deprecated -march=native")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -Wno-deprecated -march=native")
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-deprecated -march=native")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-deprecated -march=native")
endif()

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Caffe REQUIRED)
find_package(G2O REQUIRED QUIET)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${Caffe_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIR}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(g2o_LIBRARIES
    ${G2O_STUFF_LIBRARY}
    ${G2O_CORE_LIBRARY}
    ${G2O_SOLVER_CHOLMOD}
    ${G2O_SOLVER_CSPARSE}
    ${G2O_SOLVER_DENSE}
    ${G2O_SOLVER_EIGEN}
    ${G2O_TYPES_DATA}
    ${G2O_TYPES_SBA}
    ${G2O_TYPES_SIM3}
    ${G2O_TYPES_SLAM3D}
)

add_library(bayesian_segnet SHARED
    src/bayesian_segnet/bayesian_segnet.cpp
)
target_link_libraries(bayesian_segnet
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${Caffe_LIBRARIES}
)

add_library(sivo_helpers SHARED
    src/sivo_helpers/sivo_helpers.cpp
)
target_link_libraries(sivo_helpers
    ${EIGEN3_LIBS}
    ${OpenCV_LIBS}
)

add_library(orbslam SHARED
    src/orbslam/System.cc
    src/orbslam/Tracking.cc
    src/orbslam/LocalMapping.cc
    src/orbslam/LoopClosing.cc
    src/orbslam/ORBextractor.cc
    src/orbslam/ORBmatcher.cc
    src/orbslam/FrameDrawer.cc
    src/orbslam/Converter.cc
    src/orbslam/MapPoint.cc
    src/orbslam/KeyFrame.cc
    src/orbslam/Map.cc
    src/orbslam/MapDrawer.cc
    src/orbslam/Optimizer.cc
    src/orbslam/PnPsolver.cc
    src/orbslam/Frame.cc
    src/orbslam/KeyFrameDatabase.cc
    src/orbslam/Sim3Solver.cc
    src/orbslam/Viewer.cc
)

target_link_libraries(orbslam
    bayesian_segnet
    sivo_helpers
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${Pangolin_LIBRARIES}
    ${PROJECT_SOURCE_DIR}/dependencies/DBoW2/lib/libDBoW2.so
    ${g2o_LIBRARIES}
    cholmod
    pthread
)

# Build examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(${PROJECT_NAME}
    src/sivo.cc
)
target_link_libraries(${PROJECT_NAME}
    orbslam
)

# Testing
# ------------------------------------------
if (CMAKE_BUILD_TYPE MATCHES Debug)
   MESSAGE("Building Tests...")
   find_package(GTest REQUIRED)
   include_directories(${GTEST_INCLUDE_DIRS})

   set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

   add_executable(test_bayesian_segnet
       tests/test_bayesian_segnet.cpp
   )
   target_link_libraries(test_bayesian_segnet
       ${GTEST_LIBRARIES}
       pthread
       bayesian_segnet
   )
endif()
