#!/usr/bin/env bash

BUILD_TYPE=$1

if [ -z "$BUILD_TYPE" ]; then
     BUILD_TYPE="Release"
fi

echo "Build type: " $BUILD_TYPE

echo "Configuring and building DBoW2 ..."

cd dependencies/DBoW2
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=$BUILD_TYPE
make -j

cd ../../g2o

echo "Configuring and building g2o ..."
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=$BUILD_TYPE
make -j
sudo make install

cd ../../../

echo "Uncompress vocabulary ..."

cd config/Vocabulary
tar -xf ORBvoc.txt.tar.gz
cd ../../

echo "Configuring and building SIVO ..."
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=$BUILD_TYPE
make -j
