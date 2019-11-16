# Filename: setup.sh
# Function: build Linux libraries to build tensorflow 2.x pip package for specific CPU
#           Original build procedure
#           of https://intel.github.io/mkl-dnn/dev_guide_build.html
#           changed 2019115, because it did not work on my Linux platform
#
#           This script install created libraries in /usr/local/*
# 
# Howto use this script: Put this file (setup.sh) in an empty directory
#                        After this script has run, a separte directory mkl-dnn is created
#                        Start this script with ./setup.sh from seprate directory

# Reference documentation see https://github.com/intel/mkl-dnn

# Cmake requirement for Linux Ubuntu 18.04
sudo apt install build-essential
sudo apt install cmake
sudo apt install git


# get sources
git clone https://github.com/intel/mkl-dnn.git

# Generate makefile: 
export CMAKE_BUILD_TYPE='Release'
export CMAKE_INSTALL_PREFIX = '.'

# Creation of build dierctory modified because documented did not work
cd mkl-dnn
mkdir -p build
cp ./mkl-dnn/CMakeLists.txt ./build
cd build
cmake ..

# Build library
# Build based on make -j does not work on Ubuntu 18.04 @ 20191115
make

# Build doc
# turned off because of an error (probably in my environment)
# make doc

# Install the library, headers, and documentation: 
sudo make install
