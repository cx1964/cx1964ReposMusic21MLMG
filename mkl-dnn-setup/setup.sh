# filename: setup.sh
# function: build Linux library to build tensorflow 2.x pip package for specific CPU

# put this file (setup.sh) in an empty directory
# After this script has run a seprte directory mkl-dnn is created

# see https://intel.github.io/mkl-dnn/dev_guide_build.html
# see https://github.com/intel/mkl-dnn

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
make install
