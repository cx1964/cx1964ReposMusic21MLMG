# see: https://www.tensorflow.org/install/source

# Install Python and the TensorFlow package dependencies
sudo apt install python-dev python-pip  # or python3-dev python3-pip

# install the TensorFlow pip package dependencies (if using a virtual environment, omit the --user argument):
pip install -U pip six numpy wheel setuptools mock 'future>=0.17.1'
pip install -U keras_applications --no-deps
pip install -U keras_preprocessing --no-deps

# install bazel
echo "install Bazel"