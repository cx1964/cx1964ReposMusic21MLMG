# zie ook stap9 van https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart

# create environment voor Music Generation Machine Learning
conda create --name mlmg python=3
conda activate mlmg # tbv aanzetten mlmg environment

# install machine learning packages
# zie
# stap1:
# Evernote: How to Setup Your Python Environment for Machine Learning with Anaconda
# https://www.evernote.com/shard/s214/sh/384fbbb4-6e67-4b96-b892-87152a545f50/7e06f032da29c3229e9e3fef75f915d9
# bron: https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/
#
conda -V
python -V

conda update conda # dit werkt niet in env mlmg. conda list wel
conda update anaconda # dit werkt niet in env mlmg. conda list wel
python versions.py # package scipy bestaat niet in conda env mgml
conda install -c anaconda scipy # zie https://anaconda.org/anaconda/scipy
python versions.py # matplotlib niet aanwezig
conda install -c conda-forge matplotlib 
python versions.py # pandas niet aanwezig
conda install -c anaconda pandas 
python versions.py # statsmodels niet aanwezig
conda install -c anaconda statsmodels
python versions.py # sklearn niet aanwezig
conda install -c anaconda scikit-learn
python versions.py # script nu zonder foutmeldingen

# Stap2: install tensorflow en keras
# evernote https://www.evernote.com/Home.action#n=8b1b4345-116e-4629-b2dd-3bf7b52e9575&s=s214&ses=1&sh=5&sds=5&x=machine%2520learning&
# bronnen: https://www.pyimagesearch.com/2019/01/30/ubuntu-18-04-install-tensorflow-and-keras-for-deep-learning/
# en voor tensorflow voor  cpu: conda install tensorflow
# 
# gebruik altijd in eerste instantie conda install <package name>
# als dit niet luky gebruik dan pas pip install <package name>
sudo apt-get update
sudo apt-get upgrade

# CPU users: Skip to “Step #5”
# Wordt gekozen voor CPU versie
sudo apt-get install build-essential cmake unzip pkg-config
sudo apt-get install libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
sudo apt-get install libjpeg-dev libpng-dev libtiff-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
sudo apt-get install libgtk-3-dev
sudo apt-get install libopenblas-dev libatlas-base-dev liblapack-dev gfortran
sudo apt-get install libhdf5-serial-dev
sudo apt-get install python3-dev python3-tk python-imaging-tk
#Step #5: Create your Python virtual environment
# Environment is algemaakt en heet mgml
#Step #6: Install Python libraries
conda install -c anaconda numpy
# for windows: conda install -c michael_wild opencv-contrib
# install opencv ubuntu
conda install -c conda-forge opencv
conda install -c conda-forge/label/cf201901 opencv

# install tensorflow
conda install -c conda-forge theano 
# install tensorflow
conda install -c conda-forge tensorflow 

#install keras
conda install -c conda-forge keras

# test
python deep_versions.py

#install music21
pip install music21
