# Filenaam: install_pyhon3_ml_env.sh
# functie: installeren python machine learning environment met native Ubuntu 18.04 libaries

# Installing pip for Python 3
# https://linuxize.com/post/how-to-install-pip-on-ubuntu-18.04/#installing-pip-for-python-3
sudo apt update
sudo apt upgrade
sudo apt install python3-pip
pip3 --version

# https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
python3 -m pip install --user --upgrade pip
python3 -m pip --version

# Installing virtualenv
python3 -m pip install --user virtualenv

#Creating a virtual environment
sudo apt-get install python3-venv
python3 -m venv env_python3_ml

#Activating a virtual environment
source env_python3_ml/bin/activate

# create a script file to activate
echo "# Filename: activate_env.sh" > ~/bin/activate_env.sh
echo "# Function: activate python3 environment for machine learning" >> ~/bin/activate_env.sh
echo "source env_python3_ml/bin/activate" >> ~/bin/activate_env.sh
chmod uog+x ~/bin/activate_env.sh

which python

# Install python3 machine learning packages
# zie https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/
# Installeer alleen geen anaconda python maar gebruik alleen de genoemde packages
pip install scipy
pip install matplotlib
pip install pandas
pip install statsmodels
pip install sklearn # Dit command geeft foutmelding "Failed building wheel for sklearn", wat genegeerd kan worden
python versions.py

#  Install deep learning packages zie paragraaf 5 van https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/
#!!! vergeet niet eerst mbv source env_python3_ml/bin/activate de environment goed te zetten voordat men resterende ML deep learning pkgs gaat installeren
pip install theano # Dit command geeft foutmelding "Failed building wheel for theano", wat genegeerd kan worden
# zie https://www.tensorflow.org/install/pip
pip install --upgrade pip # nodig voor tensorflow 2.x install
pip install tensorflow # CPU package  
pip install keras
# check
python deep_versions.py

# isntall MIT music21 packages
pip install music21

# show all installed python3 packages
pip list