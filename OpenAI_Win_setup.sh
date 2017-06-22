#!/bin/bash
GYM="/mnt/c/Users/Alex/OneDrive/Documents/Classes/OSU/Research/PyPlan/simulators/gym-master/"
DOWNLOAD="/mnt/c/Users/Alex/Downloads/"

NUMPY="${DOWNLOAD}numpy-1.13.0+mkl-cp36-cp36m-win_amd64.whl"
SCIPY="${DOWNLOAD}scipy-0.19.1-cp36-cp36m-win_amd64.whl"

sudo apt-get update
sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
sudo apt-get install -y python-pip
# sudo apt-get install -y zlib1g-dev

pushd $GYM
sudo pip install $NUMPY
sudo pip install $SCIPY
# sudo pip install tensorflow

sudo pip install 'gym'
# sudo pip install 'gym[parameter_tuning]' 
sudo pip install 'gym[classic_control]'

popd