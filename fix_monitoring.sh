#!/bin/bash
# filepaths must start with "/mnt/". Directories must end with "/".
PY="/mnt/c/Python36/" # absolute path to Python installation directory
PYPLAN="/mnt/c/Users/Alex/OneDrive/Documents/Classes/OSU/Research/PyPlan/"

MONITORING="${PY}Lib/site-packages/gym/wrappers/" # Where the modified monitoring file will be moved

cd $PYPLAN # Go to PyPlan home directory
mv monitoring.py $MONITORING