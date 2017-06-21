PY = # absolute path to Python installation directory
PYPLAN = 
MONITORING = $PY + '/Lib/site-packages/gym/wrappers/' # Where the modified monitoring file will be moved

cd $PYPLAN # Go to PyPlan home directory
mv monitoring.py $MONITORING