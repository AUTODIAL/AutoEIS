#!/usr/bin/env bash

# The following lines are not needed if your .bashrc has them
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
echo

# Name of the virtual environment
VENV_NAME="autoeis"
PYTHON_VERSION="3.10.13"
PYTHON_DEPS="requirements.txt"
# Use single quotes to tell Bash to not interpret any characters (e.g., !)
DONE_MSG='\033[0;32mSUCCESS!\033[0m'
FAILED_MSG='\033[31mFAIL!\033[0m'

# Remove the existing virtual environment if it exists
if pyenv virtualenvs | grep -q "$VENV_NAME"; then
    echo -n "> Deleting existing virtual environment: $VENV_NAME ... "
    pyenv uninstall -f "$VENV_NAME"
    echo -e "$DONE_MSG"
fi

# Create a new virtual environment
echo -n "> Creating new virtual environment: $VENV_NAME ... "
pyenv virtualenv -q "$PYTHON_VERSION" "$VENV_NAME"
echo -e "$DONE_MSG"

# Activate the virtual environment
echo -n "> Activating virtual environment: $VENV_NAME ... "
# source $(pyenv root)/versions/$VENV_NAME/bin/activate
pyenv activate -q "$VENV_NAME"
echo -e "$DONE_MSG"

# Install Python packages
echo -n "> Installing Python packages ... "
pip install --upgrade pip -q
pip install -r "$PYTHON_DEPS" -q
# Optional requirements
pip install ipython ipykernel -q
echo -e "$DONE_MSG"
# Install AutoEIS
echo -n "> Installing AutoEIS ... "
pip install -e . -q
echo -e "$DONE_MSG"

# Install Julia and packages
echo -n "> Installing Julia packages ... "
python -c "import autoeis.julia_helpers as jh; jh.install(precompile=True, quiet=True)" 2>error.txt
if [ $? -eq 0 ]; then
    echo -e "$DONE_MSG"
else
    echo -e "$FAILED_MSG" "See error.txt for details"
fi

# Deactivate virtual environment
echo -n "> Deactivating virtual environment: $VENV_NAME ... "
pyenv deactivate -q "$VENV_NAME"
echo -e "$DONE_MSG"
