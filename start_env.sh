#!/bin/bash

# prep for ADCIRC-AI ops
# usage: cd $home && source ai/ADCIRC-AI/start_env.sh

# call this script from the $home directory
source ./.virtualenvs/LinkedIn-Learning/bin/activate

# go to the adcirc directory
cd ai/ADCIRC-AI/

# Load environment variables from .env file
if [ -f .env ]; then
  set -a  # Automatically export all variables
  source .env
  set +a  # Disable automatic export
else
  echo ".env file not found!"
fi
