#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export CARLA_ROOT=${SCRIPT_DIR}/carla
export LEADERBOARD_ROOT=${SCRIPT_DIR}/leaderboard
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${SCRIPT_DIR}/leaderboard
export PYTHONPATH=$PYTHONPATH:${SCRIPT_DIR}/scenario_runner