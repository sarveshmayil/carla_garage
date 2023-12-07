export WORK_DIR=/home/hersh/repos/carla_garage

export SCENARIOS=${WORK_DIR}/leaderboard/data/scenarios/eval_scenarios.json
export ROUTES=${WORK_DIR}/leaderboard/data/longest6.xml
export REPETITIONS=1
export CHALLENGE_TRACK_CODENAME=SENSORS
export CHECKPOINT_ENDPOINT=${WORK_DIR}/results/transfuser_plus_plus_longest6.json
export TEAM_AGENT=${WORK_DIR}/team_code/sensor_agent.py
export TEAM_CONFIG=${WORK_DIR}/pretrained_models/longest6/tfpp_all_2
export DEBUG_CHALLENGE=0
export RESUME=0
export DATAGEN=0
export SAVE_PATH=${WORK_DIR}/results
export UNCERTAINTY_THRESHOLD=0.33
export BENCHMARK=longest6

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=0 \
--resume=${RESUME} \
--timeout=600
