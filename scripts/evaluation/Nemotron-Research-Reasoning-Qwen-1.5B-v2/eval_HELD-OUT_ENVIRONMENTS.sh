path=$1
eval_data="HELD-OUT_ENVIRONMENTS data/HELD-OUT_ENVIRONMENTS/test.json data/HELD-OUT_ENVIRONMENTS/evaluation_config.json"

bash scripts/evaluation/Nemotron-Research-Reasoning-Qwen-1.5B-v2/eval.sh "${path}" "${eval_data}"
