path=$1
eval_data="HELD-OUT_ENVIRONMENTS data/HELD-OUT_ENVIRONMENTS/test.json data/HELD-OUT_ENVIRONMENTS/evaluation_config.json"

bash scripts/evaluation/Qwen2.5-7B/eval.sh "${path}" "${eval_data}"
