path=$1
eval_data=$2

if [[ "${path}" == *"iter"* ]]; then
    source scripts/models/qwen2.5-7B.sh
    eval_path="${path//[/X}"
    eval_path="${eval_path//]/Y}"
    eval_path="${eval_path}_hf"
    PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
      --input-dir "${path}" \
      --output-dir "${eval_path}" \
      --origin-hf-dir ../Qwen2.5-7B \
      --vocab-size 152064 \
      --force
else
    eval_path="${path}"
fi



# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

python -m evaluation.evaluate \
    --model-path "${eval_path}" \
    --eval-max-response-len 16384 \
    --num-gpus 8 \
    --eval-data ${eval_data} \
    --custom-prompt-preprocessor TinyZero \
    --answer-marker-type "<answer></answer>"

if [[ "${path}" == *"iter"* ]]; then
    mkdir -p "${path}/evaluation_result"
    cp -r "${eval_path}/evaluation_result/"* "${path}/evaluation_result/"
    rm -rf "${eval_path}"
fi
