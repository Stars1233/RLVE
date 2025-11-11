path=$1
eval_data=$2

if [[ "${path}" == *"iter"* ]]; then
    source scripts/models/qwen2.5-1.5B.sh
    eval_path="${path//[/X}"
    eval_path="${eval_path//]/Y}"
    eval_path="${eval_path}_hf"
    PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
      --input-dir "${path}" \
      --output-dir "${eval_path}" \
      --origin-hf-dir ../OpenThinker3-1.5B \
      --vocab-size 151936 \
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
    --eval-max-context-len 32768 \
    --num-gpus 8 \
    --eval-data ${eval_data} \
    --apply-chat-template \
    --custom-prompt-preprocessor ChatTemplate_NoSystemPrompt

if [[ "${path}" == *"iter"* ]]; then
    mkdir -p "${path}/evaluation_result"
    cp -r "${eval_path}/evaluation_result/"* "${path}/evaluation_result/"
    rm -rf "${eval_path}"
fi
