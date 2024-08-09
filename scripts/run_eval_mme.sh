model_path=$1
data_dir=$2 # /path/to/mixlora_data/mme/

python -m llava.cmoa_eval.evaluate_mme \
    --eval_mixlora \
    --model-base "lmsys/vicuna-7b-v1.3" \
    --model-path $model_path \
    --question-dir $data_dir
