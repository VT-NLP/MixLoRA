model_path=$1 # /path/to/model/
data_dir=$2 # /path/to/mixlora_data/mm_tasks


python -m llava.cmoa_eval.evaluate \
    --eval_mixlora \
    --model-base "lmsys/vicuna-7b-v1.3" \
    --question-dir $data_dir \
    --image-folder $data_dir \
    --model-path $model_path \
