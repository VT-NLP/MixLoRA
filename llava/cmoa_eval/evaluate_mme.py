import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init, get_task_def, EVAL_TASKS, get_mme_test_list
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from .utils import load_pretrained_model

from PIL import Image
import math
import pdb
from pathlib import Path
import random
import torch.nn.functional as F


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)

    answers_dir = Path(args.model_path) / 'results'
    answers_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, image_processor, context_len, n_experts = load_pretrained_model(
        args.model_path, args.model_base, model_name, eval_mixlora=args.eval_mixlora)

    question_dir, file_list = get_mme_test_list(question_dir=args.question_dir)
    for question_file in file_list:
        task_def = get_task_def(question_file)
        args.question_file = question_file
        task_answers_dir = Path(answers_dir) / 'mme'
        task_answers_dir.mkdir(parents=True, exist_ok=True)
        task_answers_file = Path(task_answers_dir) / args.question_file
        args.question_file = os.path.join(question_dir, args.question_file)
        try:
            questions = [json.loads(q) for q in open(
                f"{args.question_file}", "r")]
        except:
            print(
                f'fail to open {args.question_file} . makes sure it exist')
            continue
        if os.path.exists(task_answers_file):
            try:
                answers = [json.loads(q) for q in open(
                    f"{task_answers_file}", "r")]
                if len(answers) == len(questions):
                    print(
                        f"Predictions at {task_answers_file} exists and has the same length as number of questions. skip it.")
                    continue
            except:
                print(f'regenerate predictions at {task_answers_file}')
            # spdb.set_trace()
        print(f"Testing {args.question_file}")
        print(f"Save predictions at {task_answers_file}")

        print(f'Totally {len(questions)} testing instances')

        print(
            f"a sample instance look like this:\n\n{questions[0]['prompt']}\n\nAnswer: {questions[0]['target']}")
        print(
            f"\nIt's image is at {os.path.join(question_dir, questions[0]['image_path'])}")
        # os.makedirs(os.path.dirname(task_answers_file), exist_ok=True)
        ans_file = open(task_answers_file, "w")
        if args.in_context:
            in_context_ex = random.choice(questions)
            in_context_q = in_context_ex['prompt']
            in_context_a = in_context_ex['target']
        for line in tqdm(questions):
            idx = line["unique_id"]

            image_file = os.path.join(question_dir, line["image_path"])
            if not os.path.exists(image_file):
                image_file = image_file.replace('.png', '.jpg')
            assert os.path.exists(image_file)
            try:
                image = Image.open(image_file)
            except:
                print(
                    f"Warning {image_file} exits but can not be opened !!!")
            qs = line["prompt"]
            cur_prompt = qs
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
                    DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            if args.in_context:
                qs = f"[Example]: [Input]: {in_context_q} [Output]: {in_context_a}||||{qs}"

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt().replace('[Options]', 'Options')

            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            image_tensor = image_processor.preprocess(
                image, return_tensors='pt')['pixel_values'][0]

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(
                keywords, tokenizer, input_ids)

            image_tensor = image_tensor.unsqueeze(0).half().cuda()

            task_def_id = None
            if getattr(model.config, "cond_type", None):
                if "task_def" in model.config.cond_type:
                    task_def_id = tokenizer_image_token(
                        task_def, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                    if input_ids.shape[1] > task_def_id.shape[1]:
                        task_def_id = F.pad(
                            task_def_id, (0, input_ids.shape[1] - task_def_id.shape[1]), "constant", tokenizer.pad_token_id)
                    else:
                        input_ids = F.pad(
                            input_ids, (0, task_def_id.shape[1] - input_ids.shape[1]), "constant", tokenizer.pad_token_id)
                    input_ids = torch.cat([input_ids, task_def_id], dim=0)
                elif getattr(model.config, "mix_mm_projector", False) or model.config.cond_type == 'task_embed':
                    task_def_id = tokenizer_image_token(
                        task_def, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    images=image_tensor,
                    task_def_ids=task_def_id,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            if getattr(model.config, "cond_type", None):
                if "task_def" in model.config.cond_type:
                    output_ids, _ = torch.chunk(output_ids, 2, dim=0)

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (
                input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(
                    f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                       "prompt": prompt,
                                       "predict": outputs,
                                       "target": line['target'],
                                       "image_path": image_file,
                                       "answer_id": ans_id,
                                       "model_id": model_name,
                                       "metadata": {}}) + "\n")
            # pdb.set_trace()
            ans_file.flush()
        ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str,
                        default="lmsys/vicuna-7b-v1.3")
    parser.add_argument("--question-file", type=str,
                        default="tables/question.jsonl")
    parser.add_argument("--question-dir", type=str,
                        default="/path/to/mixlora_data/mme/")
    parser.add_argument("--eval-file", type=str, default="answer.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--in-context", action='store_true', default=False)
    parser.add_argument("--training-set", action='store_true', default=False)
    parser.add_argument("--dist-token", action='store_true', default=False)
    parser.add_argument("--eval_mixlora", action='store_true', default=False)
    args = parser.parse_args()

    eval_model(args)
