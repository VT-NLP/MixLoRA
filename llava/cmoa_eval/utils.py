
import json
import os
from pathlib import Path
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import pdb

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE


def find_all_linear_names(model, mix_mm_projector=False):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if "cmoa_attribute_encoder" in name or "vision_tower" in name:
            continue
        if not mix_mm_projector:
            if "mm_projector" in name:
                continue

        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def load_pretrained_model(model_path, model_base, model_name, device_map="auto", output_selection=False, eval_mixlora=False):
    kwargs = {"device_map": device_map}

    kwargs['torch_dtype'] = torch.float16

    n_experts = None

    if eval_mixlora:
        from ..peft_cmoa import get_peft_model, PeftModel, CMoALoraConfig

        cmoa_args = torch.load(Path(model_path) / 'cmoa_args.bin')
        n_experts = cmoa_args.n_experts
        if not (Path(model_path) / 'config.json').exists():
            raise ValueError(f"config.json not found in {model_path}")
        else:
            config = AutoConfig.from_pretrained(
                Path(model_path) / 'config.json')
            model = LlavaCmoAForCausalLM.from_pretrained(
                model_base, config=config, low_cpu_mem_usage=True, **kwargs)

        if not cmoa_args.mix_mm_projector:
            cmoa_config = CMoALoraConfig(
                r=cmoa_args.n_selected,
                lora_alpha=cmoa_args.lora_alpha,
                target_modules=find_all_linear_names(
                    model, cmoa_args.mix_mm_projector),
                lora_dropout=cmoa_args.lora_dropout,
                bias=cmoa_args.lora_bias,
                n_experts=cmoa_args.n_experts,
                cond_type=cmoa_args.cond_type,
                attribute_dim=cmoa_args.attribute_dim,
                used_scored_weight=cmoa_args.used_scored_weight,
                independent_rank=cmoa_args.independent_rank,
                mix_loraA=cmoa_args.mix_loraA,
                mix_start_layer=cmoa_args.mix_start_layer,
                add_noise=False,
                ifs_weight=getattr(cmoa_args, "ifs_weight", 1),
                output_selection=output_selection,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, cmoa_config)

        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)

    else:
        from peft import PeftModel, PeftConfig

        config = PeftConfig.from_pretrained(model_path)
        model_config = AutoConfig.from_pretrained(
            Path(model_path) / 'config.json')
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_path, config=model_config, low_cpu_mem_usage=True, **kwargs)
        model = PeftModel.from_pretrained(model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path, use_fast=False)

    image_processor = None

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(
        model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device='cuda', dtype=torch.float16)
    image_processor = vision_tower.image_processor

    # security check
    # if (Path(model_path) / 'pytorch_model.bin').exists():
    state_dict = torch.load(Path(model_path) / 'pytorch_model.bin')
    model.load_state_dict(state_dict)

    model = model.to(device='cuda', dtype=torch.float16)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len, n_experts


def plot_expert_selection(lora, label_name, name, save_dir, nrows=16, ncols=2, figsize=(20, 30)):

    n_layer = 32
    n_experts = len(lora[0])

    # Create subplots
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for i in range(n_layer):
        row = i // ncols
        col = i % ncols
        axs[row, col].bar(range(n_experts), lora[i], label=label_name)
        axs[row, col].set_title(f"{name} (layer {i})")
        axs[row, col].set_xlabel('Experts')
        axs[row, col].tick_params(axis='x', rotation=90)

        axs[row, col].set_ylabel('Count')
        axs[row, col].set_xlim(0, n_experts - 1)
        axs[row, col].set_xticks(range(n_experts))
        axs[row, col].legend()

    # Adjust layout and display plot
    plt.tight_layout()
    plt.savefig(save_dir / f"{name}.png")
    plt.close()


def cluster_expert_selection_scores(data, name, save_dir, nrows=8, ncols=4, figsize=(20, 40), alpha=0.5, task_type_dict=None):

    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.color': 'gray'})

    # layers_to_plot = [0, 5, 25, 31]
    layers_to_plot = list(range(32))

    n_layer = len(layers_to_plot)
    n_tasks = len(data)
    palette = sns.color_palette("tab10", n_tasks)

    type_colors, task_name2type = {}, {}
    if task_type_dict is not None:
        for i, t_type in enumerate(task_type_dict.keys()):
            type_colors[t_type] = palette[i]
            for task_n in task_type_dict[t_type]:
                task_name2type[task_n] = t_type

    tasks = []
    task_color = {}
    task_list = []
    for i, task_name in enumerate(data.keys()):
        # len(data[task_name][0]): instance number
        tasks.extend([task_name] * len(data[task_name][0]))
        if task_type_dict is not None:
            task_color[task_name] = type_colors[task_name2type[task_name]]
        else:
            task_color[task_name] = palette[i]
        task_list.append(task_name)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # Dictionary to track unique labels

    for count, i in enumerate(layers_to_plot):
        unique_labels = {}
        row = count // ncols
        col = count % ncols

        scores = []
        for task_name in task_list:
            scores.extend(data[task_name][i])
        scores = np.array(scores)

        tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
        tsne_scores = tsne.fit_transform(scores)

        for task_name, point in zip(tasks, tsne_scores):
            if task_name not in unique_labels:
                axs[row, col].scatter(
                    point[0], point[1], color=task_color[task_name], label=task_name, alpha=alpha)
                unique_labels[task_name] = True
            else:
                axs[row, col].scatter(point[0], point[1],
                                      color=task_color[task_name], alpha=alpha)

        axs[row, col].set_title(f"{name} (layer {i})")
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])

        # Adjust legend placement
        handles, labels = axs[row, col].get_legend_handles_labels()
        labels = task_list
        ncol = max(1, len(labels) // 2) if len(labels) > 2 else len(labels)

        # axs[row, col].legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=ncol, frameon=True, edgecolor='black')
        axs[row, col].legend()

    plt.tight_layout()
    plt.savefig(save_dir / f"{name}.pdf", dpi=300)
    plt.close()


def cluster_expert_selection_index(data, name, save_dir, nrows=1, ncols=3, figsize=(10, 4), alpha=0.5, task_type_dict=None, paired=False, task_list=None):

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.color': 'gray'})

    layers_to_plot = [0, 15, 31]
    n_layer = len(layers_to_plot)
    n_tasks = len(data)
    if paired:
        palette = sns.color_palette("Paired")
        alpha = 0.5
    else:
        palette = sns.color_palette("tab10", n_tasks)

    type_colors, task_name2type = {}, {}
    if task_type_dict is not None:
        for i, t_type in enumerate(task_type_dict.keys()):
            type_colors[t_type] = palette[i]
            for task_n in task_type_dict[t_type]:
                task_name2type[task_n] = t_type

    tasks = []
    task_color = {}
    if task_list is None:
        task_list = ['text_vqa', 'visual_spatial_reasoning',
                     'snli_ve_classification', 'cifar_10', 'cifar_100', 'mnist', 'pope_pop']
    task_list_dict = {
        'image_text_selection': 'Image-Text',
        'text_vqa': 'Text-VQA',
        'visual_spatial_reasoning': 'VSR',
        'snli_ve_classification': 'SNLI-VE',
        'VQA_object_presence': 'VQA-Object-Presence',
        'cifar_10': 'CIFAR-10',
        'cifar_100': 'CIFAR-100',
        'PlotQA+visual_question_answering': 'PlotQA',
        'infographicvqa+single_document_question': 'InfoGraphicVQA',
        'GQA': 'GQA',
        'mnist': 'MNIST',
        'pope_pop': 'Pope',
        'ExDark+object_recognition': 'ExDark',
        'Core50+Object_detection': 'Core50+Object_detection',
    }
    # for i, task_name in enumerate(data.keys()):
    for i, task_name in enumerate(task_list):
        # len(data[task_name][0]): instance number
        tasks.extend([task_name] * len(data[task_name][0]))

        data[task_name] = (np.array(data[task_name]) != 0).astype(int).tolist()
        if task_type_dict is not None:
            task_color[task_name] = type_colors[task_name2type[task_name]]
        else:
            task_color[task_name] = palette[i]

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # Dictionary to track unique labels

    for count, i in enumerate(layers_to_plot):
        unique_labels = {}
        row = count // ncols
        col = count % ncols

        scores = []
        for task_name in task_list:
            scores.extend(data[task_name][i])
        scores = np.array(scores)

        # tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
        tsne = TSNE(n_components=2, perplexity=30,
                    learning_rate=200, metric='jaccard')
        tsne_scores = tsne.fit_transform(scores)

        for task_name, point in zip(tasks, tsne_scores):
            if task_name not in unique_labels:
                axs[col].scatter(
                    point[0], point[1], color=task_color[task_name], label=task_name, alpha=alpha)
                # axs[col].scatter(point[0], point[1], color=task_color[task_name], label=task_name, alpha=alpha)
                unique_labels[task_name] = True
            else:
                axs[col].scatter(point[0], point[1],
                                 color=task_color[task_name], alpha=alpha)
                # axs[col].scatter(point[0], point[1], color=task_color[task_name], alpha=alpha)

        axs[col].set_title(f"Layer {i}")
        axs[col].set_xticks([])
        axs[col].set_yticks([])

    # Assumes all subplots have the same handles and labels
    handles, labels = axs[0].get_legend_handles_labels()

    labels = [task_list_dict[l] for l in labels]
    if paired:
        fig.legend(handles, labels, loc='upper center',
                   bbox_to_anchor=(0.5, 1),
                   ncol=len(task_list) // 2,
                   frameon=True, edgecolor='gray',
                   columnspacing=0.1,            # Adjust space between columns
                   handletextpad=0.1,            # Adjust space between the handle and the text
                   #    borderpad=0.05,
                   labelspacing=0.1)

        plt.tight_layout(rect=[0, 0, 1, 0.85])
    else:
        fig.legend(handles, labels, loc='upper center',
                   bbox_to_anchor=(0.5, 1),
                   ncol=len(task_list),
                   frameon=True, edgecolor='gray',
                   columnspacing=0.1,            # Adjust space between columns
                   handletextpad=0.1,            # Adjust space between the handle and the text
                   #    borderpad=0.05,
                   labelspacing=0.1)

        # Adjust layout to create space for the legend
        # Adjust the left, bottom, right, top values as needed
        plt.tight_layout(rect=[0, 0, 1, 0.91])

    # plt.tight_layout(pad=3)
    plt.savefig(save_dir / f"sel_{name}.pdf", dpi=300)
    plt.close()


def cluster_expert_selection_index_all(data, name, save_dir, nrows=8, ncols=4, figsize=(20, 40), alpha=0.5, task_type_dict=None, paired=False, task_list=None):

    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.color': 'gray'})

    layers_to_plot = list(range(32))
    n_layer = len(layers_to_plot)
    n_tasks = len(data)
    if paired:
        palette = sns.color_palette("Paired")
        alpha = 1
    else:
        palette = sns.color_palette("tab10", n_tasks)

    type_colors, task_name2type = {}, {}
    if task_type_dict is not None:
        for i, t_type in enumerate(task_type_dict.keys()):
            type_colors[t_type] = palette[i]
            for task_n in task_type_dict[t_type]:
                task_name2type[task_n] = t_type

    tasks = []
    task_color = {}
    if task_list is None:
        task_list = ['text_vqa', 'visual_spatial_reasoning',
                     'snli_ve_classification', 'cifar_10', 'cifar_100', 'mnist', 'pope_pop']

    task_list_dict = {
        'image_text_selection': 'Image-Text',
        'text_vqa': 'Text-VQA',
        'visual_spatial_reasoning': 'VSR',
        'snli_ve_classification': 'SNLI-VE',
        'VQA_object_presence': 'VQA-Object-Presence',
        'cifar_10': 'CIFAR-10',
        'cifar_100': 'CIFAR-100',
        'mnist': 'MNIST',
        'pope_pop': 'Pope',
        'ExDark+object_recognition': 'ExDark+object_recognition',
        'Core50+Object_detection': 'Core50+Object_detection',
    }
    for i, task_name in enumerate(task_list):
        # len(data[task_name][0]): instance number
        tasks.extend([task_name] * len(data[task_name][0]))

        data[task_name] = (np.array(data[task_name]) != 0).astype(int).tolist()
        if task_type_dict is not None:
            task_color[task_name] = type_colors[task_name2type[task_name]]
        else:
            task_color[task_name] = palette[i]
        # task_list.append(task_name)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # Dictionary to track unique labels

    for count, i in enumerate(layers_to_plot):
        unique_labels = {}
        row = count // ncols
        col = count % ncols

        scores = []
        for task_name in task_list:
            scores.extend(data[task_name][i])
        scores = np.array(scores)

        # tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
        tsne = TSNE(n_components=2, perplexity=30,
                    learning_rate=200, metric='jaccard')
        tsne_scores = tsne.fit_transform(scores)

        for task_name, point in zip(tasks, tsne_scores):
            if task_name not in unique_labels:
                axs[row, col].scatter(
                    point[0], point[1], color=task_color[task_name], label=task_name, alpha=alpha)
                # axs[col].scatter(point[0], point[1], color=task_color[task_name], label=task_name, alpha=alpha)
                unique_labels[task_name] = True
            else:
                axs[row, col].scatter(point[0], point[1],
                                      color=task_color[task_name], alpha=alpha)
                # axs[col].scatter(point[0], point[1], color=task_color[task_name], alpha=alpha)

        axs[row, col].set_title(f"Layer {i}")
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])

    # Assumes all subplots have the same handles and labels
    handles, labels = axs[0, 0].get_legend_handles_labels()

    labels = [task_list_dict[l] for l in labels]
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(
        0.5, 1), ncol=len(task_list) // 2, frameon=True, edgecolor='black')

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # plt.tight_layout(pad=3)
    plt.savefig(save_dir / f"{name}.pdf", dpi=300)
    plt.close()
