from collections import defaultdict
import math
import os
import pdb
import shutil
from types import SimpleNamespace
import numpy as np
import torch

from transformers import Trainer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.utils import (is_peft_available, is_accelerate_available,
                                is_apex_available,
                                is_datasets_available,
                                is_in_notebook,
                                is_ipex_available,
                                is_safetensors_available,
                                is_sagemaker_dp_enabled,
                                is_sagemaker_mp_enabled,
                                is_torch_compile_available,
                                is_torch_neuroncore_available,
                                is_torch_tpu_available,
                                logging,
                                strtobool)
import torch.distributed as dist
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.integrations import (
    hp_params,
)

from transformers.trainer_callback import (
    TrainerControl,
    TrainerState,
)
from transformers.deepspeed import deepspeed_init, deepspeed_load_checkpoint
from deepspeed.utils import safe_get_full_grad
from transformers.trainer_utils import has_length, ShardedDDPOption, speed_metrics, HPSearchBackend, TrainOutput
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from pathlib import Path
import time
from torch import nn

from torch.utils.data import Sampler

from transformers.trainer_pt_utils import (
    get_model_param_count,
)

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import sys
from packaging import version

logger = logging.get_logger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

CMOA_ARGS_NAME = "cmoa_args.bin"

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin

    if version.parse(accelerate_version) > version.parse("0.20.3"):
        from accelerate.utils import (
            load_fsdp_model,
            load_fsdp_optimizer,
            save_fsdp_model,
            save_fsdp_optimizer,
        )
if is_apex_available():
    from apex import amp


if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(
        SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

# target_tasks = {214: 'scienceqa+question_answering', 65: 'cinic-10+image_classification_animal', 58: 'fairface+image_classification_age', 147: 'INATURALIST+order_classification', 123: 'STVQA+image_question_answer', 115: 'PACS+elephant_image_category_classification'}


class TaskSpecificSampler(Sampler):
    def __init__(self,
                 batch_size,
                 dataset=None,
                 ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.task_batches = self._create_task_batches()

    def _create_task_batches(self):
        batches = []
        for task_id, instance_indices in self.dataset.task_to_instances.items():
            # Shuffle the indices for each task
            np.random.shuffle(instance_indices)
            # Create batches
            batches.extend([instance_indices[i:i + self.batch_size]
                            for i in range(0, len(instance_indices), self.batch_size)])
        np.random.shuffle(batches)  # Shuffle batches to mix tasks
        return batches

    def __iter__(self):
        for batch in self.task_batches:
            yield from batch

    def __len__(self):
        return len(self.dataset)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(
        key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu()
                 for k, v in to_return.items()}
    return to_return


def gather_all_task_ids_across_ranks(local_task_ids):
    """
    Gathers all unique task IDs from all ranks and returns a combined list of unique task IDs.

    :param local_task_ids: A list or tensor of task IDs processed by the local rank.
    :return: A combined list of unique task IDs across all ranks.
    """

    # Gather the size of task ID lists from all ranks
    local_size = torch.tensor([local_task_ids.numel()], device='cuda')
    sizes = [torch.tensor([0], device='cuda')
             for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, local_size)

    # Compute the maximum size and prepare tensors for all_gather
    max_size = max(size.item() for size in sizes)
    padded_local_task_ids = torch.cat([local_task_ids, torch.zeros(
        max_size - local_task_ids.numel(), device='cuda', dtype=torch.long)])

    # Gather task IDs from all ranks
    gathered_task_ids = [torch.zeros(
        max_size, device='cuda', dtype=torch.long) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_task_ids, padded_local_task_ids)

    # Combine and deduplicate task IDs
    combined_task_ids = torch.cat(gathered_task_ids).unique().tolist()

    return combined_task_ids


def reset_record_layers(record_layers, check_layers, cmoa=False, n_experts=0):
    layers = list(range(32))
    # layers = [5, 10, 25, 31]
    if cmoa:
        # base_model.model.model.layers.0.self_attn.k_proj.lora_A.default.4
        for check_layer in check_layers:
            for layer_i in layers:
                record_layers[f"layers.{layer_i}.{check_layer}"] = {
                    f"lora_A.default.{n}": [] for n in range(n_experts)
                }
                record_layers[f"layers.{layer_i}.{check_layer}"].update({
                    f"lora_B.default.{n}": [] for n in range(n_experts)
                })
    else:
        for check_layer in check_layers:
            for layer_i in layers:
                record_layers[f"layers.{layer_i}.{check_layer}"] = {
                    "lora_A.default.weight": [],
                    "lora_B.default.weight": [],
                }


def record_gradient(model, record_layers, is_deepspeed_enabled=False):
    for name, p in model.named_parameters():
        for layer_name, layer_dict in record_layers.items():
            if layer_name in name:
                for param_name in layer_dict.keys():
                    # prefix_param = '.'.join(param_name.split('.')[:-1])
                    if param_name in name and param_name.split('.')[-1] == name.split('.')[-1]:
                        if is_deepspeed_enabled:
                            grad = safe_get_full_grad(p)
                            if dist.get_rank() == 0:
                                record_layers[layer_name][param_name].append(
                                    grad.flatten().detach().cpu())
                        else:
                            grad = p.grad.flatten().detach().cpu()
                            record_layers[layer_name][param_name].append(grad)


def record_gradient_alignment(task_grads, record_layers):
    """compute the cosine similarity between gradients of different tasks"""
    task_pairs = [(task_id_1, task_id_2) for i, task_id_1 in enumerate(task_grads.keys())
                  for task_id_2 in list(task_grads.keys())[i+1:]]

    for task_id_1, task_id_2 in task_pairs:
        for layer_name in task_grads[task_id_1].keys():
            for param_name in task_grads[task_id_1][layer_name].keys():
                cos_sim = nn.functional.cosine_similarity(task_grads[task_id_1][layer_name][param_name],
                                                          task_grads[task_id_2][layer_name][param_name], dim=0)

                # if layer_name in record_layers:
                record_layers[layer_name][param_name].append(cos_sim.item())
                # check_layer = '.'.join(layer_name.split('.')[2:])
                # record_layers[f"layers.all.{check_layer}"][param_name].append(cos_sim.item())

    for layer_name in record_layers.keys():
        for param_name in record_layers[layer_name].keys():
            record_layers[layer_name][param_name] = torch.tensor(
                record_layers[layer_name][param_name]).mean().item()


def record_gradient_alignment_cmoa(task_grads, record_layers, n_experts):
    """compute the cosine similarity between gradients of different tasks"""
    # Precompute the unique pairs of tasks
    task_pairs = [(task_id_1, task_id_2) for i, task_id_1 in enumerate(task_grads.keys())
                  for task_id_2 in list(task_grads.keys())[i+1:]]
    check_layers = ["mlp.down_proj", "mlp.up_proj", "mlp.gate_proj",
                    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]

    for task_id_1, task_id_2 in task_pairs:
        for check_layer in check_layers:
            A_grads_1, A_grads_2 = [], []
            B_grads_1, B_grads_2 = [], []
            for layer_i in range(32):
                layer_name = f"layers.{layer_i}.{check_layer}"

                for n in range(n_experts):
                    A_param_name = f"lora_A.default.{n}"
                    A_grads_1.append(
                        task_grads[task_id_1][layer_name][A_param_name])
                    A_grads_2.append(
                        task_grads[task_id_2][layer_name][A_param_name])
                    B_param_name = f"lora_B.default.{n}"
                    B_grads_1.append(
                        task_grads[task_id_1][layer_name][B_param_name])
                    B_grads_2.append(
                        task_grads[task_id_2][layer_name][B_param_name])
            cos_sim_A = nn.functional.cosine_similarity(torch.stack(A_grads_1),
                                                        torch.stack(A_grads_2), dim=1)
            cos_sim_B = nn.functional.cosine_similarity(torch.stack(B_grads_1),
                                                        torch.stack(B_grads_2), dim=1)

            c = 0
            for layer_i in range(32):
                layer_name = f"layers.{layer_i}.{check_layer}"
                for n in range(n_experts):
                    A_param_name = f"lora_A.default.{n}"
                    B_param_name = f"lora_B.default.{n}"
                    record_layers[layer_name][A_param_name].append(
                        cos_sim_A[c].item())
                    record_layers[layer_name][B_param_name].append(
                        cos_sim_B[c].item())
                    c += 1

    for layer_name, params in record_layers.items():
        for param_name, values in params.items():
            record_layers[layer_name][param_name] = torch.tensor(
                values).mean().item()


class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if getattr(self.args, "same_task_batch", False):
            return TaskSpecificSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
            )
        else:
            return super()._get_train_sampler()

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(
                self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(
                    output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            Path(output_dir).mkdir(exist_ok=True, parents=True)
            torch.save(state_dict, Path(output_dir) / 'pytorch_model.bin')
            self.model.config.save_pretrained(output_dir)
            # super(LLaVATrainer, self)._save(output_dir, state_dict)
            torch.cuda.empty_cache()

            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving model checkpoint to {output_dir}")

            if getattr(self.args, "use_cmoa", False):
                from ..peft_cmoa import PeftModel
            else:
                from peft import PeftModel

            supported_classes = (PreTrainedModel,) if not is_peft_available() else (
                PreTrainedModel, PeftModel)
            # Save a trained model and configuration using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            if not isinstance(self.model, supported_classes):
                if state_dict is None:
                    state_dict = self.model.state_dict()

                if isinstance(unwrap_model(self.model), supported_classes):
                    unwrap_model(self.model).save_pretrained(
                        output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                    )
                else:
                    print(
                        "Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                    torch.save(state_dict, os.path.join(
                        output_dir, "pytorch_model.bin"))
                    torch.cuda.empty_cache()
            else:
                self.model.save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
                torch.cuda.empty_cache()

            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(self.args, os.path.join(
                output_dir, "training_args.bin"))

            if getattr(self.args, "cond_type", False):
                cmoa_args = SimpleNamespace(
                    cond_type=self.args.cond_type,
                    n_experts=self.args.n_experts,
                    n_selected=self.args.n_selected,
                    lora_alpha=self.args.lora_alpha,
                    lora_dropout=self.args.lora_dropout,
                    lora_bias=self.args.lora_bias,
                    mix_mm_projector=self.args.mix_mm_projector,
                    attribute_dim=self.args.attribute_dim,
                    used_scored_weight=self.args.used_scored_weight,
                    independent_rank=self.args.independent_rank,
                    mix_loraA=self.args.mix_loraA,
                    mix_start_layer=self.args.mix_start_layer,
                    ifs_weight=self.args.ifs_weight
                )
                torch.save(cmoa_args, os.path.join(
                    output_dir,  CMOA_ARGS_NAME))

        torch.cuda.empty_cache()

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval, extra_loss=None):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            if extra_loss is not None:
                extra_loss_scalars = {}
                for key in extra_loss.keys():
                    extra_loss_scalars[key] = self._nested_gather(
                        extra_loss[key]).mean().item()
                    extra_loss[key] -= extra_loss[key]
                    logs[key] = round(
                        extra_loss_scalars[key] / (self.state.global_step - self._globalstep_last_logged), 4)
                    self._extra_loss_scalar[key] += extra_loss_scalars[key]

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], step=1) -> torch.Tensor:
        task_loss = None
        model.train()
        inputs = self._prepare_inputs(inputs)
        # inputs["input_ids"] = torch.zeros((16, 1792), device=inputs["input_ids"].device, dtype=torch.long)
        # inputs["labels"] = torch.zeros((16, 1792), device=inputs["input_ids"].device, dtype=torch.long)
        # inputs["attention_mask"] = torch.ones((16, 1792), device=inputs["input_ids"].device, dtype=torch.long)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(
                model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            if getattr(self.args, "use_contrastive_instr_token", False):
                loss, outputs = self.compute_loss(
                    model, inputs, return_outputs=True)
                extra_loss = {"instr_loss": outputs.instr_loss,
                              "o_loss": outputs.o_loss}
            else:

                loss = self.compute_loss(model, inputs)
                extra_loss = None

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return (loss.detach() / self.args.gradient_accumulation_steps, extra_loss)

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        logger.debug(
            f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        print("train_dataloader: ", len(train_dataloader))
        print("train_dataset: ", len(self.train_dataset))

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * \
            args.gradient_accumulation_steps * args.world_size
        print("total_train_batch_size: ", total_train_batch_size)

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(
                    args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(
                    train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(
                    max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # Fairscale Sharded DDP, FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(
                        self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # deepspeed ckpt loading
        if resume_from_checkpoint is not None and self.is_deepspeed_enabled:
            deepspeed_load_checkpoint(
                self.model_wrapped, resume_from_checkpoint)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(
            f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(
                f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(
            f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        self.n_experts = getattr(self.args, "n_experts", 0)
        self.use_cmoa = getattr(self.args, "use_cmoa", False)
        self.record_steps = getattr(self.args, "record_steps", -1)

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (
                    num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(
                f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)

        if getattr(self.args, "use_contrastive_instr_token", False):
            extra_loss = {"instr_loss": torch.tensor(0.0).to(
                args.device), "o_loss": torch.tensor(0.0).to(args.device)}
            self._extra_loss_scalar = {"instr_loss": 0.0, "o_loss": 0.0}
        else:
            extra_loss = None
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                for _ in train_dataloader:
                    break

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(
                    epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(
                        args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    tr_loss_step, extra_loss_step = self.training_step(
                        model, inputs, step=self.state.global_step)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / \
                        (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                if extra_loss_step is not None:
                    for key in extra_loss.keys():
                        if (
                            args.logging_nan_inf_filter
                            and not is_torch_tpu_available()
                            and (torch.isnan(extra_loss_step[key]) or torch.isinf(extra_loss_step[key]))
                        ):
                            # if loss is nan or inf simply add the average of previous logged losses
                            extra_loss[key] += extra_loss[key] / \
                                (1 + self.state.global_step -
                                 self._globalstep_last_logged)
                        else:
                            extra_loss[key] += extra_loss_step[key]

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (
                        step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):

                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc or (
                        version.parse(
                            accelerate_version) <= version.parse("0.20.3")
                    ):
                        self.accelerator.gradient_state._set_sync_gradients(
                            True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients,
                                              scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(
                                args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            # tpu-comment: accelerate wrapped optimizers call xm.optimizer_step
                            self.optimizer.step()
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()
                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped

                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + \
                        (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(
                        args, self.state, self.control)

                    self._maybe_log_save_evaluate(
                        tr_loss, model, trial, epoch, ignore_keys_for_eval, extra_loss=extra_loss)
                else:
                    self.control = self.callback_handler.on_substep_end(
                        args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(
                args, self.state, self.control)
            self._maybe_log_save_evaluate(
                tr_loss, model, trial, epoch, ignore_keys_for_eval, extra_loss=extra_loss)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metric_loss = {}
        if extra_loss_step is not None:
            for key in extra_loss.keys():
                self._extra_loss_scalar[key] += extra_loss[key].item()
                metric_loss[key] = self._extra_loss_scalar[key] / \
                    self.state.global_step

        metrics = speed_metrics(
            "train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss
        if extra_loss_step is not None:
            for key in metric_loss.keys():
                metrics[key] = metric_loss[key]

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(
            use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(
                        f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)
