import pdb
from torch import nn
from transformers.activations import get_activation

import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn import Module, ModuleList
import torch.nn.functional as F
import copy
from einops import rearrange, repeat
from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple, Union, cast

"""
This file contains the code for selecting experts.
"""

ATTRIBUTE = 'attr'
INPUT = 'input'
INSTRUCTION = 'instr'
TASK_DEF = 'task_def'
TASK_EMBED = 'task_embed'

normal_map: Dict[torch.device, Callable] = {}


def normal_rsample(shape: Tuple, device: torch.device, num_expert: int) -> Tensor:
    normal = normal_map.get(device)
    if normal is None:
        std = torch.tensor(1.0/num_expert, device=device)
        mean = torch.tensor(0.0, device=device)
        normal = torch.distributions.normal.Normal(
            mean, std).rsample  # type: ignore
        normal_map[device] = normal
    return normal(shape)


class CMoALoRASelector(nn.Module):
    """
    Selector conditioned on input_token and attribute templates
    """

    def __init__(self, dim, num_experts=4, r=4, attr_dim=32, add_noise=False, cond_type=None, independent_rank=True, mix_loraA=True, mix_loraB=True):
        super().__init__()

        self.add_noise = add_noise
        self.num_experts = num_experts
        self.r = r
        self.noise_std = 1
        self.cond_type = cond_type

        in_dim = 0
        if INSTRUCTION in self.cond_type:
            in_dim += dim
        if TASK_DEF in self.cond_type:
            in_dim += dim
        if TASK_EMBED in self.cond_type:
            in_dim += dim
        if INPUT in self.cond_type:
            in_dim += dim
        if ATTRIBUTE in self.cond_type:
            in_dim += dim
        self.mix_loraA, self.mix_loraB = mix_loraA, mix_loraB
        if self.mix_loraA:
            self.loraA_selector = nn.Linear(in_dim, num_experts, bias=False)

        if self.mix_loraB:
            self.loraB_selector = nn.Linear(in_dim, num_experts, bias=False)

        self.independent_rank = independent_rank

        if not self.independent_rank:
            self.loraB_selector = None
            self.loraB_selector_v = None
            self.loraB_selector_t = None
            self.loraB_selector_mm = None

    def select(self, logits, r, output_score=False):

        if self.add_noise:
            logits = logits + normal_rsample(logits.shape,
                                             device=logits.device,
                                             num_expert=r / self.noise_std)

        topk_indices = torch.topk(logits, r, dim=-1).indices

        topk_indices = topk_indices.view(-1, r)

        if output_score:
            score = F.softmax(logits, dim=-1)
            # mask scores beyond top-k
            mask = torch.zeros([topk_indices.shape[0], self.num_experts],
                               device=topk_indices.device,
                               dtype=topk_indices.dtype)
            mask.scatter_(1, topk_indices, 1)

            masked_score = score.view(-1, self.num_experts) * mask

            # normalize the score
            denom_s = torch.clamp(masked_score.sum(dim=1),
                                  min=torch.finfo(score.dtype).eps)

            norm_score = masked_score / \
                denom_s.unsqueeze(1).expand_as(masked_score)
        else:
            norm_score = None

        return norm_score, topk_indices

    def forward(self, input_x, instr_x=None, attr_x=None, task_x=None):
        """_summary_

        Args:
            input_x (_type_): [bz, seq, dim]
            instr_x (_type_, optional): [bz, dim]. Defaults to None.
            attr_x (_type_, optional): [bz, seq, dim]. Defaults to None.
            task_x (_type_, optional): [bz, dim]. Defaults to None.

        Returns:
            _type_: _description_
        """

        # [instr, input, attr]
        x = None

        if INSTRUCTION in self.cond_type:
            x = instr_x

        if TASK_DEF in self.cond_type or TASK_EMBED in self.cond_type:
            if x is not None:
                x = torch.cat([x, task_x.mean(dim=1)], dim=-1)
            else:
                x = task_x.mean(dim=1)

        if INPUT in self.cond_type:
            if x is not None:
                x = torch.cat([x, input_x.mean(dim=1)], dim=-1)
            else:
                x = input_x.mean(dim=1)

        if ATTRIBUTE in self.cond_type:
            if x is not None:
                x = torch.cat([x, attr_x.mean(dim=1)], dim=-1)
            else:
                x = attr_x.mean(dim=1)

        x = x.to(input_x.dtype)

        if self.mix_loraA:

            loraA_logits = self.loraA_selector(x)
            loraA_scores, loraA_indices = self.select(
                loraA_logits, r=self.r)  # [(bz seq_len), r]
        else:
            loraA_scores, loraA_indices = None, None

        if not self.independent_rank:
            return loraA_scores, loraA_scores, loraA_indices, loraA_indices
        else:
            if self.mix_loraB:

                loraB_logits = self.loraB_selector(x)
                loraB_scores, loraB_indices = self.select(
                    loraB_logits, r=self.r)
            else:
                loraB_scores, loraB_indices = None, None

        return loraA_scores, loraB_scores, loraA_indices, loraB_indices


class CMoALoRA2B_Selector(nn.Module):
    """
    Selector conditioned on selection on LoRA_A
    """

    def __init__(self, dim, num_experts=4, r=4, add_noise=False, cond_type=None, ifs_weight=1):
        super().__init__()

        self.add_noise = add_noise
        self.num_experts = num_experts
        self.r = r
        self.noise_std = 1
        self.cond_type = cond_type
        self.ifs_weight = ifs_weight

        in_dim = 0
        if INSTRUCTION in self.cond_type:
            in_dim += dim
        if TASK_DEF in self.cond_type:
            in_dim += dim
        if INPUT in self.cond_type:
            in_dim += dim
        if ATTRIBUTE in self.cond_type:
            in_dim += dim

        self.loraB_selector = nn.Linear(in_dim, num_experts, bias=False)

        if 'lora_a_choice' in self.cond_type:
            self.loraA_choice_embed = nn.Embedding(num_experts, in_dim)
            self.loraA_choice2loraB_selector = nn.Linear(
                in_dim, num_experts, bias=False)

        if 'lora_a_param' in self.cond_type:
            # d_modal = 512
            # self.loraA_param_proj = nn.Linear(in_dim, d_modal)
            # encoder_layer = nn.TransformerEncoderLayer(d_model=d_modal, nhead=4)
            # self.loraA_param_trans = nn.TransformerEncoder(encoder_layer, num_layers=2)
            # self.loraA_param2loraB_selector = nn.Linear(d_modal, num_experts, bias=False)
            self.loraA_param2loraB_selector = nn.Linear(
                r * in_dim, num_experts, bias=False)

    def select(self, logits1, logits2, output_score=False):

        logits = F.softmax(logits1, dim=-1) + \
            self.ifs_weight * F.softmax(logits2, dim=-1)

        if self.add_noise:
            logits = logits + normal_rsample(logits.shape,
                                             device=logits.device,
                                             num_expert=self.r / self.noise_std)

        topk_indices = torch.topk(logits, self.r, dim=-1).indices

        topk_indices = topk_indices.view(-1, self.r)

        if output_score:
            score = F.softmax(logits, dim=-1)

            # mask scores beyond top-k
            mask = torch.zeros([topk_indices.shape[0], self.num_experts],
                               device=topk_indices.device,
                               dtype=topk_indices.dtype)
            mask.scatter_(1, topk_indices, 1)

            masked_score = score.view(-1, self.num_experts) * mask

            # normalize the score
            denom_s = torch.clamp(masked_score.sum(dim=1),
                                  min=torch.finfo(score.dtype).eps)

            norm_score = masked_score / \
                denom_s.unsqueeze(1).expand_as(masked_score)
        else:
            norm_score = None

        return norm_score, topk_indices

    def forward(self, input_x, instr_x=None, attr_x=None, task_x=None, loraA_indices=None, lora_A_param=None):
        """_summary_

        Args:
            input_x (_type_): [bz, seq, dim]
            instr_x (_type_, optional): [bz, dim]. Defaults to None.
            attr_x (_type_, optional): [bz, seq, dim]. Defaults to None.
            task_x (_type_, optional): [bz, dim]. Defaults to None.
            loraA_indices (_type_, optional): [bz, r]. Defaults to None.
            lora_A_param (_type_, optional): [bz r dim]. Defaults to None.

        Returns:
            _type_: _description_
        """

        x = None

        if INSTRUCTION in self.cond_type:
            x = instr_x

        if TASK_DEF in self.cond_type:
            if x is not None:
                x = torch.cat([x, task_x.mean(dim=1)], dim=-1)
            else:
                x = task_x.mean(dim=1)

        if INPUT in self.cond_type:
            if x is not None:
                x = torch.cat([x, input_x.mean(dim=1)], dim=-1)
            else:
                x = input_x.mean(dim=1)

        if ATTRIBUTE in self.cond_type:
            if x is not None:
                x = torch.cat([x, attr_x.mean(dim=1)], dim=-1)
            else:
                x = attr_x.mean(dim=1)

        x = x.to(input_x.dtype)

        loraB_logits1 = self.loraB_selector(x)

        if 'lora_a_choice' in self.cond_type:
            hidden_states = self.loraA_choice_embed(
                loraA_indices)  # [bz, r, dim]
            hidden_states = hidden_states.mean(dim=1)  # [bz, dim]
            loraB_logits2 = self.loraA_choice2loraB_selector(hidden_states)

        if 'lora_a_param' in self.cond_type:
            # hidden_states = self.loraA_param_proj(lora_A_param)
            # hidden_states = self.loraA_param_trans(hidden_states) # [bz, r, d_model]
            # hidden_states = hidden_states.mean(dim=1)

            # hidden_states = lora_A_param.mean(dim=1)
            # loraB_logits2 = self.loraA_param2loraB_selector(hidden_states)
            loraB_logits2 = torch.einsum('brd,rfe->bre', lora_A_param, rearrange(
                self.loraA_param2loraB_selector.weight, 'E (r dim)-> r dim E', r=self.r))
            loraB_logits2 = F.softmax(loraB_logits2, dim=-1)
            loraB_logits2 = loraB_logits2.sum(dim=1)

        loraB_scores, loraB_indices = self.select(loraB_logits1, loraB_logits2)

        return loraB_scores, loraB_indices
