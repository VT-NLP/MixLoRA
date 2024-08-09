import pdb
from torch import nn
from transformers.activations import get_activation

import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn import Module, ModuleList
import torch.nn.functional as F
import copy
import typing


class MixtureExperts(torch.nn.Module):

    def __init__(self, expert, num_experts):
        super(MixtureExperts, self).__init__()

        self.deepspeed_experts = torch.nn.ParameterList(
            [copy.deepcopy(expert) for i in range(num_experts)])
        self.num_experts = num_experts

    def get_expert_by_idx(self, idx):
        return self.deepspeed_experts[idx]

    def all_experts(self):
        
        all_weight = []
        for idx in range(self.num_experts):
            single_expert = self.deepspeed_experts[idx]
            all_weight.append(single_expert)
        
        all_weight = torch.cat(all_weight)
        
        return all_weight

    def forward(self, selection_score, expert_indices):
        """perform mixture of adpater based forward

        Args:
            x (Tensor): shape of [bs, seq_len, in_dim]
            selection_score (Tensor): shape of [bs, num_experts]

        Returns:
            mixture_prompt: 
        """
        
        all_weight = self.all_experts() # [k_experts, cmop_len, embed_dim]
        
        expert_pormpt = all_weight[expert_indices] # get the weights of experts we need -> [B, k_experts, cmop_len, embed_dim]
        
        expert_scores = torch.gather(selection_score, 1, expert_indices) # -> [B, k experts]
        mixture_prompt = expert_scores.unsqueeze(-1).unsqueeze(-1) * expert_pormpt
        
        mixture_prompt = torch.sum(mixture_prompt, dim=1) # -> [B, cmop_len, embed_dim]
        
        return mixture_prompt


class Experts(nn.Module):
    def __init__(self, 
                 expert,
                 n_experts):
        super().__init__()

        self.cmop_prompt = MixtureExperts(expert, n_experts)
        
    def forward(self, scores, indices):
        
        prompt = self.cmop_prompt(scores, indices)
        
        return prompt