#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import pdb
from typing import List, Optional, Tuple, Union
import copy
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaForCausalLM, LlamaModel
                         
from .cmoa.modeling_llama_cmoa import LlamaCMoAModel                         


from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from dataclasses import dataclass
from transformers.utils import ModelOutput
@dataclass
class CausalLMOutputWithPast(ModelOutput):


    loss: Optional[torch.FloatTensor] = None
    o_loss: Optional[torch.FloatTensor] = None
    instr_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class LlavaCmoAConfig(LlamaConfig):
    model_type = "llava_cmoa"


class LlavaCmoAModel(LlavaMetaModel, LlamaCMoAModel):

    config_class = LlavaCmoAConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaCmoAModel, self).__init__(config)


class LlavaCmoAForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaCmoAConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaCmoAModel(config)
        self.use_contrastive_instr_token = getattr(config, "use_contrastive_instr_token", False)
        
        if self.use_contrastive_instr_token:
            self.log_vars = nn.Parameter(torch.zeros((2)))

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.cond_type = getattr(config, "cond_type", None)
        self.mix_mm_projector = getattr(config, "mix_mm_projector", False)

        # Initialize weights and apply final processing
        self.post_init()
        
        if self.mix_mm_projector or self.cond_type == 'task_embed':
            self.task_encoder = LlamaForCausalLM.from_pretrained(config.task_model_name_or_path)
            self.task_encoder.requires_grad_(False)
        

    def get_model(self):
        return self.model
    
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlavaCmoAModel):
            module.gradient_checkpointing = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        task_def_ids: torch.LongTensor = None,
        task_attention_mask: Optional[torch.Tensor] = None,
        attribute_ids: torch.LongTensor = None,
        task_ids: torch.LongTensor = None,
        evaluate: bool = False
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if not self.training:
            labels = copy.deepcopy(input_ids)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.mix_mm_projector or self.cond_type == 'task_embed':
            task_embed = self.task_encoder(input_ids=task_def_ids,
                                            attention_mask=task_attention_mask, 
                                            use_cache=False,
                                            output_hidden_states=True)['hidden_states'][-1]
        else:
            task_embed = None
        
        input_ids, attention_mask, past_key_values, inputs_embeds, labels, task_embed = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images, task_embed=task_embed)
        

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            task_embed=task_embed,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # attribute_ids=attribute_ids
        )

        hidden_states = outputs[0]
        if self.use_contrastive_instr_token:
            instr_token = outputs.instruction_token
            
        if "task_def" in self.cond_type:
            hidden_states, _ = torch.chunk(hidden_states, 2, dim=0)
            if labels is not None and self.training:
                labels, _ = torch.chunk(labels, 2, dim=0)
            
        logits = self.lm_head(hidden_states)
        
        if "task_def" in self.cond_type and labels is None:
            logits = torch.cat([logits, logits], dim=0)
        loss, total_loss, o_loss, instr_loss = None, None, None, None
        if labels is not None and self.training:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            
            if self.use_contrastive_instr_token:
                instr_embeddings = instr_token / instr_token.norm(dim=-1, keepdim=True)
                sim = instr_embeddings @ instr_embeddings.t()
                # logits_instr = self.logit_scale.exp() * sim
                # construct true logits
                task_equals = task_ids.unsqueeze(0) == task_ids.unsqueeze(1)
                true_logits = task_equals.long().to(instr_embeddings.device)
                # if task_equals.sum() ==  batch_size: # no positive examples
                #     return 0.0
                # self-supervised contrastive loss
                # based on https://arxiv.org/pdf/2004.11362.pdf
                instruction_loss = 0.
                for bs in range(instr_embeddings.shape[0]):
                    pos_sim = sim[bs,task_equals[bs]]
                    instr_loss_i = - torch.log(torch.exp(pos_sim) / torch.exp(sim[bs]).sum()).mean()
                    instruction_loss += instr_loss_i 
                
                precision_original = torch.exp(-self.log_vars[0])
                precision_instr = torch.exp(-self.log_vars[1])
                total_loss = precision_instr * instruction_loss + precision_original * loss + \
                    0.5 * (self.log_vars[0] + self.log_vars[1])
                o_loss = loss
                instr_loss = instruction_loss
            else:
                total_loss = loss
                o_loss = None
                instr_loss = None
            

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=total_loss,
            o_loss=o_loss,
            instr_loss=instr_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, attribute_ids=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "task_def_ids": kwargs.get("task_def_ids", None),
                "task_attention_mask": kwargs.get("task_attention_mask", None),
            }
        )
        return model_inputs

AutoConfig.register("llava_cmoa", LlavaCmoAConfig)
AutoModelForCausalLM.register(LlavaCmoAConfig, LlavaCmoAForCausalLM)
