from transformers import AutoModelForCausalLM,AutoModel
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import Trainer
from torch import nn
import heapq
import torch
import random
import os
import pdb
import json
import pandas as pd
from torch.nn.functional import cosine_similarity


def spars(tensor):

    k = int(tensor.numel() * 0.2)

    indices = torch.argsort(tensor.abs().view(-1), descending=True)[:k]

    mask = torch.zeros_like(tensor)
    mask.view(-1)[indices] = 1
    tensor.mul_(mask)
    return tensor

path_model = "/mnt2/name/model/llama3_1_8b_instruct"

pre_model = "/mnt2/name/model/LLM-Research/Meta-Llama-3___1-8B"

tokenizer = AutoTokenizer.from_pretrained(path_model, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(path_model, torch_dtype=torch.bfloat16).cuda()

model_pre = AutoModelForCausalLM.from_pretrained(pre_model, torch_dtype=torch.bfloat16).cuda()

config = AutoConfig.from_pretrained(path_model)

qwen_template = "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}<|im_end|>\n"
llama_template = "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{}<|eot_id|>"
mistral_template = "[INST] {}[/INST]  {}</s>"
template = llama_template

model.eval()
random.seed(42)

idx = [torch.tensor(23),torch.tensor(24),torch.tensor(25),torch.tensor(26),torch.tensor(27),torch.tensor(28),torch.tensor(29),torch.tensor(30)]

layers = model.model.layers
layers_pre = model_pre.model.layers

for (name1, param1), (name2, param2), (name3, param3), (name4, param4), (name5, param5), (name6, param6),(name7, param7), (name8, param8),(name9, param9), (name10, param10) in zip(layers[idx[0]-1].named_parameters(), layers[idx[0]].named_parameters(),layers[idx[1]].named_parameters(),layers[idx[2]].named_parameters(),layers[idx[3]].named_parameters(),layers_pre[idx[0]-1].named_parameters(),layers_pre[idx[0]].named_parameters(),layers_pre[idx[1]].named_parameters(),layers_pre[idx[2]].named_parameters(),layers_pre[idx[3]].named_parameters()):
    if 'input_layernorm' in name1 or  'post_attention_layernorm' in name1:
        continue
    if name1 == name2:
        new_param = param6 + 0.8*((param1-param6) + spars((param2-param7)) + spars((param3-param8))+ spars((param4-param9)) +spars((param5-param10)))
        with torch.no_grad():
            param1.copy_(new_param) 
layers_middle_1 = layers[idx[0]-1]

for (name1, param1), (name2, param2), (name3, param3), (name4, param4), (name5, param5), (name6, param6), (name7, param7), (name8, param8),(name9, param9), (name10, param10) in zip(layers[idx[7]+1].named_parameters(), layers[idx[4]].named_parameters(), layers[idx[5]].named_parameters(),layers[idx[6]].named_parameters(),layers[idx[7]].named_parameters(),layers_pre[idx[7]+1].named_parameters(), layers_pre[idx[4]].named_parameters(), layers_pre[idx[5]].named_parameters(), layers_pre[idx[6]].named_parameters(), layers_pre[idx[7]].named_parameters()):
    if 'input_layernorm' in name1 or  'post_attention_layernorm' in name1:
        continue
    if name1 == name2:
        new_param = param6 + 0.8*((param1-param6) + spars((param2-param7)) + spars((param3-param8))+ spars((param4-param9)) +spars((param5-param10)))
        with torch.no_grad():
            param1.copy_(new_param)

total_params = sum(p.numel() for p in model.parameters())

print(f"Model has {total_params} parameters.")

layers_middle_2 = layers[idx[7]+1]

layers_bottom = layers[:idx[0]-1]
layers_top = layers[idx[7]+2:]
student = nn.ModuleList()
for ans in layers_bottom:
    student.append(ans)
student.append(layers_middle_1)
student.append(layers_middle_2)
for ans in layers_top:
    student.append(ans)
model.model.layers = student

new_config = AutoConfig.from_pretrained(path_model)
new_config.num_hidden_layers = len(model.model.layers)

model.config = new_config

model.save_pretrained('layer_prune_llama_important', safe_serialization=False)
tokenizer.save_pretrained('layer_prune_llama_important')
