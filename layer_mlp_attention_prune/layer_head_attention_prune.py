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
import pickle
import pandas as pd
from tqdm import tqdm
from torch.nn.functional import cosine_similarity

def remove_numbers(tensor):
    arr = tensor.cpu().numpy()
    result = []
    i = 0
    while i < len(arr):
        if i < len(arr) - 1 and arr[i] + 1 == arr[i + 1]:
            result.append(arr[i])
            i += 2
        else:
            result.append(arr[i])
            i += 1
    return torch.tensor(result)

def spars(tensor):

    k = int(tensor.numel() * 0.2)

    indices = torch.argsort(tensor.abs().view(-1), descending=True)[:k]

    mask = torch.zeros_like(tensor)
    mask.view(-1)[indices] = 1
    tensor.mul_(mask)
    return tensor

features_mlp_out = []
features_in = []
def hook(module, input, output): 
    features_in.append(input)

def hook_mlp(module, input, output): 
    features_mlp_out.append(output)


path_model = "/mnt2/name_prune/layer_prune/layer_prune_llama_important"

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

head_dim = int(config.hidden_size / config.num_attention_heads)
num_attention_heads = config.num_attention_heads
num_key_value_heads = config.num_key_value_heads
feature_gate = torch.zeros(num_attention_heads).cuda() 
feature_mlp_gate = torch.zeros(model.model.layers[0].mlp.gate_proj.weight.size(0)).cuda() 
layers_important = torch.zeros(len(model.model.layers))

data_path = "/mnt2/name_prune/data/pubmed_train_acl.json"
with open(data_path,'r',encoding='utf-8') as f:
    data = json.load(f)
data = data[:512]

for ans in tqdm(data):
    q = ans['conversations'][0]['value']
    a = ans['conversations'][1]['value']
    text = template.format(q,a)
    inputs = tokenizer(text, return_tensors="pt")
    inputs['input_ids'] = inputs['input_ids'].cuda()
    inputs['attention_mask'] = inputs['attention_mask'].cuda()

    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks1 = layer.self_attn.o_proj.register_forward_hook(hook)
        hooks2 = layer.mlp.gate_proj.register_forward_hook(hook_mlp)
        hooks.append(hooks1)
        hooks.append(hooks2)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states

    for i, layer_output in enumerate(hidden_states):
        if i >= len(model.model.layers):
            break
        A = hidden_states[i]
        B = hidden_states[i+1]
        similarities = cosine_similarity(A.squeeze(), B.squeeze(), dim=1)
        score = similarities.mean()
        layers_important[i] = layers_important[i] + score

    for i in range(0,len(features_in)):
        for j in range(num_attention_heads):
            features = torch.norm(features_in[i][0].squeeze()[:,j*head_dim:(j+1)*head_dim], dim=1, keepdim=True)
            feature_gate[j] = feature_gate[j] + torch.mean(features.abs(),dim=0)

    for i in range(0,len(features_mlp_out)):
        feature_mlp_gate += torch.mean(features_mlp_out[i].squeeze().abs(),dim=0)

    for h in hooks:
        h.remove()
    features_in.clear()
    features_mlp_out.clear()

bi = 1 - layers_important/len(data)
bi = 1 / (bi / sum(bi))
bi = bi / sum(bi)

feature_gate = feature_gate/len(data)/24
indices_gate = torch.argsort(feature_gate.abs().view(-1), descending=True)

feature_mlp_gate = feature_mlp_gate/len(data)/24
indices_mlp_gate = torch.argsort(feature_mlp_gate.abs().view(-1), descending=True)
pdb.set_trace()

indices_gate_kv = [i.item() for i in indices_gate if i.item() < num_key_value_heads]

total_params = sum(p.numel() for p in model.parameters())
print(f"Model has {total_params} parameters.")

layers = model.model.layers

layers_pre = model_pre.model.layers

l = model.model.layers[0].mlp.gate_proj.weight.size(0)

i = 0
save = []
save_mlp = []
s = num_key_value_heads / (num_key_value_heads*2 + num_attention_heads*2)
for layer,layer_pre in tqdm(zip(layers,layers_pre)):        
    if i >=18 and i <= 22:
        indices_prune_qo,_= torch.tensor(indices_gate[24:]).sort() 
        indices_prune_kv = torch.tensor([4,  5]) 


        q_proj_data = layer.self_attn.q_proj.weight.data
        q_proj_data_pre = layer_pre.self_attn.q_proj.weight.data
        q_proj_data_task_vector = q_proj_data - q_proj_data_pre

        o_proj_data = layer.self_attn.o_proj.weight.data
        o_proj_data_pre = layer_pre.self_attn.o_proj.weight.data
        o_proj_data_task_vector = o_proj_data - o_proj_data_pre

        k = 0
        while k < 7:
            base = indices_prune_qo[k]

            if indices_prune_qo[k] == indices_prune_qo[k+1]-1:
                q_proj_data[(base-1)*head_dim:base*head_dim] = q_proj_data_pre[(base-1)
                *head_dim:base*head_dim] + 0.8*(0.8*q_proj_data_task_vector[(base-1)*head_dim:base*head_dim] + 0.2*spars(q_proj_data_task_vector[base*head_dim:(base+1)*head_dim]))
                q_proj_data[(base+2)*head_dim:(base+3)*head_dim] = q_proj_data_pre[(base+2)
                *head_dim:(base+3)*head_dim] + 0.8*(0.8*q_proj_data_task_vector[(base+2)*head_dim:(base+3)*head_dim] + 0.2*spars(q_proj_data_task_vector[(base+1)*head_dim:(base+2)*head_dim]))

                o_proj_data[:,(base-1)*head_dim:base*head_dim] = o_proj_data_pre[:,(base-1)
                *head_dim:base*head_dim] + 0.8*(0.8*o_proj_data_task_vector[:,(base-1)*head_dim:base*head_dim] + 0.2*spars(o_proj_data_task_vector[:,base*head_dim:(base+1)*head_dim]))
                o_proj_data[:,(base+2)*head_dim:(base+3)*head_dim] = o_proj_data_pre[:,(base+2)
                *head_dim:(base+3)*head_dim] + 0.8*(0.8*o_proj_data_task_vector[:,(base+2)*head_dim:(base+3)*head_dim] + 0.2*spars(o_proj_data_task_vector[:,(base+1)*head_dim:(base+2)*head_dim]))
                k = k+2
            else: 
                q_proj_data[(base-1)*head_dim:base*head_dim] = q_proj_data_pre[(base-1)
                *head_dim:base*head_dim] + 0.8*(0.8*q_proj_data_task_vector[(base-1)*head_dim:base*head_dim] + 0.2*spars(q_proj_data_task_vector[base*head_dim:(base+1)*head_dim]))

                o_proj_data[:,(base-1)*head_dim:base*head_dim] = o_proj_data_pre[:,(base-1)*head_dim:base*head_dim] + 0.8*(0.8*o_proj_data_task_vector[:,(base-1)*head_dim:base*head_dim] + 0.2*spars(o_proj_data_task_vector[:,base*head_dim:(base+1)*head_dim]))
                k = k + 1
        if k == 7:
            base = indices_prune_qo[7]
            q_proj_data[(base-1)*head_dim:base*head_dim] = q_proj_data_pre[(base-1)
            *head_dim:base*head_dim] + 0.8*(0.8*q_proj_data_task_vector[(base-1)*head_dim:base*head_dim] + 0.2*spars(q_proj_data_task_vector[base*head_dim:(base+1)*head_dim]))

            o_proj_data[:,(base-1)*head_dim:base*head_dim] = o_proj_data_pre[:,(base-1)*head_dim:base*head_dim] + 0.8*(0.8*o_proj_data_task_vector[:,(base-1)*head_dim:base*head_dim] + 0.2*spars(o_proj_data_task_vector[:,base*head_dim:(base+1)*head_dim]))

        k_proj_data = layer.self_attn.k_proj.weight.data
        k_proj_data_pre = layer_pre.self_attn.k_proj.weight.data
        k_proj_data_task_vector = k_proj_data - k_proj_data_pre

        v_proj_data = layer.self_attn.v_proj.weight.data
        v_proj_data_pre = layer_pre.self_attn.v_proj.weight.data
        v_proj_data_task_vector = v_proj_data - v_proj_data_pre

        base = indices_prune_kv[0]
        k_proj_data[(base-1)*head_dim:base*head_dim] = k_proj_data_pre[(base-1)
        *head_dim:base*head_dim] + 0.8*(0.8*k_proj_data_task_vector[(base-1)*head_dim:base*head_dim] + spars(k_proj_data_task_vector[base*head_dim:(base+1)*head_dim]) + 0.2*spars(k_proj_data_task_vector[(base+1)*head_dim:(base+2)*head_dim]))
        v_proj_data[(base-1)*head_dim:base*head_dim] = v_proj_data_pre[(base-1)
        *head_dim:base*head_dim] + 0.8*(0.8*v_proj_data_task_vector[(base-1)*head_dim:base*head_dim] + spars(v_proj_data_task_vector[base*head_dim:(base+1)*head_dim]) + 0.2*spars(v_proj_data_task_vector[(base+1)*head_dim:(base+2)*head_dim]))

        layer.self_attn.q_proj.weight.data = q_proj_data
        layer.self_attn.k_proj.weight.data = k_proj_data
        layer.self_attn.v_proj.weight.data = v_proj_data
        layer.self_attn.o_proj.weight.data = o_proj_data

        save.append((24,6))

        mask = torch.isin(indices_gate, indices_prune_qo.cuda())
        indices_qo = indices_gate[~mask]
        indices_kv = torch.tensor(indices_gate_kv[:6])

        #################prune mlp#############

        length = int(14336*0.30)
        indices_mlp_remian,_ = indices_mlp_gate[:l-length].sort()
        indices_mlp_prune,_ = indices_mlp_gate[l-length:].sort()
        indices_mlp_prune = indices_mlp_prune[1:]
        indices_mlp_prune = remove_numbers(indices_mlp_prune).cuda()
        save_mlp.append(l-length)

        gate_proj_data = layer.mlp.gate_proj.weight.data
        up_proj_data = layer.mlp.up_proj.weight.data
        down_proj_data = layer.mlp.down_proj.weight.data

        gate_proj_data_pre = layer_pre.mlp.gate_proj.weight.data
        up_proj_data_pre = layer_pre.mlp.up_proj.weight.data
        down_proj_data_pre = layer_pre.mlp.down_proj.weight.data

        gate_proj_data_task_vector = gate_proj_data - gate_proj_data_pre
        up_proj_data_task_vector = up_proj_data - up_proj_data_pre
        down_proj_data_task_vector = down_proj_data - down_proj_data_pre

        for base in indices_mlp_prune:
            base = base.item()

            gate_proj_data[base-1:base] = gate_proj_data_pre[base-1:base] + 0.8*(0.8*gate_proj_data_task_vector[base-1:base] + 0.2*spars(gate_proj_data_task_vector[base:base+1]))

            up_proj_data[base-1:base] = up_proj_data_pre[base-1:base] + 0.8*(0.8*up_proj_data_task_vector[base-1:base] + spars(0.2*up_proj_data_task_vector[base:base+1]))

            down_proj_data[:,base-1:base] = down_proj_data_pre[:,base-1:base] + 0.8*(0.8*down_proj_data_task_vector[:,base-1:base] + spars(0.2*down_proj_data_task_vector[:,base:base+1]))

    else:
        save.append((32,8))
        indices_qo = indices_gate[:32]
        indices_kv = torch.tensor(indices_gate_kv[:8])

        indices_mlp_remian,_ = indices_mlp_gate.sort()
        save_mlp.append(l)

    indices_qo,_ = indices_qo.sort()
    indices_kv,_ = indices_kv.sort()

    parts = []
    for base in indices_qo:
        base = base.item()
        part = torch.arange(base*head_dim, (base+1)*head_dim)  # 生成从 base 到 base + 127 的范围
        parts.append(part)
    indices_qo = torch.cat(parts)
    indices_qo = indices_qo.cuda()

    parts = []
    for base in indices_kv:
        base = base.item()
        part = torch.arange(base*head_dim, (base+1)*head_dim)  # 生成从 base 到 base + 127 的范围
        parts.append(part)
    indices_kv = torch.cat(parts)
    indices_kv = indices_kv.cuda()

    layer.self_attn.q_proj.weight.data = torch.index_select(layer.self_attn.q_proj.weight.data,0, indices_qo)
    layer.self_attn.k_proj.weight.data = torch.index_select(layer.self_attn.k_proj.weight.data,0, indices_kv)
    layer.self_attn.v_proj.weight.data = torch.index_select(layer.self_attn.v_proj.weight.data,0, indices_kv)
    layer.self_attn.o_proj.weight.data = torch.index_select(layer.self_attn.o_proj.weight.data,1, indices_qo)

    layer.mlp.gate_proj.weight.data = torch.index_select(layer.mlp.gate_proj.weight.data,0, indices_mlp_remian)
    layer.mlp.up_proj.weight.data = torch.index_select(layer.mlp.up_proj.weight.data,0, indices_mlp_remian)
    layer.mlp.down_proj.weight.data = torch.index_select(layer.mlp.down_proj.weight.data,1, indices_mlp_remian)
    i = i + 1

with open('model/llama_pubmed_attn.pkl', 'wb') as f:
    pickle.dump(save, f)

with open('model/llama_pubmed_mlp.pkl', 'wb') as f:
    pickle.dump(save_mlp, f)
pdb.set_trace()

total_params = sum(p.numel() for p in model.parameters())

print(f"Model has {total_params} parameters.")

model.save_pretrained('model/llama_pubmed_prune', safe_serialization=True)
tokenizer.save_pretrained('model/llama_pubmed_prune')
