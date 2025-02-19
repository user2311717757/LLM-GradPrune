import torch
from safetensors.torch import safe_open
import pdb

optim = torch.load("/mnt2/name_prune/saves/llama3-1-8b-instruct/lora/winogrande_acl_swift_test/v11-20250214-163153/checkpoint-40/optimizer.pt", map_location=torch.device('cpu'))

optimizers = optim['state']

lora_a = torch.zeros(24).cuda()
lora_b = torch.zeros(24).cuda()

for i in range(0,len(optim['state']),14):
    for j in range(i,i+14):
        exp_avg = optim['state'][j]['exp_avg']
        ten = exp_avg.abs().sum()
        if j%2 == 0:
            lora_a[int(i/14)] += ten
        else:
            lora_b[int(i/14)] += ten

pdb.set_trace()

print(lora_a,lora_b,(lora_a+lora_b)/2)
