# LLM-SPISM: Structural Pruning of Large Language Models via Importance-aware Sparse Merging
This repo contains the code for our ACL 2025 submitted paper.
## Requirements
**•** ms-swift == 3.0.0

**•** torch == 2.4.0

**•** transformers == 4.45.2

**•** vllm == 0.6.3.post1

**•** peft == 0.12.0

**•** python == 3.11.9

## Layer Important Pipeline
### Step1: Lora Training
The experiments showed that a significant drop in loss occurs during the initial phase of model training (approximately within the first 40 steps). This phase represents the period during which the model rapidly learns downstream tasks. Accompanying this phenomenon, layers of the model exhibit varying degrees of learning capability. Gradient updates, as a method for adjusting model parameters, effectively reflect the extent to which the model adapts to and responds to downstream tasks. Consequently, the gradient updates during the early stages of training were utilized to assess the importance of different layers within the model.

```
run_lora_llama3_1_layer_important.sh
```
### Step2: Obtaining layer importance scores
```
layer_important.py
```
## Layer Prune Pipeline
Prune the model layers based on their layer importance scores. The pruned layers are not directly discarded but are instead added to the remaining layers using Sparse Merged.
```
layer_prune/layer_important_prune.py #Add the paths for path_model and pre_model in the file.
```

## MLP and Attention Prune Pipeline
We evaluate the importance of MLP and Attention heads and assign different pruning ratios based on the layer importance.
### Step1: Run Layer Important Pipeline
### Step2: MLP and Attention Prune
```
layer_mlp_attention_prune/layer_head_attention_prune.py #The path_model in the file refers to the model after layer pruning.
```
## Training model
Use the .py files in the evaluation folder to conduct configuration/configuration_layer_prune/configuration_layer_mlp_attention_prune.
## Evaluation
Use the .py files in the evaluation folder to conduct testing.
## Things to note
*If you are training using a non-pruned model or a model with only layer pruning, please use the original modeling_llama.py file provided in the transformers library. However, if you are training a model that incorporates both MLP and attention pruning, replace the original modeling_llama.py file with the modified version we provide. The modifications are located at lines 289 and 359.*
