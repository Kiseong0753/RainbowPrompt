# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for dualprompt implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class PreT_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prompt, probability):
        if probability is None:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
            self.key_ = q
            if prompt is not None:
                prompt = prompt.permute(1, 0, 3, 2, 4).contiguous() # 2, B, num_heads, prompt_length, C // num_heads
                key_prefix = prompt[0] # B, num_heads, prompt_length, embed_dim // num_heads
                value_prefix = prompt[1] # B, num_heads, prompt_length, embed_dim // num_heads
                expected_shape = (B, self.num_heads, C // self.num_heads)
                assert (key_prefix.shape[0], key_prefix.shape[1], key_prefix.shape[3]) == expected_shape, f'key_prefix.shape: {key_prefix.shape} not match k.shape: {k.shape}'
                assert (value_prefix.shape[0], value_prefix.shape[1], value_prefix.shape[3]) == expected_shape, f'value_prefix.shape: {value_prefix.shape} not match v.shape: {v.shape}'
                k = torch.cat([key_prefix, k], dim=2)
                v = torch.cat([value_prefix, v], dim=2)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        else:
            self.alpha = probability[0]
            self.half_alpha = probability[1]
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)   

            self.key_ = q
            prompt = prompt.permute(1, 0, 3, 2, 4).contiguous()
            
            key_prefix = prompt[0] 
            value_prefix = prompt[1]
            expected_shape = (B, self.num_heads, C // self.num_heads)
            assert (key_prefix.shape[0], key_prefix.shape[1], key_prefix.shape[3]) == expected_shape, f'key_prefix.shape: {key_prefix.shape} not match k.shape: {k.shape}'
            assert (value_prefix.shape[0], value_prefix.shape[1], value_prefix.shape[3]) == expected_shape, f'value_prefix.shape: {value_prefix.shape} not match v.shape: {v.shape}'
            
            k = torch.cat([key_prefix, k * self.half_alpha], dim=2) * self.alpha
            v = torch.cat([value_prefix, v * self.half_alpha], dim=2) * self.alpha
            
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        return x

def train_sample_policy(self, index):
    parameter_list = self.soft_gate_dict[str(index)]  
    logits = torch.cat([param for param in parameter_list], dim=0) 
    policy = F.gumbel_softmax(logits, 2.0, hard=False)
    return policy.float()

def test_sample_policy(self, index):
    task_logits = self.soft_gate_dict[str(index)]  
    cuda_device = task_logits[0].get_device() if task_logits[0].is_cuda else -1
    logits = torch.cat([param for param in task_logits], dim=0).detach().cpu().numpy()  # Shape: (12, 2)
    distribution = softmax(logits, axis=-1)  # Shape: (12, 2)
    single_policys = []
    for tmp_d in distribution:  # tmp_d: probability distribution (2,)
        sampled = np.random.choice(2, p=tmp_d)  # Randomly sample index (0 or 1)
        policy = [0, 0]
        policy[sampled] = 1  # Set the sampled index to 1
        single_policys.append(policy)
    policy_tensor = torch.from_numpy(np.array(single_policys))  # Shape: (12, 2)
    if cuda_device != -1:
        policy_tensor = policy_tensor.to('cuda:%d' % cuda_device)
    selected_indices = torch.nonzero(policy_tensor[:, 0]).squeeze().tolist()
    if isinstance(selected_indices, int):  # If a single index is selected, convert to list
        selected_indices = [selected_indices]
    return selected_indices

def add_hardgate(self, final_decision):
    self.hard_gate.append(final_decision)
    logger.info(f'Hard gard list : {self.hard_gate}')
    
def add_self_attn_idx(self, current_task_id):
    current_hard_gate = self.hard_gate[current_task_id]
    length = len(current_hard_gate)
    mid_idx = length // 2 if length % 2 == 0 else (length // 2)
    indicies_for_self_attn = current_hard_gate[:mid_idx]
    self.self_attn_idx_adaptive.append(indicies_for_self_attn)
    logger.info(f'self attn idx list : {self.self_attn_idx_adaptive}')
    
def backward_policy(self, usage_index):
    parameter_list = self.soft_gate_dict[str(usage_index)]  
    current_task_logit = torch.cat([param for param in parameter_list], dim=0)  # Shape: (12, 2)
    logits_column_0 = current_task_logit[:, 0]  # Shape: (12,)
    gt = torch.zeros_like(logits_column_0).to(logits_column_0.device)  # Shape: (12,)
    criterion = nn.BCEWithLogitsLoss()
    sparsity_loss = criterion(logits_column_0, gt)  # Sparsity loss
    # L1 regularization
    l1_lambda = 0.0001 
    l1_regularization = l1_lambda * torch.norm(logits_column_0, p=1)
    total_loss = sparsity_loss + l1_regularization
    return total_loss.item()