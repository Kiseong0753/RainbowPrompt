import torch
import torch.nn as nn
import math
import copy
from loguru import logger
import torch.nn.functional as F
import numpy as np

class RainbowPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_e_prompt=False, num_heads=-1, same_key_value=False, cls_rank=None, 
                 prompt_rank=None, prompt_tune_idx=None, n_tasks=None,  D1=None,relation_type=None, use_linear=None,
                 KI_iter=None, self_attn_idx=None, D2=None):
        super().__init__()

        self.length = length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value
        self.embed_dim = embed_dim
        self.prompt_tune_idx = prompt_tune_idx
        self.n_tasks = n_tasks
        self.D1 = D1
        self.relation_type = relation_type
        self.use_linear = use_linear
        self.KI_iter = KI_iter
        self.self_attn_idx = self_attn_idx
        self.D2 = D2
        self.register_buffer(
            'stored_rainbow_prompts',
            torch.zeros(n_tasks, len(prompt_tune_idx), length, embed_dim)
        )
###########################################################################################################################   
        if self.use_linear:
            self.query_matcher = nn.ModuleList([nn.Linear(self.embed_dim, self.D2) for _ in range(len(self.prompt_tune_idx))])
            self.key_matcher = nn.ModuleList([nn.Linear(self.embed_dim, self.D2) for _ in range(len(self.prompt_tune_idx))])
            self.value_matcher = nn.ModuleList([nn.Linear(self.embed_dim, self.D2) for _ in range(len(self.prompt_tune_idx))])
            self.dense = nn.ModuleList([nn.Linear(self.D2, self.embed_dim) for _ in range(len(self.prompt_tune_idx))])

            self.fc1 = nn.ModuleList([
                nn.Linear(self.embed_dim, self.D1)
                for _ in range(len(self.prompt_tune_idx))
            ])

            self.fc2 = nn.ModuleList([
                nn.Linear(self.D1, self.embed_dim)
                for _ in range(len(self.prompt_tune_idx))
            ])
              
        for l in self.prompt_tune_idx:                
            base_knowledge = self.tensor_matrix(self.pool_size, self.length, self.embed_dim)
            setattr(self, f'base_knowledge_{l}', base_knowledge)
        base_key = self.tensor_matrix(self.n_tasks, self.embed_dim, None)
        setattr(self, f'base_key', base_key)


    def tensor_matrix(self, a, b, c):
        if c is None:
            p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
            nn.init.uniform_(p)
        else:
            p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
            nn.init.uniform_(p)
        return p
            
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def freeze_components(self, task_id):
        if task_id > 0:
            for layer in self.query_matcher:
                for param in layer.parameters():
                    param.requires_grad = False

            for layer in self.key_matcher:
                for param in layer.parameters():
                    param.requires_grad = False

            for layer in self.value_matcher:
                for param in layer.parameters():
                    param.requires_grad = False

            for layer in self.dense:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def task_conditioning_step(self, base_knowledge, current_task_embed):
        if self.relation_type == 'attention':
            key_expanded = current_task_embed.unsqueeze(0).unsqueeze(0) 
            key_expanded = key_expanded.expand(base_knowledge.size(0), base_knowledge.size(1), -1)  
            relevance_scores = torch.matmul(base_knowledge, key_expanded.transpose(-1, -2)) 
            relevance_scores = F.softmax(relevance_scores, dim=-1)  
            conditioned_base_knowledge = torch.matmul(relevance_scores, base_knowledge)  
        else:
            relevance_scores = torch.einsum('nld,d->nl', base_knowledge, current_task_embed)
            relevance_scores = F.sigmoid(relevance_scores)
            conditioned_base_knowledge = torch.einsum('nl,nld->nld', relevance_scores, base_knowledge)
            
        return conditioned_base_knowledge  
    
    def Prompt_Evolution(self, layer, attended_prev, attended_curr, d_model, d_ff, dropout=0.1):
        def Attention_based_Transformation(q, k, v, d_model):
            if self.use_linear:
                q = self.query_matcher[layer](q)  
                k = self.key_matcher[layer](k)  
                v = self.value_matcher[layer](v)  

                scaled_attention_logits = torch.matmul(q, k.transpose(1,2)) / torch.sqrt(torch.tensor(q.shape[-1] , dtype=torch.float32).to(q.device)) 
                attention_weights = F.softmax(scaled_attention_logits, dim=-1)  
                output = torch.matmul(attention_weights, v) 
                
                q_transpose = q.transpose(1,2)
                k_transpose = k.transpose(1,2)
                transpose_logits = torch.matmul(q_transpose, k_transpose.transpose(1,2)) / torch.sqrt(torch.tensor(q_transpose.shape[-1] , dtype=torch.float32).to(q.device)) 
                transpose_weights = F.softmax(transpose_logits, dim=-1) 
                output = torch.matmul(transpose_weights, output.transpose(1,2)).transpose(1,2)  
                output = self.dense[layer](output) 
            else:
                scaled_attention_logits = torch.matmul(q, k.transpose(1,2)) / torch.sqrt(torch.tensor(q.shape[-1] , dtype=torch.float32).to(q.device))
                attention_weights = F.softmax(scaled_attention_logits, dim=-1)
                output = torch.matmul(attention_weights, v) 
            return output

        def Task_guided_Alignment(l_index, x, d_model, d_ff):
            x = F.relu(self.fc1[layer](x))
            x = self.fc2[layer](x)
            return x

        def Evolving(l_index, prev, curr, d_model, d_ff, dropout):
            if self.use_linear:
                attn_output = Attention_based_Transformation(curr, prev, prev, d_model)
                attn_output = F.dropout(attn_output, dropout, training=True)
                out1 = F.layer_norm(prev + attn_output, [d_model])  

                ffn_output = Task_guided_Alignment(l_index, out1, d_model, d_ff)
                ffn_output = F.dropout(ffn_output, dropout, training=True)
                out2 = F.layer_norm(out1 + ffn_output, [d_model])  
                return out2
            else:
                attn_output = Attention_based_Transformation(curr, prev, prev, d_model)
                attn_output = F.dropout(attn_output, dropout, training=True)
                out1 = F.layer_norm(prev + attn_output, [d_model]) 
                return out1
        
        task_wise_evolved_results = []
        if layer not in self.self_attn_idx:
            for KI_layer in range(self.task_id+1):
                if KI_layer == self.task_id:
                    attended_p = attended_curr
                    attended_c = attended_curr
                else:
                    attended_p = attended_prev[KI_layer*self.top_k:KI_layer*self.top_k+self.top_k]
                    attended_c = attended_curr
                Evolved_knowledge = Evolving(self.task_id, attended_p, attended_c, d_model, d_ff, dropout)
                task_wise_evolved_results.append(Evolved_knowledge)
            final_representation = torch.cat(task_wise_evolved_results, dim=0)
        else:
            attended_p, attended_c = attended_curr, attended_curr
            for iteration in range(self.KI_iter):
                Evolved_knowledge = Evolving(self.task_id, attended_p, attended_c, d_model, d_ff, dropout)
                attended_p = Evolved_knowledge
            final_representation = Evolved_knowledge
            
            
        return final_representation
    
    
    
    def forward(self, x_embed, layer, previous_mask=None, cls_features=None, task_id=None, cur_id=None, train=False, p_type=None):
        if p_type == 'Rainbow':
            out = dict()
            self.task_id = None
            base_knowledge = getattr(self, f'base_knowledge_{layer}')
            base_key = getattr(self, f'base_key')

            if train:
                self.task_id = cur_id
                if self.task_id == 0:
                    prev_base_knowledge = base_knowledge[self.task_id:self.top_k]
                    curr_base_knowledge = base_knowledge[self.task_id:self.top_k]
                    base_key_set = base_key[self.task_id]
                else:
                    prev_base_knowledge = base_knowledge[0:self.task_id*self.top_k].detach().clone()
                    curr_base_knowledge = base_knowledge[self.task_id*self.top_k:self.task_id*self.top_k+self.top_k]
                    base_key_set = base_key[self.task_id]

                key_norm = self.l2_normalize(base_key_set, dim=-1) 
                embed_norm = self.l2_normalize(cls_features, dim=-1)

                similarity = torch.matmul(key_norm, embed_norm.t()) 
                similarity = torch.sum(similarity) / embed_norm.shape[0]
                out['sim_loss'] = similarity
                attended_prev_base = self.task_conditioning_step(prev_base_knowledge, key_norm) 
                attended_curr_base = self.task_conditioning_step(curr_base_knowledge, key_norm) 
                Evolved_knowledge_set = self.Prompt_Evolution(layer, attended_prev_base, attended_curr_base, self.embed_dim, self.D1)

                RainbowPrompt = torch.mean(Evolved_knowledge_set, dim=0)
                with torch.no_grad():
                    self.stored_rainbow_prompts[self.task_id, layer].copy_(RainbowPrompt)
                RainbowPrompt = RainbowPrompt.expand(embed_norm.shape[0], -1, -1) 
                key_prompt = RainbowPrompt[:, :int(self.length/2),:]
                value_prompt = RainbowPrompt[:,int(self.length/2):,:]
                out['batched_prompt'] = [key_prompt, value_prompt]

            else:
                embed_norm = self.l2_normalize(cls_features, dim=-1)
                matching_result = []
                for certain_task in range(cur_id+1):
                    certain_task_key = base_key[certain_task]
                    certain_task_key = self.l2_normalize(certain_task_key, dim=-1)
                    sim_score = torch.matmul(certain_task_key, embed_norm.t())
                    sim_score = torch.sum(sim_score) / embed_norm.shape[0]
                    matching_result.append(sim_score)

                matching_result_tensor = torch.stack(matching_result)
                max_index = torch.argmax(matching_result_tensor)

                self.task_id = int(max_index)

                stored = self.stored_rainbow_prompts[self.task_id, layer]  
                RainbowPrompt = stored.expand(embed_norm.shape[0], -1, -1) 

                k_p = RainbowPrompt[:, :int(self.length/2),:]
                v_p = RainbowPrompt[:,int(self.length/2):,:]
                out['batched_prompt'] = [k_p, v_p]

            return out
        
        else:
            out = dict()
            base_knowledge = getattr(self, f'base_knowledge_{layer}')
            base_key = getattr(self, f'base_key')
            self.task_id = cur_id
            
            if self.task_id == 0:
                curr_base_knowledge = base_knowledge[self.task_id:self.top_k]
                base_key_set = base_key[self.task_id]
                
            else:
                curr_base_knowledge = base_knowledge[self.task_id*self.top_k:self.task_id*self.top_k+self.top_k]
                base_key_set = base_key[self.task_id]
                
            key_norm = self.l2_normalize(base_key_set, dim=-1) 
            embed_norm = self.l2_normalize(cls_features, dim=-1)
            similarity = torch.matmul(key_norm, embed_norm.t()) 
            similarity = torch.sum(similarity) / embed_norm.shape[0]
            
            out['sim_loss'] = similarity
            
            RainbowPrompt = torch.mean(curr_base_knowledge, dim=0)
            RainbowPrompt = RainbowPrompt.expand(embed_norm.shape[0], -1, -1) 
            key_prompt = RainbowPrompt[:, :int(self.length/2),:]
            value_prompt = RainbowPrompt[:,int(self.length/2):,:]
            out['batched_prompt'] = [key_prompt, value_prompt]

            return out
