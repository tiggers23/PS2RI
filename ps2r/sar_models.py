from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import pickle
import sys
from global_config import *
import math
import torch
from torch.nn.utils.rnn import pad_sequence
import copy

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn import init
from transformers import (
    AlbertModel,
    AlbertPreTrainedModel,
    AlbertConfig,
    load_tf_weights_in_albert,
)
from transformers.models.albert.modeling_albert import AlbertEmbeddings, AlbertLayerGroup


"""
Implementation taken from:
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""


class Transformer(nn.Module):
    def __init__(self, d_model, num_layers=1, nhead=1, dropout=0.1, dim_feedforward=128, max_seq_length=5000):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.pos_encoder = nn.Embedding(max_seq_length, d_model)
        self.encoder = TransformerEncoder(TransformerLayer(d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout), num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input, attention_mask=None):
        seq_length = input.size()[1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input.device)
        positions_embedding = self.pos_encoder(position_ids).unsqueeze(0).expand(input.size()) # (seq_length, d_model) => (batch_size, seq_length, d_model)
        input = input + positions_embedding
        input = self.norm(input)
        hidden = self.encoder(input, attention_mask=attention_mask)
        out = self.decoder(hidden) # (batch_size, seq_len, hidden_dim)
        out = (out[:,0,:], out, hidden) # ([CLS] token embedding, full output, last hidden layer)
        return out



class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, nhead=1, dim_feedforward=128, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.self_attention = Attention(hidden_size, nhead, dropout)
        self.fc = nn.Sequential(nn.Linear(hidden_size, dim_feedforward), nn.ReLU(), nn.Linear(dim_feedforward, hidden_size))
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, attention_mask=None):
        src_1 = self.self_attention(src, src, attention_mask=attention_mask)
        src = src + self.dropout1(src_1)
        src = self.norm1(src)
        src_2 = self.fc(src)
        src = src + self.dropout2(src_2)
        src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(layer, num_layers)
    def forward(self, src, attention_mask=None):
        for layer in self.layers:
            new_src = layer(src, attention_mask=attention_mask)
            src = src + new_src
        return src

class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, ctx_dim=None):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim = hidden_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the attention mask is
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class AttentionOutput(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(AttentionOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class Gate(nn.Module):
    def __init__(self, in_sz, out_sz):
        super(Gate, self).__init__()

        self.W_1 = nn.Parameter(torch.Tensor(in_sz, out_sz))
        self.W_2 = nn.Parameter(torch.Tensor(in_sz, out_sz))
        self.b = nn.Parameter(torch.Tensor(out_sz))
        self.init_weights()

    def forward(self, hidden_states1, hidden_states2):
        G = torch.sigmoid(hidden_states1 @ self.W_1 + hidden_states2 @ self.W_2 + self.b)
        Z = G * hidden_states1 + (1-G) * hidden_states2
        return Z

    def init_weights(self):
        init.kaiming_uniform_(self.W_1, a=math.sqrt(5))
        init.kaiming_uniform_(self.W_2, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.W_1)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.b, -bound, bound)

class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size, cross_size1, cross_size2, nhead=1, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.cross_attention_1 = Attention(hidden_size, nhead, dropout, ctx_dim=cross_size1)
        self.cross_attention_2 = Attention(hidden_size, nhead, dropout, ctx_dim=cross_size2)
        self.self_attention = Attention(hidden_size, nhead, dropout)

        self.out1 = AttentionOutput(hidden_size, dropout)
        self.out2 = AttentionOutput(hidden_size, dropout)
        self.out3 = AttentionOutput(hidden_size, dropout)
        self.out4 = AttentionOutput(hidden_size, dropout)

        self.gate = Gate(hidden_size, hidden_size)

    def forward(self, hidden_states, cross_states_1, cross_states_2, attention_mask=None):
        self_out = self.self_attention(hidden_states, hidden_states, attention_mask=attention_mask)
        self_out = self.out3(self_out, hidden_states)

        cross_1 = self.cross_attention_1(self_out, cross_states_1, attention_mask=attention_mask)
        cross_1 = self.out1(cross_1, self_out)

        cross_2 = self.cross_attention_2(self_out, cross_states_2, attention_mask=attention_mask)
        cross_2 = self.out2(cross_2, self_out)

        cross_fusion = self.gate(cross_1, cross_2)
        # cross_fusion = self.out4(cross_fusion, cross_fusion)

        return cross_fusion

class FusionEncoder(nn.Module):
    def __init__(self, layer, num_layers, fusion_dim, nhead=None, dropout=None, modal=False):
        super(FusionEncoder, self).__init__()
        self.num_layers = num_layers
        self.modal = modal
        self.fusion1 = _get_clones(layer, num_layers)
        self.fusion2 = _get_clones(layer, num_layers)
        self.fusion3 = _get_clones(layer, num_layers)
        if modal:
            self.U1 = nn.Parameter(torch.Tensor(fusion_dim, fusion_dim))
            self.U2 = nn.Parameter(torch.Tensor(fusion_dim, fusion_dim))
            self.U3 = nn.Parameter(torch.Tensor(fusion_dim, fusion_dim))

            self.linear1 = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.Tanh()
            )
            self.linear2 = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.Tanh()
            )
            self.linear3 = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.Tanh()
            )

            self.softmax = nn.Softmax(dim=0)
            self.self_attention = Attention(fusion_dim, nhead, dropout)
            self.out = AttentionOutput(fusion_dim, dropout)
            self.init_weights()

    def init_weights(self):
        init.kaiming_uniform_(self.U2, a=math.sqrt(5))
        init.kaiming_uniform_(self.U1, a=math.sqrt(5))
        init.kaiming_uniform_(self.U3, a=math.sqrt(5))

    def forward(self, modal_states1, modal_states2, modal_states3, attention_mask=None):
        hidden_states1 = modal_states1
        hidden_states2 = modal_states2
        hidden_states3 = modal_states3

        for i in range(self.num_layers):
            layer1 = self.fusion1[i]
            layer2 = self.fusion2[i]
            layer3 = self.fusion3[i]

            out_1 = layer1(hidden_states1, hidden_states2, hidden_states3, attention_mask=attention_mask)
            out_2 = layer2(hidden_states2, hidden_states1, hidden_states3)
            out_3 = layer3(hidden_states3, hidden_states1, hidden_states2)

            hidden_states1 = out_1
            hidden_states2 = out_2
            hidden_states3 = out_3

        if self.modal:
            u1 = torch.matmul(self.linear1(hidden_states1), self.U1)
            u2 = torch.matmul(self.linear2(hidden_states2), self.U2)
            u3 = torch.matmul(self.linear3(hidden_states3), self.U3)

            U = torch.cat([u1.unsqueeze(0), u2.unsqueeze(0), u3.unsqueeze(0)], dim=0)
            alpha = self.softmax(U)
            m = torch.cat([hidden_states1.unsqueeze(0), hidden_states2.unsqueeze(0), hidden_states3.unsqueeze(0)], dim=0)
            z = torch.sum(alpha * m, dim=0).squeeze(0)
            hidden_states = self.self_attention(z, z, attention_mask=attention_mask)
            self_out = self.out(hidden_states, z)

            return self_out


        return hidden_states1, hidden_states2, hidden_states3

class task_linear(nn.Module):
    def __init__(self, fusion_dim, cls_num, dropout):
        super(task_linear, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, cls_num)
        )

    def forward(self, hidden_states):
        return self.linear(hidden_states)


class Sarcasm(nn.Module):
    def __init__(self, text_model, visual_model, acoustic_model, args, dropout=0.1, fusion_dim=128):
        super(Sarcasm, self).__init__()

        self.newly_added_config = args
        self.text_model = text_model
        self.tf = nn.Linear(args.text_dim, fusion_dim)
        self.visual_model = visual_model
        self.vf = nn.Linear(visual_model.d_model, fusion_dim)
        self.acoustic_model = acoustic_model
        self.af = nn.Linear(acoustic_model.d_model, fusion_dim)

        fusion_layer = CrossAttentionLayer(fusion_dim, fusion_dim, fusion_dim, nhead=args.sar_cross_n_heads, dropout=args.sar_dropout)
        self.modal_fusion = FusionEncoder(fusion_layer, args.sar_n_layers, fusion_dim, nhead=args.sar_cross_n_heads, dropout=args.sar_dropout, modal=True)

        self.se = nn.Linear(fusion_dim, 2)


    def get_params(self):
        acoustic_params = list(self.acoustic_model.named_parameters())
        visual_params = list(self.visual_model.named_parameters())

        other_params = list(self.text_model.named_parameters()) + list(self.modal_fusion.named_parameters()) \
                       + list(self.tf.named_parameters()) + list(self.vf.named_parameters()) \
                       + list(self.af.named_parameters()) + list(self.se.named_parameters())


        return acoustic_params, visual_params, other_params

    def forward(self, input_ids, visual, acoustic, attention_mask=None, token_type_ids=None):
        text_output = self.tf(self.text_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0])
        acoustic_output = self.af(self.acoustic_model(acoustic)[2])
        visual_output = self.vf(self.visual_model(visual)[2])

        # attention mask conversion
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        modal_fusion = self.modal_fusion(text_output, visual_output, acoustic_output, attention_mask=extended_attention_mask)

        # for i in range(len(modal_fusion.shape[0])):
        #     features = modal_fusion[i,:].unsqueeze().detach().cpu().numpy()
        #     np.save(f'./output/sar_features/{id[i]}.npy',features)
        sentiment_embedding = F.max_pool1d(modal_fusion.permute(0,2,1).contiguous(), modal_fusion.shape[1]).squeeze(-1)

        se_out = self.se(sentiment_embedding)

        # return modal_fusion
        return modal_fusion, se_out

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])