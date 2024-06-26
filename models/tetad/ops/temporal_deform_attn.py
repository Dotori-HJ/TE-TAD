# ------------------------------------------------------------------------
# Modified from TadTR (Modified from TadTR (https://github.com/xlliu7/TadTR)
# Copyright (c) 2021. Xiaolong Liu.
# ------------------------------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_



def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class DeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=1, n_heads=8, n_points=4):
        """
        Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                'd_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in DeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.seq2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(
            d_model, n_heads * n_levels * n_points)
        self.attention_weights = nn.Linear(
            d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        grid_init = torch.linspace(-self.n_points, self.n_points, self.n_heads * self.n_points + 1)
        grid_init = torch.cat([
            grid_init[:self.n_heads * self.n_points // 2],
            grid_init[self.n_heads * self.n_points // 2 + 1:]
        ], dim=0)[:, None]
        grid_init = grid_init.view(self.n_heads, 1, self.n_points, 1).repeat(1, self.n_levels, 1, 1)

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_temporal_lens, input_level_start_index, input_padding_mask=None):
        """
        :param query (= src + pos)         (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 1), range in [0, 1], left (0), right (1), including padding area
                                        or (N, Length_{query}, n_levels, 2), add additional (t) to form reference segments
        :param input_flatten (=src)        (N, \sum_{l=0}^{L-1} T_l, C)
        :param input_temporal_lens         (n_levels), [T_0, T_1, ..., T_(L-1)]
        :param input_level_start_index     (n_levels, ), [0, T_0, T_1, T_2, ..., T_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} T_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert input_temporal_lens.sum() == Len_in

        # Len_values = sum([input_temporal_lens[i].item() for i in self.n_levels])
        Len_values = input_temporal_lens[:self.n_levels].sum().item()
        value = self.value_proj(input_flatten[:, :Len_values])
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[:, :Len_values, None], float(0))
        value = value.view(N, Len_values, self.n_heads,
                           self.d_model // self.n_heads)
        # the predicted offset in temporal axis. They are *absolute* values, not normalized
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 1)
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(
            attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        if reference_points.shape[-1] == 1:
            # the reference points are normalized, but the offset are unnormalized
            # so we need to normalize the offsets
            offset_normalizer = input_temporal_lens[..., None]
            # (N, Length_{query}, n_heads, n_levels, n_points, 1)
            sampling_locations = reference_points[:, :, None, :self.n_levels, None, :] \
                + sampling_offsets / \
                offset_normalizer[None, None, None, :self.n_levels, None, :]
        # deform attention in the l-th (l >= 2) decoder layer when segment refinement is enabled
        elif reference_points.shape[-1] == 2:
            # offsets are related with the size of the reference segment
            sampling_locations = reference_points[:, :, None, :self.n_levels, None, :1] \
                + sampling_offsets / (self.n_points * self.n_heads // 2) * \
                reference_points[:, :, None, :self.n_levels, None, 1:]# * 0.5

        sampling_locations = torch.cat((sampling_locations, torch.ones_like(sampling_locations)*0.5), dim=-1)
        input_temporal_lens = input_temporal_lens[:self.n_levels]
        output = deform_attn_core_pytorch(value, input_temporal_lens, sampling_locations, attention_weights)

        output = self.output_proj(output)
        return output, (sampling_locations, attention_weights)


def deform_attn_core_pytorch(value, temporal_lens, sampling_locations, attention_weights):
    '''deformable attention implemeted with grid_sample.'''
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([T_ for T_ in temporal_lens], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    # value = value.flatten(2).transpose(1, 2).reshape(N_*M_, D_, 1, S_)
    sampling_value_list = []
    for lid_, T_ in enumerate(temporal_lens):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, 1, T_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode='bilinear',
            padding_mode='zeros', align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()