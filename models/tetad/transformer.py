# ------------------------------------------------------------------------
# Modified from TadTR (https://github.com/xlliu7/TadTR)
# Copyright (c) 2021. Xiaolong Liu.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------
# and DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_

from .ops.temporal_deform_attn import DeformAttn
from .utils import MLP, _get_activation_fn
from .modules import DropPath


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_deform_heads=2, base_scale=2.0, num_queries=300, window_size=128, max_queries=3000,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1, droppath=0.1,
                 activation="relu", return_intermediate_dec=False, fix_encoder_proposals=True,
                 mixed_selection=False, num_feature_levels=4, num_sampling_levels=4, dec_n_points=4,  enc_n_points=4, length_ratio=-1,
                 temperature=10000,
                 ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_deform_heads = n_deform_heads
        self.base_scale = base_scale
        self.num_queries = num_queries
        self.window_size = window_size
        self.max_queries = max_queries
        self.mixed_selection = mixed_selection
        self.fix_encoder_proposals = fix_encoder_proposals
        self.length_ratio = length_ratio

        encoder_layer = DeformableTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, droppath, activation,
            num_sampling_levels, n_deform_heads, enc_n_points
        )
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers, droppath)

        decoder_layer = DeformableTransformerDecoderLayer(
            d_model, dim_feedforward, dropout, droppath, activation,
            num_sampling_levels, n_heads, n_deform_heads, dec_n_points
        )
        self.decoder = DeformableTransformerDecoder(
            decoder_layer, num_decoder_layers,
            d_model=d_model, return_intermediate=return_intermediate_dec,
            temperature=temperature
        )
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.enc_output = nn.Linear(d_model, d_model)
        self.enc_output_norm = nn.LayerNorm(d_model)
        if not self.mixed_selection:
            self.enc_trans = nn.Linear(d_model, d_model)
            self.enc_trans_norm = nn.LayerNorm(d_model)

        self.enc_class_embed = None
        self.enc_segment_embed = None
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, DeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        temperature = 10000

        dim_t = torch.arange(self.d_model // 2, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.d_model)
        # N, L, 4, 128
        pos_ct = proposals[:, :, [0]] / dim_t
        pos_w = proposals[:, :, [1]] / dim_t
        pos_ct = torch.stack((pos_ct[:, :, 0::2].sin(), pos_ct[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_ct, pos_w), dim=2)
        # N, L, 4, 64, 2
        return pos

    def get_valid_ratio(self, mask):
        _, T = mask.shape
        valid_T = torch.sum(~mask, 1)
        valid_ratio = valid_T.float() / T
        return valid_ratio    # shape=(bs)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, temporal_lengths, grids, fps):
        N_, S_, C_ = memory.shape
        proposals = []
        _cur = 0
        for lvl, T_ in enumerate(temporal_lengths):
            timeline = grids[:, _cur:(_cur + T_)]

            scale = torch.ones_like(timeline) * (fps[..., None] * self.base_scale) * 2 ** lvl
            proposal = torch.stack((timeline, scale), -1).view(N_, -1, 2)
            proposals.append(proposal)
            _cur += T_

        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ~memory_padding_mask
        output_proposals[..., 1] = output_proposals[..., 1].log_()

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask[..., None], float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))

        return output_memory, output_proposals, output_proposals_valid


    def forward(self, srcs, masks, pos_embeds, grids,
                feature_durations, fps, query_embed=None,
                input_query_label=None, input_query_segment=None, attn_mask=None):
        '''
        Params:
            srcs: list of Tensor with shape (bs, c, t)
            masks: list of Tensor with shape (bs, t)
            pos_embeds: list of Tensor with shape (bs, c, t)
            query_embed: list of Tensor with shape (nq, 2c)
        Returns:
            hs: list, per layer output of decoder
            init_reference_out: reference points predicted from query embeddings
            inter_references_out: reference points predicted from each decoder layer
            memory: (bs, c, t), final output of the encoder
        '''
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        grid_flatten = []
        temporal_lengths = []
        for lvl, (src, mask, pos_embed, grid) in enumerate(zip(srcs, masks, pos_embeds, grids)):
            bs, c, t = src.shape
            temporal_lengths.append(t)
            # (bs, c, t) => (bs, t, c)
            src = src.transpose(1, 2)
            pos_embed = pos_embed.transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed = self.level_embed[lvl].repeat(bs, t, 1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
            grid_flatten.append(grid)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        grid_flatten = torch.cat(grid_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        temporal_lengths = torch.as_tensor(temporal_lengths, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((temporal_lengths.new_zeros((1, )), temporal_lengths.cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)   # (bs, nlevels)
        # deformable encoder
        memory = self.encoder(
            src_flatten, temporal_lengths,
            grid_flatten, feature_durations,
            level_start_index, valid_ratios,
            lvl_pos_embed_flatten, mask_flatten
        )  # shape=(bs, t, c)

        bs, t, c = memory.shape

        output_memory, output_proposals, output_proposals_valid = self.gen_encoder_output_proposals(
            memory, mask_flatten, temporal_lengths, grid_flatten.detach(), fps
        )
        enc_outputs_class = self.decoder.class_embed[-1](output_memory)

        enc_outputs_mask = ~output_proposals_valid
        valid_scores = enc_outputs_class[..., 0].masked_fill(enc_outputs_mask, float('-1e9'))

        if self.fix_encoder_proposals:
            enc_outputs_segment = output_proposals
        else:
            enc_outputs_segment = self.decoder.segment_embed[-1](output_memory)
            enc_outputs_segment = torch.stack([
                output_proposals[..., 0] + enc_outputs_segment[..., 0] * output_proposals[..., 1].exp().detach(),
                output_proposals[..., 1] + enc_outputs_segment[..., 1],
            ], dim=-1)

        # enc_outputs_segment = output_proposals
        if self.enc_class_embed is not None and self.enc_segment_embed is not None:
            # Hybrid
            enc_hybrid_outputs_class = self.enc_class_embed(output_memory)
            enc_hybrid_outputs_segement = self.enc_segment_embed(output_memory)
        else:
            enc_hybrid_outputs_class = None
            enc_hybrid_outputs_segement = None

        num_queries = self.num_queries if self.training else int(self.num_queries * 2)
        num_chunks = max(temporal_lengths[0] // self.window_size, 1)
        # num_chunks = 1
        if num_chunks > 1:
            _cur = 0
            chunked_valid_scores = []
            # chunked_memory = []
            chunked_enc_outputs_segment = []
            for lvl, T_ in enumerate(temporal_lengths):
                cur_valid_scores = valid_scores[:, _cur:_cur + T_]
                cur_enc_outputs_segment = enc_outputs_segment[:, _cur:_cur + T_]

                cur_valid_scores = torch.chunk(cur_valid_scores, num_chunks, dim=1)
                cur_enc_outputs_segment = torch.chunk(cur_enc_outputs_segment, num_chunks, dim=1)

                cur_valid_scores = torch.stack(cur_valid_scores, dim=1)
                cur_enc_outputs_segment = torch.stack(cur_enc_outputs_segment, dim=1)

                chunked_valid_scores.append(cur_valid_scores)
                chunked_enc_outputs_segment.append(cur_enc_outputs_segment)
                _cur += T_
            chunked_valid_scores = torch.cat(chunked_valid_scores, dim=2) # [bs, num_chunks, t, 1]
            chunked_enc_outputs_segment = torch.cat(chunked_enc_outputs_segment, dim=2)

            num_topk = num_queries if chunked_valid_scores.size(2) > num_queries else chunked_valid_scores.size(2)
            chunked_topk_valid_scores, chunked_topk_indices = torch.topk(chunked_valid_scores, k=num_topk, dim=2)
            topk_valid_scores = chunked_topk_valid_scores.view(bs, -1)

            topk_segments = torch.gather(chunked_enc_outputs_segment, 2, chunked_topk_indices.unsqueeze(-1).expand(-1, -1, -1, 2))
            topk_segments = topk_segments.view(bs, -1, 2)
        else:
            num_topk = num_queries if t > num_queries else t
            topk_valid_scores, topk_proposals = torch.topk(valid_scores, num_topk, dim=1)

            topk_segments = torch.gather(enc_outputs_segment, 1, topk_proposals.unsqueeze(-1).expand(-1, -1, 2))

        if topk_valid_scores.size(-1) > self.max_queries:
            topk_valid_scores, topk_proposals = torch.topk(topk_valid_scores, self.max_queries, dim=1)
            topk_segments = torch.gather(topk_segments, 1, topk_proposals.unsqueeze(-1).expand(-1, -1, 2))

        query_mask = ~(topk_valid_scores > float('-1e9'))
        tgt = torch.zeros(bs, query_mask.size(1), self.d_model, device=topk_segments.device)
        query_embed = None

        if input_query_label is not None and input_query_segment is not None:
            tgt = torch.cat([input_query_label, tgt], dim=1)
            topk_segments = torch.cat([input_query_segment, topk_segments], dim=1)

        # decoder
        hs, inter_grids, sampling_locations = self.decoder(
            tgt, topk_segments, feature_durations,
            memory, temporal_lengths, level_start_index,
            valid_ratios, mask_flatten, query_embed, query_mask,
            attn_mask,
        )

        return (
            hs, inter_grids, output_memory,
            enc_outputs_class, enc_outputs_segment, enc_outputs_mask, output_proposals,
            enc_hybrid_outputs_class, enc_hybrid_outputs_segement, mask_flatten, grid_flatten,
            temporal_lengths, sampling_locations, query_mask,
        )


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, droppath=0.1, activation="relu",
                 n_levels=4, n_deform_heads=2, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = DeformAttn(d_model, n_levels, n_deform_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.droppath = DropPath(droppath)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        input_src = src
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        src = self.droppath(src, input_src)
        return src

    def forward(self, src, pos, reference_points, temporal_lengths, level_start_index, padding_mask=None):
        input_src = src
        src2, _ = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, temporal_lengths, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.droppath(src, input_src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, droppath=0.1):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.droppath = droppath

    @staticmethod
    def get_reference_points(temporal_lengths, valid_ratios, device):
        reference_points_list = []
        for lvl, T_ in enumerate(temporal_lengths):
            ref = torch.linspace(0.5, T_ - 0.5, T_, dtype=torch.float32, device=device)  # (t,)
            ref = ref[None] / (valid_ratios[:, None, lvl] * T_)                          # (bs, t)
            reference_points_list.append(ref)

        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]          # (N, t, n_levels)
        return reference_points[..., None]

    def forward(self, src, temporal_lens, grids, feature_durations, level_start_index, valid_ratios, pos=None, padding_mask=None):
        '''
        src: shape=(bs, t, c)
        temporal_lens: shape=(n_levels). content: [t1, t2, t3, ...]
        level_start_index: shape=(n_levels,). [0, t1, t1+t2, ...]
        valid_ratios: shape=(bs, n_levels).
        '''
        output = src
        # (bs, t, levels, 1)
        normalized_grids = grids[:, :, None] / feature_durations[:, None, None]
        reference_points = normalized_grids[..., None] * valid_ratios[:, None, :, None]

        for _, layer in enumerate(self.layers):
            layer_output = layer(output, pos, reference_points, temporal_lens, level_start_index, padding_mask)
            if self.droppath > 0 and self.training:
                p = torch.rand(output.size(0), dtype=torch.float32, device=output.device)
                p = (p > self.droppath)[:, None, None]
                output = torch.where(p, layer_output, output)
            else:
                output = layer_output
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, droppath=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_deform_heads=2, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = DeformAttn(d_model, n_levels, n_deform_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.droppath = DropPath(droppath)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        input_tgt = tgt
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        tgt = self.droppath(tgt, input_tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, temporal_lengths, level_start_index, src_padding_mask=None, query_mask=None, attn_mask=None):
        # self attention
        input_tgt = tgt
        q = k = self.with_pos_embed(tgt, query_pos)

        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), key_padding_mask=query_mask, attn_mask=attn_mask)[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt = self.droppath(tgt, input_tgt)

        # cross attention
        input_tgt = tgt
        tgt2, (sampling_locations, attention_weights) = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, temporal_lengths, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt = self.droppath(tgt, input_tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt, sampling_locations


class DeformableTransformerDecoder(nn.Module):
    def __init__(
        self, decoder_layer, num_layers, d_model=256, return_intermediate=False, temperature=10000
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.segment_embed = None
        self.class_embed = None
        self.d_model = d_model
        self.grid_head = MLP(d_model * 2, d_model, d_model, 2)
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.temperature = temperature

    def get_proposal_pos_embed(self, proposals):
        scale = 2 * math.pi

        dim_t = torch.arange(self.d_model, dtype=torch.float32, device=proposals.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.d_model)
        # N, L, 4
        proposals = proposals * scale
        # N, L, 4, 128
        pos_ct = proposals[:, :, [0]] / dim_t
        pos_w = proposals[:, :, [1]] / dim_t
        pos_ct = torch.stack((pos_ct[:, :, 0::2].sin(), pos_ct[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_ct, pos_w), dim=2)
        # N, L, 4, 64, 2
        return pos

    def get_proposal_pos_embed2(self, proposals):
        scale = 2 * math.pi

        dim_t = torch.arange(self.d_model, dtype=torch.float32, device=proposals.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.d_model)
        # N, L, 4
        proposals = proposals * scale
        # N, L, 4, 128
        pos_ct = proposals[:, :, [0]] / dim_t
        pos_ct = torch.stack((pos_ct[:, :, 0::2].sin(), pos_ct[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = pos_ct
        # N, L, 4, 64, 2
        return pos


    def forward(self, tgt, enc_output_segments, feature_durations,
                src, temporal_lens, src_level_start_index, src_valid_ratios,
                src_padding_mask=None, query_pos=None, query_mask=None, attn_mask=None):
        '''
        tgt: [bs, nq, C]
        reference_points: [bs, nq, 1 or 2]
        src: [bs, T, C]
        src_valid_ratios: [bs, levels]
        '''
        output = tgt
        intermediate = []
        intermediate_grids = []
        segment_outputs = enc_output_segments.detach()
        reference_points = torch.stack([
            segment_outputs[..., 0], segment_outputs[..., 1].exp()
        ], dim=-1)
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points / feature_durations[:, None, None]
            reference_points_input = reference_points_input[:, :, None, :] * src_valid_ratios[:, None, :, None]

            grid_sine_embed = self.get_proposal_pos_embed(reference_points)
            raw_query_pos = self.grid_head(grid_sine_embed) # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos

            output, sampling_locations = layer(output, query_pos, reference_points_input, src, temporal_lens, src_level_start_index, src_padding_mask, query_mask, attn_mask)

            # segment refinement
            if self.segment_embed is not None:
                segment_outputs_detach = segment_outputs.detach()
                segment_outputs = self.segment_embed[lid](output)
                segment_outputs = torch.stack([
                    segment_outputs_detach[..., 0] + segment_outputs[..., 0] * segment_outputs_detach[..., 1].exp(),
                    segment_outputs_detach[..., 1] + segment_outputs[..., 1],
                ], dim=-1)

                new_reference_points = torch.stack([
                    segment_outputs[..., 0], segment_outputs[..., 1].exp()
                ], dim=-1)
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_grids.append(segment_outputs)
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_grids), sampling_locations

        return output, segment_outputs, sampling_locations


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def build_deformable_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        n_heads=args.n_heads,
        n_deform_heads=args.n_deform_heads,
        base_scale=args.base_scale,
        num_queries=args.num_queries,
        window_size=args.window_size,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        droppath=args.droppath,
        activation=args.transformer_activation,
        return_intermediate_dec=True,
        mixed_selection=args.mixed_selection,
        max_queries=args.max_queries,
        fix_encoder_proposals=args.fix_encoder_proposals,
        num_feature_levels=args.num_feature_levels,
        num_sampling_levels=args.num_sampling_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        length_ratio=args.length_ratio,
        temperature=args.temperature,
    )