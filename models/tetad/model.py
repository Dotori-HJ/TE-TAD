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
import torch.nn.functional as F
from torch import nn

from .matcher import build_matcher
from .utils import get_feature_grids, MLP
from .position_encoding import build_position_encoding
from util.misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)
from ..registry import MODULE_BUILD_FUNCS
from .modules import ConvBackbone
from util.segment_ops import segment_cw_to_t1t2, segment_t1t2_to_cw, segment_iou, diou_loss, log_ratio_width_loss
from .utils import sigmoid_focal_loss
from .transformer import build_deformable_transformer


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_norm(norm_type, dim, num_groups=None):
    if norm_type == 'gn':
        assert num_groups is not None, 'num_groups must be specified'
        return nn.GroupNorm(num_groups, dim)
    elif norm_type == 'bn':
        return nn.BatchNorm1d(dim)
    else:
        raise NotImplementedError


class TETAD(nn.Module):
    def __init__(self, position_embedding, transformer, num_classes, num_queries, feature_dim, num_feature_levels,
                 num_sampling_levels, kernel_size=3, num_cls_head_layers=3, num_reg_head_layers=3,
                 aux_loss=True, with_segment_refine=False, mixed_selection=False, hybrid=False,
                 emb_norm_type='bn', emb_relu=False, share_class_embed=False, share_segment_embed=False,
                 fix_encoder_proposals=True,):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.feature_dim = feature_dim
        self.num_sampling_levels = num_sampling_levels
        self.num_feature_levels = num_feature_levels
        self.mixed_selection = mixed_selection
        self.hybrid = hybrid
        self.emb_norm_type = emb_norm_type
        self.emb_relu = emb_relu
        self.fix_encoder_proposals = fix_encoder_proposals
        self.num_cls_head_layers = num_cls_head_layers
        self.num_reg_head_layers = num_reg_head_layers


        self.label_enc = None

        if mixed_selection:
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
            nn.init.normal_(self.query_embed.weight.data)

        self.input_proj = ConvBackbone(
            feature_dim, hidden_dim, kernel_size=kernel_size,
            arch=(1, 0), num_feature_levels=num_feature_levels,
            with_ln=True,
        )
        self.position_embedding = position_embedding
        self.aux_loss = aux_loss
        self.with_segment_refine = with_segment_refine

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        if num_cls_head_layers > 1:
            self.class_embed = MLP(hidden_dim, hidden_dim, num_classes, num_layers=num_cls_head_layers)
            nn.init.constant_(self.class_embed.layers[-1].bias, bias_value)
        else:
            self.class_embed = nn.Linear(hidden_dim, num_classes)
            nn.init.constant_(self.class_embed.bias, bias_value)

        if share_class_embed:
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(transformer.decoder.num_layers)])
        else:
            self.class_embed = _get_clones(self.class_embed, transformer.decoder.num_layers)

        enc_class_embed = nn.Linear(hidden_dim, 1)
        nn.init.constant_(enc_class_embed.bias, bias_value)
        self.class_embed.append(enc_class_embed)
        self.transformer.decoder.class_embed = self.class_embed

        if num_reg_head_layers > 1:
            self.segment_embed = MLP(hidden_dim, hidden_dim, 2, num_layers=num_reg_head_layers)
            nn.init.zeros_(self.segment_embed.layers[-1].weight)
            nn.init.zeros_(self.segment_embed.layers[-1].bias)
        else:
            self.segment_embed = nn.Linear(hidden_dim, 2)
            nn.init.zeros_(self.segment_embed.bias)

        if share_segment_embed:
            self.segment_embed = nn.ModuleList([self.segment_embed for _ in range(transformer.decoder.num_layers)])
            if not self.fix_encoder_proposals:
                enc_embed = MLP(hidden_dim, hidden_dim, 2, num_layers=3)
                nn.init.zeros_(enc_embed.layers[-1].weight)
                nn.init.zeros_(enc_embed.layers[-1].bias)
                self.segment_embed.append(enc_embed)
        else:
            if not self.fix_encoder_proposals:
                num_layers = transformer.decoder.num_layers + 1
            else:
                num_layers = transformer.decoder.num_layers
            self.segment_embed = _get_clones(self.segment_embed, num_layers)
        self.transformer.decoder.segment_embed = self.segment_embed

        if hybrid:
            self.transformer.enc_class_embed = nn.Linear(hidden_dim, num_classes)
            nn.init.constant_(self.transformer.enc_class_embed.bias, bias_value)
            self.transformer.enc_segment_embed = MLP(hidden_dim, hidden_dim, 3, num_layers=3)
            nn.init.zeros_(self.transformer.enc_segment_embed.layers[-1].weight)
            nn.init.zeros_(self.transformer.enc_segment_embed.layers[-1].bias)

    def _to_roi_align_format(self, rois, T, scale_factor=1):
        '''Convert RoIs to RoIAlign format.
        Params:
            RoIs: normalized segments coordinates, shape (batch_size, num_segments, 2)
            T: length of the video feature sequence
        '''
        # transform to absolute axis
        B, N = rois.shape[:2]
        # rois_center = rois[:, :, 0:1]
        # rois_size = rois[:, :, 1:2]# * scale_factor
        rois_abs = rois * T
        # rois_abs = torch.cat(
        #     (rois_center - rois_size, rois_center + rois_size), dim=2) * T
        # expand the RoIs
        # rois_abs = torch.clamp(rois_abs, min=0, max=T)  # (N, T, 2)
        # add batch index
        batch_ind = torch.arange(0, B).view((B, 1, 1)).to(rois_abs.device)
        batch_ind = batch_ind.repeat(1, N, 1)
        rois_abs = torch.cat((batch_ind, rois_abs), dim=2)
        # NOTE: stop gradient here to stablize training
        return rois_abs.view((B*N, 3)).detach()

    def forward(self, samples: NestedTensor, info):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            or a tuple of tensors and mask

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-action) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_segments": The normalized segments coordinates for all queries, represented as
                               (center, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized segment.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            if isinstance(samples, (list, tuple)):
                samples = NestedTensor(*samples)
            else:
                samples = nested_tensor_from_tensor_list(samples)  # (n, c, t)

        # Multi scale
        # features = self.fpn(self.tdm(samples.tensors))

        # srcs = [feature for feature in features]
        # masks = [F.interpolate(samples.mask[None].float(), size=feature.size(-1), mode='nearest').to(torch.bool)[0] for feature in features]
        # pos = [self.position_embedding(NestedTensor(feature, mask)) for feature, mask in zip(features, masks)]

        features = samples
        src, mask = features.tensors, features.mask
        srcs, masks = self.input_proj(features.tensors, features.mask)

        fps = torch.stack([item['fps'] for item in info if 'fps' in item])
        stride = torch.stack([item['stride'] for item in info if 'stride' in item])
        feature_durations = torch.stack([item['feature_duration'] for item in info if 'feature_duration' in item])

        grid = get_feature_grids(mask, fps, stride, stride)
        grids = [grid]
        poss = [self.position_embedding(grid)]

        cur_stride = stride
        if self.num_feature_levels > 1:
            for l in range(1, self.num_feature_levels):

                mask = masks[l]
                cur_stride = stride * 2 ** l
                grid = get_feature_grids(mask, fps, cur_stride, cur_stride)
                # for i, (g, m, m2) in enumerate(zip(grids[-1], mask, masks[-1])):
                for i, (g, m, m2) in enumerate(zip(grids[-1], mask, masks[l-1])):
                    g = F.interpolate(g[None, None, ~m2], size=(~m).sum().item(), mode='linear')[0, 0, :]
                    grid[i, :g.size(0)] = g
                pos = self.position_embedding(grid)

                grids.append(grid)
                poss.append(pos)

        if self.mixed_selection:
            query_embed = self.query_embed.weight
        else:
            query_embed = None
        (
            hs, inter_grids, enc_memory,
            enc_outputs_class, enc_outputs_coord_unact, enc_mask, enc_proposals,
            enc_hybrid_outputs_class, enc_hybrid_outputs_segement, enc_hybrid_mask, enc_hybrid_grid,
            temporal_lengths, sampling_locations, query_mask#, topk_proposals
        ) = self.transformer(
            srcs, masks, poss, grids,
            feature_durations, fps, query_embed,
            None, None, None, None,
        )

        outputs_classes, outputs_coords = [], []
        # gather outputs from each decoder layer
        for lvl in range(hs.shape[0]):
            outputs_class = self.class_embed[lvl](hs[lvl])
            output_segments = inter_grids[lvl]

            outputs_classes.append(outputs_class)
            outputs_coords.append(output_segments)
        outputs_class = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        out = {
            'pred_logits': outputs_class[-1],
            'pred_segments': outputs_coords[-1],
            'mask': query_mask,
        }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coords, query_mask)

        if self.hybrid:
            out['enc_hybrid_outputs'] = {
                'pred_logits': enc_hybrid_outputs_class,
                'pred_binary_logits': enc_outputs_class,
                'pred_segments': enc_hybrid_outputs_segement,
                'pred_cw_segments': enc_outputs_coord_unact,
                'proposals': enc_proposals,
                'mask': enc_mask,
                'grids': enc_hybrid_grid,
                'temporal_lengths': temporal_lengths,
                'fps': fps,
            }

        out['enc_outputs'] = {
            'pred_logits': enc_outputs_class,
            'pred_segments': enc_outputs_coord_unact,
            'mask': enc_mask,
            'proposals': enc_proposals,
        }

        # out['reference'] = reference
        out['sampling_locations'] = sampling_locations

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, query_mask):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_segments': b, 'mask': query_mask}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for TadTR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth segments and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and segment)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, diou=False, base_scale=3.0, label_smoothing=0):
        """ Create the criterion.
        Parameters:
            num_classes: number of action categories, omitting the special no-action category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.label_smoothing = label_smoothing
        self.diou = diou
        self.base_scale = base_scale

    def loss_hybrid(self, outputs, targets, num_segments):
        assert 'pred_logits' in outputs
        assert 'pred_segments' in outputs
        # assert 'pred_enc_logits' in outputs
        assert 'mask' in outputs
        assert 'grids' in outputs
        pred_logits = outputs['pred_logits']
        pred_ctl_segments = outputs['pred_segments']
        pred_binary_logits = outputs['pred_binary_logits']
        pred_cw_segments = outputs['pred_cw_segments']
        pred_cw_proposals = outputs['proposals']
        mask = outputs['mask']
        grids = outputs['grids']
        fps = outputs['fps']
        pred_cw_segments = torch.stack([
            pred_cw_segments[..., 0] * pred_cw_proposals[..., 1].exp().detach() + pred_cw_proposals[..., 0],
            pred_cw_segments[..., 1] + pred_cw_proposals[..., 1],
        ], dim=-1)
        pred_ctl_segments = torch.stack([
            grids,
            pred_ctl_segments[..., 1],
            pred_ctl_segments[..., 2],
        ], dim=-1)
        temporal_lengths = outputs['temporal_lengths']
        segments = [t['segments'] for t in targets]
        gt_labels = [t['labels'] for t in targets]

        target_labels = torch.zeros_like(pred_logits)
        target_centerness = torch.zeros_like(pred_binary_logits)

        # Generate Labels
        num_level = len(temporal_lengths)
        scale = torch.stack([(self.base_scale * fps) * 2 ** lvl for lvl in range(num_level)], dim=0)
        scale = torch.log(scale)
        scale = (scale[:-1] + scale[1:]) * 0.5
        scale = scale.exp()

        scale_range = []
        for lvl in range(num_level):  # Exclude the last value
            start = scale[lvl - 1] if lvl != 0 else torch.zeros_like(scale[0])
            end = scale[lvl] if (lvl + 1) != num_level else torch.full_like(scale[-1], float('inf'))
            r = torch.stack([start, end], dim=-1)
            scale_range.append(r)
        scale_range = torch.stack(scale_range, dim=1)


        pos_segments_ctl, pos_segments_cw, pos_length_targets, pos_targets_t1t2 = [], [], [], []
        for b_idx, (seg, label, scale) in enumerate(zip(segments, gt_labels, scale_range)):
            st = seg[..., 0]
            et = seg[..., 1]

            segment_length = et - st
            _cur = 0
            for lvl, T_ in enumerate(temporal_lengths):
                timeline = grids[b_idx, _cur:(_cur + T_)]

                inbound_mask = torch.logical_and(
                    segment_length >= scale_range[b_idx, lvl, 0],
                    segment_length < scale_range[b_idx, lvl, 1]
                )
                inbound_segments = seg[inbound_mask]

                for (s, e), class_idx in zip(inbound_segments, label):
                    m = torch.logical_and(timeline > s, timeline < e)
                    masked_indices_ = m.nonzero(as_tuple=True)[0]
                    masked_indices = masked_indices_ + _cur

                    target_labels[b_idx, masked_indices, class_idx.item()] = 1
                    pos_ctl_segment = pred_ctl_segments[b_idx, masked_indices]
                    pred_cw_segment = pred_cw_segments[b_idx, masked_indices]
                    masked_timeline = timeline[masked_indices_]

                    s_ctl = pos_ctl_segment[..., 0] - pos_ctl_segment[..., 1].exp()
                    e_ctl = pos_ctl_segment[..., 0] + pos_ctl_segment[..., 1].exp()
                    l = torch.abs(masked_timeline - s_ctl)
                    r = torch.abs(e_ctl - masked_timeline)
                    current_centerness = torch.minimum(l, r) / torch.maximum(l, r)
                    target_centerness[b_idx, masked_indices, 0] = torch.maximum(target_centerness[b_idx, masked_indices, 0], current_centerness)
                    pos_segments_ctl.append(pos_ctl_segment)
                    pos_segments_cw.append(pred_cw_segment)
                    pos_length_target = torch.stack([masked_timeline - s, e - masked_timeline], dim=-1)
                    pos_length_targets.append(pos_length_target)
                    pos_targets_t1t2.append(torch.tensor([s, e], device=m.device)[None, :].repeat(pos_ctl_segment.size(0), 1))

                _cur += T_


        if len(pos_segments_ctl) > 0:
            pos_segments_ctl = torch.cat(pos_segments_ctl)
            pos_length_targets = torch.cat(pos_length_targets)
            pos_targets_t1t2 = torch.cat(pos_targets_t1t2)
            pos_segments_cw = torch.cat(pos_segments_cw)
            num_pos = max(len(pos_segments_ctl), 1)
        else:
            num_pos = 1

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_pos)

        if len(pos_segments_ctl) > 0:
            loss_segment = F.smooth_l1_loss(pos_segments_ctl[..., 1:], pos_length_targets.log().detach(), reduction='none')
            loss_segment = loss_segment.sum() / num_pos

            pos_segments_t1t2 = torch.stack([
                pos_segments_ctl[..., 0] - pos_segments_ctl[..., 1].exp(),
                pos_segments_ctl[..., 0] + pos_segments_ctl[..., 2].exp(),
            ], dim=-1)

            loss_iou = 1 - torch.diag(
                segment_iou(
                    pos_segments_t1t2,
                    pos_targets_t1t2.detach(),
                )
            )
            loss_iou = loss_iou.sum() / num_pos

        else:
            loss_iou = 0
            loss_segment = 0

        if self.label_smoothing != 0:
            target_labels *= 1 - self.label_smoothing
            target_labels += self.label_smoothing / (target_labels.size(-1) + 1)

        loss_ce = sigmoid_focal_loss(pred_logits, target_labels, mask, num_pos, alpha=self.focal_alpha, gamma=2)

        losses = {
            'loss_ce': loss_ce,
            'loss_segments': loss_segment,
            'loss_iou': loss_iou,
        }
        return losses

    def loss_actionness(self, outputs, targets, mask):
        """Compute the actionness regression loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center, width), normalized by the video length.
        """
        assert 'pred_segments' in outputs
        if not 'pred_actionness' in outputs:
            return {}
        assert 'pred_actionness' in outputs
        src_segments = outputs['pred_segments'].reshape((-1, 2))
        target_segments = torch.cat([t['segments'] for t in targets], dim=0)

        src_segments = torch.stack([
            src_segments[..., 0], src_segments[..., 1].exp()
        ], dim=-1)
        losses = {}

        iou_mat = segment_iou(
            segment_cw_to_t1t2(src_segments),
            target_segments
        )

        gt_iou = iou_mat.max(dim=1)[0]
        pred_actionness = outputs['pred_actionness']
        if mask is not None:
            valid = ~mask.view(-1)
            loss_actionness = F.l1_loss(pred_actionness.view(-1)[valid], gt_iou.view(-1).detach()[valid])
        else:
            loss_actionness = F.l1_loss(pred_actionness.view(-1), gt_iou.view(-1).detach())

        losses['loss_actionness'] = loss_actionness
        return losses

    def loss_labels(self, outputs, targets, indices, num_segments, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_segments]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        target_classes_onehot = F.one_hot(target_classes_o, num_classes=src_logits.shape[2]).to(src_logits.dtype)

        target_onehot_shape = [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]]
        target_classes_onehot_full = torch.zeros(target_onehot_shape, dtype=src_logits.dtype, device=src_logits.device)
        target_classes_onehot_full[idx] = target_classes_onehot

        if self.label_smoothing != 0:
            target_classes_onehot_full *= 1 - self.label_smoothing
            target_classes_onehot_full += self.label_smoothing / (target_classes_onehot.size(-1) + 1)
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot_full, outputs['mask'], num_segments, alpha=self.focal_alpha, gamma=2)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    def loss_segments(self, outputs, targets, indices, num_segments):
        """Compute the losses related to the segmentes, the L1 regression loss and the IoU loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center, width), normalized by the video length.
        """
        assert 'pred_segments' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_segments = outputs['pred_segments'][idx]
        target_segments = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_segment = log_ratio_width_loss(src_segments, segment_t1t2_to_cw(target_segments))
        src_segments = torch.stack([
            src_segments[..., 0], src_segments[..., 1].exp()
            # src_segments[..., 0], F.softplus(src_segments[..., 1])
        ], dim=-1)
        if self.diou:
            loss_iou = diou_loss(segment_cw_to_t1t2(src_segments), target_segments)
        else:
            loss_iou = 1 - torch.diag(
                segment_iou(
                    segment_cw_to_t1t2(src_segments),
                    target_segments,
                )
            )

        losses = {}
        losses['loss_segments'] = loss_segment.sum() / num_segments
        losses['loss_iou'] = loss_iou.sum() / num_segments
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_soft_loss(self, loss, outputs, targets, approx_match_matrices, num_segments, **kwargs):
        loss_map = {
            'labels': self.loss_labels_soft,
            'segments': self.loss_segments_soft,
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, approx_match_matrices, num_segments, **kwargs)

    def get_loss(self, loss, outputs, targets, indices, num_segments, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'segments': self.loss_segments,
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_segments, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target segments accross all nodes, for normalization purposes
        num_segments = sum(len(t["labels"]) for t in targets)
        num_segments = torch.as_tensor([num_segments], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_segments)
        num_segments = torch.clamp(num_segments / get_world_size(), min=1).item()

        losses = {}

        # Compute all the requested losses
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_segments, **kwargs))
        if 'pred_actionness' in outputs:
            losses.update(self.loss_actionness(outputs, targets, outputs['mask']))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    # we do not compute actionness loss for aux outputs
                    if 'actionness' in loss:
                        continue

                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_segments, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)


        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])

            indices = self.matcher(enc_outputs, bin_targets, encoder=True)
            for loss in self.losses:
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False

                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_segments, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        if 'enc_hybrid_outputs' in outputs:
            enc_hybrid_outputs = outputs['enc_hybrid_outputs']
            l_dict = self.loss_hybrid(enc_hybrid_outputs, targets, num_segments)
            l_dict = {k + f'_enc_hybrid': v for k, v in l_dict.items()}
            losses.update(l_dict)

        # self.indices = indices
        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the TADEvaluator"""

    @torch.no_grad()
    def forward(self, outputs, video_durations, feature_durations, offsets, duration_thresh=0.05):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size] containing the duration of each video of the batch
        """
        out_logits, out_segments = outputs['pred_logits'], outputs['pred_segments']
        # out_logits, out_segments = outputs['aux_outputs'][0]['pred_logits'], outputs['aux_outputs'][0]['pred_segments']
        query_mask = outputs['mask'] if 'mask' in outputs else None
        assert len(out_logits) == len(video_durations)
        assert len(out_logits) == len(feature_durations)
        # assert target_sizes.shape[1] == 1

        bs = out_logits.size(0)
        prob = out_logits.sigmoid()   # [bs, nq, C]
        if 'pred_actionness' in outputs:
            prob *= outputs['pred_actionness']
        scores, labels = prob.max(dim=-1)
        if out_segments.size(-1) == 2:
            segments = torch.stack([
                out_segments[..., 0], out_segments[..., 1].exp()
            ], dim=-1)
            segments = segment_cw_to_t1t2(segments) + offsets[:, None, None]
        else:
            segments = torch.stack([
                out_segments[..., 0] - out_segments[..., 1].exp(),
                out_segments[..., 0] + out_segments[..., 2].exp(),
            ], dim=-1)
            segments = segments + offsets[:, None, None]

        results = []
        for i in range(bs):
            cur_scores, cur_labels, cur_segments = scores[i], labels[i], segments[i]
            cur_segments = torch.clip(cur_segments, 0, video_durations[i].item())

            valid_mask = (cur_segments[..., 1] - cur_segments[..., 0]) > duration_thresh
            if query_mask is not query_mask:
                valid_mask = torch.logical_and(valid_mask, ~query_mask[i])
            else:
                valid_mask = valid_mask
            cur_scores, cur_labels, cur_segments = cur_scores[valid_mask], cur_labels[valid_mask], cur_segments[valid_mask]

            results.append({
                'scores': cur_scores,
                'labels': cur_labels,
                'segments': cur_segments,
            })
        return results


@MODULE_BUILD_FUNCS.registe_with_name(module_name='tetad')
def build(args):
    if args.binary:
        num_classes = 1
    else:
        num_classes = args.num_classes

    pos_embed = build_position_encoding(args)
    transformer = build_deformable_transformer(args)

    model = TETAD(
        pos_embed,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        feature_dim=args.feature_dim,
        num_sampling_levels=args.num_sampling_levels,
        num_feature_levels=args.num_feature_levels,
        kernel_size=args.kernel_size,
        num_cls_head_layers=args.num_cls_head_layers,
        num_reg_head_layers=args.num_reg_head_layers,
        aux_loss=args.aux_loss,
        with_segment_refine=False,
        mixed_selection=args.mixed_selection,
        hybrid=args.hybrid,
        emb_norm_type=args.emb_norm_type,
        emb_relu=args.emb_relu,
        fix_encoder_proposals=args.fix_encoder_proposals,
        share_class_embed=args.share_class_embed,
        share_segment_embed=args.share_segment_embed
    )

    matcher = build_matcher(args)
    losses = ['labels', 'segments']

    weight_dict = {
        'loss_ce': args.cls_loss_coef,
        'loss_segments': args.seg_loss_coef,
        'loss_iou': args.iou_loss_coef,
        'loss_actionness': 1,
    }

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc_hybrid': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)


    criterion = SetCriterion(num_classes, matcher,
        weight_dict, losses, focal_alpha=args.focal_alpha,
        diou=args.diou, base_scale=args.base_scale,
        label_smoothing=args.label_smoothing,
    )

    postprocessor = PostProcess()

    return model, criterion, postprocessor
