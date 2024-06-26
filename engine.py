# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import logging
import math
import sys
from typing import Iterable

from util.utils import to_device

import torch

import util.misc as utils
from datasets import build_evaluator

@torch.no_grad()
def compute_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    torch.autograd.set_detect_anomaly(True)
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('norm_total', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('norm_enc', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('norm_dec', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0
    for samples, info in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        samples = samples.to(device)
        info = [{
            k: v.to(device) if not isinstance(v, str) else v
            for k, v in t.items()
        } for t in info]

        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(samples, info)

            loss_dict = criterion(outputs, info)
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            norm = compute_gradient_norm(model)
            norm_enc = compute_gradient_norm(model.transformer.encoder)
            norm_dec = compute_gradient_norm(model.transformer.decoder)
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(norm_total=norm)
        metric_logger.update(norm_enc=norm_enc)
        metric_logger.update(norm_dec=norm_dec)

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device, output_dir, wo_class_error=False, args=None, logger=None, epoch=0):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'
    action_evaluator = build_evaluator('val', args)

    _cnt = 0
    encoder_outptus = {}
    output_state_dict = {} # for debug only
    for samples, info in metric_logger.log_every(data_loader, 50, header, logger=logger):
        samples = samples.to(device)
        info = [{
            k: v.to(device) if not isinstance(v, str) else v
            for k, v in t.items()
        } for t in info]

        video_durations = torch.stack([t["video_duration"] for t in info], dim=0)
        feature_durations = torch.stack([t["feature_duration"] for t in info], dim=0)
        offsets = torch.stack([t["offset"] for t in info], dim=0)

        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(samples, info)
            loss_dict = criterion(outputs, info)

        scores = outputs['enc_outputs']['pred_logits'].sigmoid()
        segments = outputs['enc_outputs']['pred_segments']
        segments = torch.stack([
            segments[..., 0] - segments[..., 1].exp(), segments[..., 0] + segments[..., 1].exp()
        ], dim=-1)
        masks = ~outputs['enc_outputs']['mask']
        for i, tgt in enumerate(info):
            video_name = tgt['video_name']
            encoder_outptus[video_name] = {
                'pred_logits': scores[i][masks[i]],
                'pred_segments': segments[i][masks[i]],
            }

        results = postprocessors(outputs, video_durations, feature_durations, offsets, args.duration_thresh)
        pred = {
            t['video_name']: output
            for t, output in zip(info, results)
        }

        action_evaluator.update(pred)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        save_path = osp.join("outputs", f"results-{utils.get_rank()}.pkl")
        action_evaluator.dump_detection(save_path)
        print(f"Saving res to {save_path}")
        torch.save(output_state_dict, save_path)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if action_evaluator is not None:
        action_evaluator.synchronize_between_processes()
        action_evaluator.accumulate()
        # dump detections
        action_evaluator.summarize()

    stats = {}
    stats.update({f'{k}': meter.global_avg for k,meter in metric_logger.meters.items() if meter.count > 0})
    if action_evaluator is not None:
        for k, v in action_evaluator.stats.items():
            for vk, vv in v.items():
                stats[vk + '_' + k] = vv

        mAP_values = ' '.join([f'{k}: {100*v:.2f}'.format(k, v)
                              for k, v in stats.items() if k.startswith('mAP')])
        logging.info(mAP_values)

        stats['stats_summary'] = action_evaluator.stats_summary

    return stats, action_evaluator


@torch.no_grad()
def test(model, criterion, postprocessors, data_loader, device, output_dir, wo_class_error=False, args=None, logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    final_res = []
    for samples, grids, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)
        grids = grids.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        outputs = model(samples, grids)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=True)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        for image_id, outputs in res.items():
            _scores = outputs['scores'].tolist()
            _labels = outputs['labels'].tolist()
            _boxes = outputs['boxes'].tolist()
            for s, l, b in zip(_scores, _labels, _boxes):
                assert isinstance(l, int)
                itemdict = {
                        "image_id": int(image_id),
                        "category_id": l,
                        "bbox": b,
                        "score": s,
                        }
                final_res.append(itemdict)

    if args.output_dir:
        import json
        with open(args.output_dir + f'/results{args.rank}.json', 'w') as f:
            json.dump(final_res, f)

    return final_res
