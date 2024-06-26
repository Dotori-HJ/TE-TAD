# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .thumos14 import build_thumos14
from .epickitchen import build_epickitchen
from .activitynet import build_anet
from .action_eval import TADEvaluator

def get_dataset_info(args):
    if args.dataset_name == 'thumos14':
        subset_mapping = {'train': 'val', 'val': 'test'}
        ignored_videos = ['video_test_0000270', 'video_test_0001292', 'video_test_0001496']
    else:
        subset_mapping = {'train': 'training', 'val': 'validation'}
        ignored_videos = []
    return subset_mapping, ignored_videos


def build_evaluator(subset, args):
    subset_mapping, ignored_videos = get_dataset_info(args)
    return TADEvaluator(
        args.gt_path, subset_mapping[subset], ignored_videos, args.extra_cls_path,
        args.nms_mode, args.iou_range, args.display_metric_indices,
        args.nms_thr, args.nms_sigma, args.voting_thresh, args.min_score, args.nms_multi_class, args.eval_topk, args.eval_workers, args.binary,
    )

def build_dataset(subset, mode, args):
    subset_mapping, ignored_videos = get_dataset_info(args)
    if args.dataset_name == 'thumos14':
        return build_thumos14(subset_mapping[subset], mode, ignored_videos, args)
    if 'epickitchens' in args.dataset_name:
        return build_epickitchen(subset_mapping[subset], mode, ignored_videos, args)
    if args.dataset_name == 'activitynet':
        return build_anet(subset_mapping[subset], mode, ignored_videos, args)

    raise ValueError(f'dataset {args.dataset_file} not supported')
