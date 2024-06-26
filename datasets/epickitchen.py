import random
import json
import os.path as osp

import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset

from .data_util import load_feature, get_classes, truncate_feats, add_noise_to_segments


class EpicKitchen(Dataset):
    def __init__(self, feature_folder, gt_path, meta_info_path, subset, mode, ignored_videos, name_format='{}', transforms=None,
                 max_seq_len=None, downsample_rate=1.0, normalize=False, mem_cache=True, noise_scale=0, seg_noise_scale=0, binary=False):
        '''TADDataset
        Parameters:
            subset: train/val/test
            mode: train, or test
            ann_file: path to the ground truth file
            ft_info_file: path to the file that describe other information of each video
            transforms: which transform to use
            mem_cache: cache features of the whole dataset into memory.
            binary: transform all gt to binary classes. This is required for training a class-agnostic detector
            padding: whether to pad the input feature to `slice_len`
        '''

        super().__init__()
        self.feature_folder = feature_folder
        self.gt_path = gt_path
        self.meta_info_path = meta_info_path
        self.subset = subset
        self.mode = mode
        self.ignored_videos = ignored_videos
        self.name_format = name_format
        self.transforms = transforms
        self.max_seq_len = max_seq_len
        self.downsample_rate = downsample_rate
        self.normalize = normalize
        self.mem_cache = mem_cache
        self.binary = binary
        self.noise_scale = noise_scale
        self.seg_noise_scale = seg_noise_scale

        with open(gt_path, 'rt') as f:
            self.gt = json.load(f)['database']

        self.video_names = []
        for video_name, video_info in self.gt.items():
            # Filtering out 'Ambiguous' annotations
            video_info['annotations'] = [anno for anno in video_info['annotations'] if anno['label'] != 'Ambiguous']
            if video_info['subset'] == subset and not video_name in ignored_videos:
                # Appending video name to the list after removing ambiguous annotations
                self.video_names.append(video_name)

        self.classes = get_classes(self.gt)
        self.classname_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_classname = {idx: cls for idx, cls in enumerate(self.classes)}

        with open(meta_info_path, 'rt') as f:
            self.meta_info = json.load(f)
        self.meta_info = {
            video_name: info
            for video_name, info in self.meta_info.items()
            if video_name in self.video_names
        }

        if self.mem_cache:
            self.cache = {}
            for video_name in self.video_names:
                self.cache[video_name] = self._load_feature(video_name)

    def __len__(self):
        return len(self.video_names)

    def _load_feature(self, video_name):
        if self.mem_cache and video_name in self.cache:
            return self.cache[video_name]
        feature_path = osp.join(self.feature_folder, self.name_format.format(video_name))
        return load_feature(feature_path)

    def _load_annotations(self, video_name):
        annotations = self.gt[video_name]['annotations']
        segments, labels = [], []

        for anno in annotations:
            segments.append(torch.tensor(anno['segment'], dtype=torch.float32))
            labels.append(torch.tensor(self.classname_to_idx[anno['label']], dtype=torch.long))

        segments = torch.stack(segments)
        labels = torch.stack(labels)
        return segments, labels

    def __getitem__(self, i):
        video_name = self.video_names[i]
        features = self._load_feature(video_name)
        segments, labels = self._load_annotations(video_name)
        video_duration = self.gt[video_name]['duration']

        meta_info = self.meta_info[video_name]
        base_frames = meta_info['base_frames']
        stride = meta_info['stride']
        fps = meta_info['fps']

        if self.mode == 'train' and self.seg_noise_scale > 0:
            segments = add_noise_to_segments(segments, self.seg_noise_scale)

        if self.downsample_rate > 1:
            if self.mode == 'train':
                st_idx = random.randrange(min(features.size(1), self.downsample_rate))
                segments = segments - stride * st_idx / fps
            else:
                st_idx = 0
            features = features[:, st_idx::self.downsample_rate]
            stride = stride * self.downsample_rate

        if self.mode == 'train':
            features, segments, labels = truncate_feats(
                features, segments, labels,
                max_seq_len=self.max_seq_len,
                fps=fps,
                base_frames=base_frames,
                stride=stride,
                crop_ratio=[0.9, 1.0],
                trunc_thr=0.3,
                max_num_trials=200,
                has_action=True,
                no_trunc=False,
            )
        if stride != base_frames:
            offset = (base_frames - stride) * 0.5 / fps
            segments = segments - offset
        else:
            offset = 0
        feature_duration = features.size(1) * stride / fps

        if self.normalize:
            segments = segments / feature_duration

        if self.normalize:
            segments /= feature_duration

        if self.mode == 'train' and self.noise_scale > 0:
            features = features + self.noise_scale * torch.randn_like(features)
        info = {
            'video_name': video_name,
            'segments': segments,
            'labels': labels,
            'video_duration': torch.tensor(video_duration),
            'feature_duration': torch.tensor(feature_duration),
            'fps': torch.tensor(fps),
            'base_frames': torch.tensor(base_frames),
            'offset': torch.tensor(offset),
            'stride': torch.tensor(stride),
        }
        return features, info

def build_epickitchen(subset, mode, ignored_videos, args):
    return EpicKitchen(
        args.feature_folder,
        args.gt_path,
        args.meta_info_path,
        subset=subset,
        mode=mode,
        ignored_videos=ignored_videos,
        name_format=args.name_format,
        max_seq_len=args.max_seq_len,
        downsample_rate=args.downsample_rate,
        normalize=args.normalize,
        noise_scale=args.noise_scale,
        seg_noise_scale=args.seg_noise_scale,
        mem_cache=True,
    )