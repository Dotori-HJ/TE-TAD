import os
import json

import numpy as np
from tqdm import tqdm


def make_thumos14_ft_info(
    root='data/thumos14/features',
    gt_path='data/thumos14/gt.json',
    output_path='data/thumos14/ft_info.json',
    frames=16,
    stride=4,
    fps=None,
):
    result = {}
    with open(gt_path, 'rt') as f:
        gt = json.load(f)['database']

    feature_lengths = []
    video_names = os.listdir(root)
    for video_name in tqdm(video_names):
        path = os.path.join(root, video_name)

        features = np.load(path)
        video_name = os.path.splitext(video_name)[0]
        fps = gt[video_name]['fps']

        feature_length = len(features)
        feature_lengths.append(feature_length)
        feature_second = stride / fps * feature_length + (frames - stride) / fps
        result[video_name] = {
            "feature_length": feature_length,
            "feature_second": feature_second,
            "feature_fps": round(feature_length / feature_second, 2),
            "fps": fps,
            "base_frames": frames,
            "stride": stride,
        }

    with open(output_path, 'wt') as f:
        json.dump(result, f)

def make_activitynet_ft_info(
    root='data/activitynet/features',
    gt_path=None,
    output_path='data/activitynet/ft_info.json',
    frames=16,
    stride=16,
    fps=15,
):
    result = {}

    video_names = os.listdir(root)
    for video_name in tqdm(video_names):
        path = os.path.join(root, video_name)

        features = np.load(path)
        video_name = os.path.splitext(video_name)[0]

        feature_length = len(features)
        feature_second = stride / fps * feature_length + (frames - stride) / fps
        result[video_name[2:]] = {
            "feature_length": feature_length,
            "feature_second": feature_second,
            "feature_fps": round(feature_length / feature_second, 2),
            "fps": fps,
            "base_frames": frames,
            "stride": stride,
        }

    with open(output_path, 'wt') as f:
        json.dump(result, f)


def make_epic_kitchens_ft_info(
    root='data/epic_kitchens/features',
    gt_path=None,
    output_path='data/epic_kitchens/ft_info.json',
    frames=30,
    stride=16,
    fps=30,
):
    result = {}
    video_names = os.listdir(root)
    for video_name in tqdm(video_names):
        path = os.path.join(root, video_name)

        data = np.load(path)
        features = data['feats']
        video_name = os.path.splitext(video_name)[0]

        feature_length = len(features)
        feature_second = stride / fps * feature_length
        result[video_name] = {
            "feature_length": feature_length,
            "feature_second": feature_second,
            "feature_fps": feature_length / feature_second,
            "fps": fps,
            "base_frames": frames,
            "stride": stride,
        }

    with open(output_path, 'wt') as f:
        json.dump(result, f)


if __name__ == '__main__':
    make_thumos14_ft_info()
    # make_activitynet_ft_info()
    # make_epic_kitchens_ft_info()