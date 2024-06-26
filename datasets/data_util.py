import os
import time
import random
import shutil
import logging
import datetime

import numpy as np

import torch

from util.slconfig import SLConfig
from util.segment_ops import get_time_coverage, segment_length

class Error(OSError):
    pass

def add_segment_occlusions(features, segment_length, max_occlusion_ratio=0.5, occlusion_value=0):
    """
    Add occlusions within a segment without covering the whole segment.

    :param features: torch.Tensor, the feature tensor to augment.
    :param segment_length: int, the length of the full segment.
    :param max_occlusion_ratio: float, the maximum ratio of the segment that can be occluded.
    :param occlusion_value: int or float, the value to set for occlusion.
    :return: torch.Tensor, the augmented feature tensor with occlusion.
    """
    # occlusions_to_add = random.randint(1, 3)  # Decide on a number of occlusions to add
    max_occlusion_length = int(segment_length * max_occlusion_ratio)
    if max_occlusion_length > 0:
        num_occlusion = random.randint(1, 3)
        for i in range(num_occlusion):
            occlusion_length = random.randint(1, max_occlusion_length)
            start = random.randint(0, segment_length - occlusion_length)
            features[start:start+occlusion_length] = occlusion_value

    return features

def add_background_occlusions(features, segments, max_occlusion_ratio=0.1, occlusion_value=0):
    """
    Add occlusions to the background (non-segment) areas of the features.

    :param features: torch.Tensor, the feature tensor to augment.
    :param segments: torch.Tensor, the segments within the features.
    :param max_occlusion_ratio: float, the maximum ratio of each background area that can be occluded.
    :param occlusion_value: int or float, the value to set for occlusion.
    :return: torch.Tensor, the augmented feature tensor with background occlusions.
    """
    total_length = features.size(1)

    # Create a mask with the same length as the features, initialized to False
    background_mask = torch.ones(total_length, dtype=torch.bool)

    # Mask out the segment areas (these are not background)
    for segment in segments:
        segment_start, segment_end = int(segment[0].item()), int(segment[1].item())
        background_mask[segment_start:segment_end] = False

    # Find indices where the background is True
    background_indices = torch.nonzero(background_mask).squeeze()

    if len(background_indices) > 0:
        # Randomly determine the number of occlusions based on the occlusion ratio
        num_occlusions = int(len(background_indices) * max_occlusion_ratio)

        # Select random indices from the background to occlude
        occlusion_indices = background_indices[random.sample(range(len(background_indices)), num_occlusions)]

        # Apply the occlusion to the selected indices
        features[:, occlusion_indices] = occlusion_value

    return features

def add_noise_to_segments(segments, noise_level):
    segment_lengths = segments[:, 1] - segments[:, 0]
    noise_std = noise_level * segment_lengths

    start_noise = (torch.rand_like(segment_lengths) * 2 - 1) * noise_std
    end_noise = (torch.rand_like(segment_lengths) * 2 - 1) * noise_std

    new_starts = torch.clamp(segments[:, 0] + start_noise, min=0)
    new_ends = torch.clamp(segments[:, 1] + end_noise, min=new_starts)

    new_segments = torch.stack((new_starts, new_ends), dim=1)

    return new_segments

def truncate_feats(
    features,
    segments,
    labels,
    max_seq_len,
    fps,
    base_frames,
    stride,
    trunc_thr=0.5,
    crop_ratio=None,
    max_num_trials=200,
    has_action=True,
    no_trunc=False,
):
    # Modified from Actionformer https://github.com/happyharrycn/actionformer_release/blob/main/libs/datasets/data_utils.py
    feat_len = features.size(1)

    # seq_len < max_seq_len
    if feat_len <= max_seq_len:
        # do nothing
        if crop_ratio == None:
            return features, segments, labels
        # randomly crop the seq by setting max_seq_len to a value in [l, r]
        else:
            max_seq_len = random.randint(
                max(round(crop_ratio[0] * feat_len), 1),
                min(round(crop_ratio[1] * feat_len), feat_len),
            )
            # # corner case
            if feat_len == max_seq_len:
                return features, segments, labels

    # otherwise, deep copy the dict
    feature_duration = get_time_coverage(
        max_seq_len, fps, base_frames, stride,
    )

    # try a few times till a valid truncation with at least one action
    for _ in range(max_num_trials):
        # sample a random truncation of the video feats
        start_idx = random.randint(0, feat_len - max_seq_len)
        st = start_idx * stride / fps
        et = st + feature_duration


        # compute the intersection between the sampled window and all segments
        window = torch.tensor([st, et]).unsqueeze(0)

        l = torch.max(window[:, 0], segments[:, 0])
        r = torch.min(window[:, 1], segments[:, 1])
        inter = (r - l).clamp(min=0)  # [N,M]

        segment_window = segment_length(segments)
        inter_ratio = inter / segment_window

        # only select those segments over the thresh
        valid = inter_ratio > trunc_thr  # ensure center point

        if no_trunc:
            # with at least one action and not truncating any actions
            seg_trunc_idx = torch.logical_and(
                (inter_ratio > 0.0), (inter_ratio < 1.0)
            )
            if (valid.sum().item() > 0) and (seg_trunc_idx.sum().item() == 0):
                break
        elif has_action:
            # with at least one action
            if valid.sum().item() > 0:
                break
        else:
            # without any constraints
            break

    # feats: C x T
    features = features[:, start_idx:start_idx + max_seq_len].clone()
    # segments: N x 2 in feature grids
    segments = segments[valid] - st
    # shift the time stamps due to truncation
    segments = torch.clamp(segments, 0, feature_duration)
    # labels: N
    labels = labels[valid].clone()

    return features, segments, labels


def get_feature_grid(feature_length, fps=30, window_size=16, stride=4):
    # Create feature indices: [0, 1, 2, ..., total_features - 1]
    feature_indices = torch.arange(0, feature_length)

    # Calculate the center frame index for each feature
    center_frame_indices = feature_indices * stride + window_size // 2

    # Convert the center frame indices to time in seconds
    feature_grid = center_frame_indices.float() / fps

    return feature_grid


def get_time_coverage(feature_length, fps=30, window_size=16, stride=4):
    # return stride / fps * feature_length + (window_size - stride) / fps
    return ((feature_length - 1) * stride + window_size) / fps


def load_feature(ft_path, shape=None):
    ext = os.path.splitext(ft_path)[-1]
    if ext == '.npy':
        video_df = torch.from_numpy(np.load(ft_path).T).float()
    elif ext == 'torch' or ext == '':
        video_df = torch.load(ft_path).T
    elif ext == '.npz':
        video_df = torch.from_numpy(np.load(ft_path)['feats'].T).float()
    elif ext == '.pkl':
        # video_df = torch.from_numpy(np.load(ft_path, allow_pickle=True))
        feats = np.load(ft_path, allow_pickle=True)
        # 1 x 2304 x T --> T x 2304
        video_df = torch.concat([feats['slow_feature'], feats['fast_feature']], dim=1).squeeze(0)#.transpose(0, 1)
    else:
        raise ValueError('unsupported feature format: {}'.format(ext))
    return video_df


def get_classes(gt):
    '''get class list from the annotation dict'''
    if 'classes' in gt:
        classes = gt['classes']
    else:
        database = gt
        all_gts = []
        for vid in database:
            all_gts += database[vid]['annotations']
        classes = list(sorted({x['label'] for x in all_gts}))
    return classes


def slcopytree(src, dst, symlinks=False, ignore=None, copy_function=shutil.copyfile,
             ignore_dangling_symlinks=False):
    """
    modified from shutil.copytree without copystat.

    Recursively copy a directory tree.

    The destination directory must not already exist.
    If exception(s) occur, an Error is raised with a list of reasons.

    If the optional symlinks flag is true, symbolic links in the
    source tree result in symbolic links in the destination tree; if
    it is false, the contents of the files pointed to by symbolic
    links are copied. If the file pointed by the symlink doesn't
    exist, an exception will be added in the list of errors raised in
    an Error exception at the end of the copy process.

    You can set the optional ignore_dangling_symlinks flag to true if you
    want to silence this exception. Notice that this has no effect on
    platforms that don't support os.symlink.

    The optional ignore argument is a callable. If given, it
    is called with the `src` parameter, which is the directory
    being visited by copytree(), and `names` which is the list of
    `src` contents, as returned by os.listdir():

        callable(src, names) -> ignored_names

    Since copytree() is called recursively, the callable will be
    called once for each directory that is copied. It returns a
    list of names relative to the `src` directory that should
    not be copied.

    The optional copy_function argument is a callable that will be used
    to copy each file. It will be called with the source path and the
    destination path as arguments. By default, copy2() is used, but any
    function that supports the same signature (like copy()) can be used.

    """
    errors = []
    if os.path.isdir(src):
        names = os.listdir(src)
        if ignore is not None:
            ignored_names = ignore(src, names)
        else:
            ignored_names = set()

        os.makedirs(dst)
        for name in names:
            if name in ignored_names:
                continue
            srcname = os.path.join(src, name)
            dstname = os.path.join(dst, name)
            try:
                if os.path.islink(srcname):
                    linkto = os.readlink(srcname)
                    if symlinks:
                        # We can't just leave it to `copy_function` because legacy
                        # code with a custom `copy_function` may rely on copytree
                        # doing the right thing.
                        os.symlink(linkto, dstname)
                    else:
                        # ignore dangling symlink if the flag is on
                        if not os.path.exists(linkto) and ignore_dangling_symlinks:
                            continue
                        # otherwise let the copy occurs. copy2 will raise an error
                        if os.path.isdir(srcname):
                            slcopytree(srcname, dstname, symlinks, ignore,
                                    copy_function)
                        else:
                            copy_function(srcname, dstname)
                elif os.path.isdir(srcname):
                    slcopytree(srcname, dstname, symlinks, ignore, copy_function)
                else:
                    # Will raise a SpecialFileError for unsupported file types
                    copy_function(srcname, dstname)
            # catch the Error from the recursive copytree so that we can
            # continue with other files
            except Error as err:
                errors.extend(err.args[0])
            except OSError as why:
                errors.append((srcname, dstname, str(why)))
    else:
        copy_function(src, dst)

    if errors:
        raise Error(errors)
    return dst

def check_and_copy(src_path, tgt_path):
    if os.path.exists(tgt_path):
        return None

    return slcopytree(src_path, tgt_path)


def remove(srcpath):
    if os.path.isdir(srcpath):
        return shutil.rmtree(srcpath)
    else:
        return os.remove(srcpath)


def preparing_dataset(pathdict, image_set, args):
    start_time = time.time()
    dataset_file = args.dataset_file
    data_static_info = SLConfig.fromfile('util/static_data_path.py')
    static_dict = data_static_info[dataset_file][image_set]

    copyfilelist = []
    for k,tgt_v in pathdict.items():
        if os.path.exists(tgt_v):
            if args.local_rank == 0:
                print("path <{}> exist. remove it!".format(tgt_v))
                remove(tgt_v)
            # continue

        if args.local_rank == 0:
            src_v = static_dict[k]
            assert isinstance(src_v, str)
            if src_v.endswith('.zip'):
                # copy
                cp_tgt_dir = os.path.dirname(tgt_v)
                filename = os.path.basename(src_v)
                cp_tgt_path = os.path.join(cp_tgt_dir, filename)
                print('Copy from <{}> to <{}>.'.format(src_v, cp_tgt_path))
                os.makedirs(cp_tgt_dir, exist_ok=True)
                check_and_copy(src_v, cp_tgt_path)

                # unzip
                import zipfile
                print("Starting unzip <{}>".format(cp_tgt_path))
                with zipfile.ZipFile(cp_tgt_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(cp_tgt_path))

                copyfilelist.append(cp_tgt_path)
                copyfilelist.append(tgt_v)
            else:
                print('Copy from <{}> to <{}>.'.format(src_v, tgt_v))
                os.makedirs(os.path.dirname(tgt_v), exist_ok=True)
                check_and_copy(src_v, tgt_v)
                copyfilelist.append(tgt_v)

    if len(copyfilelist) == 0:
        copyfilelist = None
    args.copyfilelist = copyfilelist

    if args.distributed:
        torch.distributed.barrier()
    total_time = time.time() - start_time
    if copyfilelist:
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Data copy time {}'.format(total_time_str))
    return copyfilelist


