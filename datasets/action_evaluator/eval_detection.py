import json
import logging
import pdb
import sys
import traceback
import urllib.error
import urllib.parse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .utils import get_blocked_videos, interpolated_prec_rec, segment_iou

logger_initilized = False


def setup_logger(log_file_path, name=None, level=logging.INFO):
    """
    Setup a logger that simultaneously output to a file and stdout
    ARGS
        log_file_path: string, path to the logging file
    """
    # logging settings
    #   log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    log_formatter = logging.Formatter(
            "[%(asctime)s][%(levelname)s] %(pathname)s: %(lineno)4d: %(message)s",
            datefmt="%m/%d %H:%M:%S")
    root_logger = logging.getLogger(name)

    if name:
        root_logger.propagate = False
    root_logger.setLevel(level)
    # file handler
    if log_file_path is not None:
        log_file_handler = logging.FileHandler(log_file_path)
        log_file_handler.setFormatter(log_formatter)
        root_logger.addHandler(log_file_handler)

    log_formatter = logging.Formatter(
            "[%(asctime)s][%(levelname)s]: %(message)s",
            datefmt="%m/%d %H:%M:%S")
    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    # log_stream_handler.setLevel(logging.INFO)
    root_logger.addHandler(log_stream_handler)

    logging.info('Log file is %s' % log_file_path)
    global logger_initilized
    logger_initilized = True
    return root_logger


def get_classes(anno_dict):
    if 'classes' in anno_dict:
        classes = anno_dict['classes']
    else:
        database = anno_dict['database']
        all_gts = []
        for vid in database:
            all_gts += database[vid]['annotations']
        classes = list(sorted({x['label'] for x in all_gts}))
    return classes


class ANETdetection(object):

    GROUND_TRUTH_FIELDS = ['database', 'taxonomy', 'version']
    PREDICTION_FIELDS = ['results', 'version', 'external_data']

    def __init__(self, ground_truth_filename=None, prediction_filename=None,
                 ground_truth_fields=GROUND_TRUTH_FIELDS,
                 prediction_fields=PREDICTION_FIELDS,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10),
                 subset='validation', verbose=False,
                 check_status=False, log_path=None, exclude_videos=None):

        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.subset = subset
        # if log_path is None:
        if not logger_initilized:
            print('setup logger')
            logger = setup_logger(log_path)
        else:
            logger = logging.getLogger()
        self.logger = logger

        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.gt_fields = ground_truth_fields
        self.pred_fields = prediction_fields
        self.ap = None
        self.check_status = check_status

        self.blocked_videos = exclude_videos if exclude_videos else list()
        # self.blocked_videos = ['video_test_0000270', 'video_test_0001292', 'video_test_0001496']
        # Import ground truth and predictions.
        self.ground_truth, self.activity_index = self._import_ground_truth(
            ground_truth_filename)
        self.prediction = self._import_prediction(prediction_filename)

        if self.verbose:
            print('[INIT] Loaded annotations from {} subset.'.format(subset))
            nr_gt = len(self.ground_truth)
            print('\tNumber of ground truth instances: {}'.format(nr_gt))
            nr_pred = len(self.prediction)
            print('\tNumber of predictions: {}'.format(nr_pred))
            print('\tFixed threshold for tiou score: {}'.format(self.tiou_thresholds))

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        if isinstance(ground_truth_filename, str):
            with open(ground_truth_filename, 'r') as fobj:
                data = json.load(fobj)
        else:
            data = ground_truth_filename
        # # Checking format
        # if not all([field in list(data.keys()) for field in self.gt_fields]):
        #     raise IOError('Please input a valid ground truth file.')

        # Read ground truth data.
        # activity_index, cidx = {}, 0

        # class_list = get_classes(data)
        # class_list = set()
        class_list = []
        activity_index = {cls_name: idx for idx, cls_name in enumerate(class_list)}
        video_lst, t_start_lst, t_end_lst, label_lst, difficult_lst = [], [], [], [], []
        for videoid, v in data['database'].items():
            if self.subset != v['subset']:
                continue
            if videoid in self.blocked_videos:
                continue
            for ann in v['annotations']:
                if ann['label'] not in class_list:
                    activity_index[ann['label']] = len(class_list)
                    class_list.append(ann['label'])
                video_lst.append(videoid)
                t_start_lst.append(float(ann['segment'][0]))
                t_end_lst.append(float(ann['segment'][1]))
                label_lst.append(activity_index[ann['label']])
                difficult = 0 if 'difficult' not in ann else ann['difficult']
                difficult_lst.append(difficult)

        ground_truth = pd.DataFrame({'video-id': video_lst,
                                     't-start': t_start_lst,
                                     't-end': t_end_lst,
                                     'label': label_lst,
                                     'difficult': difficult_lst})
        # self.class_list = [x for x in class_list]

        return ground_truth, activity_index

    def _import_prediction(self, prediction_filename):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        if isinstance(prediction_filename, str):
            with open(prediction_filename, 'r') as fobj:
                data = json.load(fobj)
        else:
            data = prediction_filename
        # Checking format...
        if not all([field in list(data.keys()) for field in self.pred_fields]):
            raise IOError('Please input a valid prediction file.')

        # Read predicitons.
        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst = [], []
        for videoid, v in data['results'].items():
            if videoid in self.blocked_videos:
                continue
            for result in v:
                label = self.activity_index[result['label']]
                video_lst.append(videoid)
                t_start_lst.append(result['segment'][0])
                t_end_lst.append(result['segment'][1])
                label_lst.append(label)
                score_lst.append(result['score'])
        prediction = pd.DataFrame({'video-id': video_lst,
                                   't-start': t_start_lst,
                                   't-end': t_end_lst,
                                   'label': label_lst,
                                   'score': score_lst})
        return prediction

    # def wrapper_compute_average_precision(self):
    #     """Computes average precision for each class in the subset.
    #     """
    #     ap = np.zeros((len(self.tiou_thresholds), len(list(self.activity_index.items()))))
    #     for activity, cidx in self.activity_index.items():
    #         gt_idx = self.ground_truth['label'] == cidx
    #         pred_idx = self.prediction['label'] == cidx
    #         ap[:,cidx] = compute_average_precision_detection(
    #             self.ground_truth.loc[gt_idx].reset_index(drop=True),
    #             self.prediction.loc[pred_idx].reset_index(drop=True),
    #             tiou_thresholds=self.tiou_thresholds)
    #     return ap

    ################################# copied from GTAD #######################################
    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label.
        """
        try:
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        except:
            if self.verbose:
                print('Warning: No predictions of label \'%s\' were provdied.' % label_name)
            return pd.DataFrame()

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = self.prediction.groupby('label')

        results = Parallel(n_jobs=len(self.activity_index))(
                    delayed(compute_average_precision_detection)(
                        ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                        prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
                        tiou_thresholds=self.tiou_thresholds,
                    ) for label_name, cidx in self.activity_index.items())

        for i, cidx in enumerate(self.activity_index.values()):
            ap[:,cidx] = results[i]

        return ap
    #################################################################################

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        self.ap = self.wrapper_compute_average_precision()
        self.mAP = self.ap.mean(axis=1)
        if self.verbose:
            print('[RESULTS] Performance on ActivityNet detection task.')
            print('\n{}'.format(' '.join(['%.4f' % (x * 1) for x in self.mAP])))
            print('\tAverage-mAP: {}'.format(self.mAP.mean()))

def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """

    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap
    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')
    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():
        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            # print(e)
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        # matched_to_difficult = False
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, int(this_gt.loc[jdx]['index'])] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, int(this_gt.loc[jdx]['index'])] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float64)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float64)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])
        # # Computing prec-rec
        # this_tp = np.cumsum(tp[tidx,:]).astype(np.float64)
        # this_fp = np.cumsum(fp[tidx,:]).astype(np.float64)
        # rec = this_tp / npos
        # prec = this_tp / (this_tp + this_fp)
        # ap[tidx] = interpolated_prec_rec(prec, rec)

    return ap
