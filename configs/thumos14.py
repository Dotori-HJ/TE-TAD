feature_folder = 'data/thumos14/I3D_2stream_act_Pth'
name_format = '{}'
gt_path = 'data/thumos14/th14_annotations_with_fps_duration.json'
meta_info_path = 'data/thumos14/th14_i3d2s_act_ft_info.json'
dataset_name = 'thumos14'
num_classes = 20
binary = False

repeat_trainset = 1

# Evaluation
prime_metric = 'mAP_raw'
nms_mode = ['raw', 'nms']
nms_thr = 0.75
nms_sigma = 0.5
nms_multi_class = True
voting_thresh = -1
min_score = 0.001
duration_thresh = 0.05
extra_cls_path = None
iou_range = [0.3, 0.4, 0.5, 0.6, 0.7]
display_metric_indices = [0, 1, 2, 3, 4]


modelname = 'tetad'

noise_scale = 0.25
seg_noise_scale = 0.0
label_smoothing = 0.0
eval_interval = 5
temperature = 1000
normalize = False
diou = True
hybrid = False
fix_encoder_proposals = True

max_seq_len = 2304
downsample_rate = 1
base_scale = 0.025 # level 8

eval_topk = 900
length_ratio = -1
eval_workers = None

enc_layers = 4
dec_layers = 6
num_cls_head_layers = 3
num_reg_head_layers = 3
num_feature_levels = 8
num_sampling_levels = 1
two_stage = True
mixed_selection = False
emb_norm_type = 'ln'
emb_relu = True
kernel_size = 3
share_class_embed = False
share_segment_embed = False

feature_dim = 2048
hidden_dim = 256

#
set_cost_class = 5
set_cost_seg = 2
set_cost_iou = 2
cls_loss_coef = 2
seg_loss_coef = 2
iou_loss_coef = 2
enc_loss_coef = 1

lr = 1e-4
weight_decay = 0.05
param_dict_type = 'default'
lr_backbone = 1e-05
lr_backbone_names = ['backbone.0']
lr_linear_proj_names = ['sampling_offsets']
lr_linear_proj_mult = 0.1
clip_max_norm = 0.1

epochs = 200
lr_drop = 190
batch_size = 8
save_checkpoint_interval = 100
optimizer = 'adamw'
onecyclelr = False
multi_step_lr = False

use_checkpoint = False

pre_norm = False
dim_feedforward = 2048
dropout = 0.0
droppath = 0.0
n_heads = 8
n_deform_heads = 8
window_size = 128
num_queries = 50
max_queries = 3000
transformer_activation = 'relu'

enc_n_points = 4
dec_n_points = 4
aux_loss = True
focal_alpha = 0.25

# for ema
use_ema = True
ema_decay = 0.999
ema_epoch = 0
