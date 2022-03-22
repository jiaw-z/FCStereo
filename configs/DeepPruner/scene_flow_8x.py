# ---------------------------------------------------------------------------
# DeepPruner: Learning Efficient Stereo Matching via Differentiable PatchMatch
#
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This is a reimplementation of DeepPruner which is written by Shivam Duggal
# Original code: https://github.com/uber-research/DeepPruner
# ---------------------------------------------------------------------------

import os.path as osp

# the task of the model for, including 'stereo' and 'flow', default 'stereo'
task = 'stereo'

# model settings
max_disp = 192
# number of disparity samples to be generated by PatchMatch
patch_match_disparity_sample_number=14
# number of disparity samples to be generated by uniform sampler
uniform_disparity_sample_number=9

model = dict(
    meta_architecture="DeepPruner",
    # max disparity
    max_disp=max_disp,
    # the model whether or not to use BatchNorm
    batch_norm=True,
    backbone=dict(
        type="FastDeepPruner",
        # the in planes of feature extraction backbone
        in_planes=3,
    ),
    disp_sampler=dict(
        type='DeepPruner',
        # the maximum disparity of disparity search range under the resolution of feature
        max_disp=int(max_disp//8),
        # the filter size in propagation of PatchMatch
        propagation_filter_size=3,
        # number of PatchMatch iterations
        iterations=3,
        # to raise the max and lower the other values when using soft-max
        temperature=7,
        # number of disparity samples to be generated by PatchMatch
        patch_match_disparity_sample_number=patch_match_disparity_sample_number,
        # number of disparity samples to be generated by uniform sampler
        uniform_disparity_sample_number=uniform_disparity_sample_number,
    ),
    cost_processor=dict(
        type='DeepPruner',
        # number of disparity samples to be generated by PatchMatch
        patch_match_disparity_sample_number=patch_match_disparity_sample_number,
        # number of disparity samples to be generated by uniform sampler
        uniform_disparity_sample_number=uniform_disparity_sample_number,
        confidence_range_predictor = dict(
            # the in-planes of confidence range predictor, 2 * backbone.out_planes + 1
            in_planes= 2*32+1,
            # the in-planes of hourglass module when cost aggregating
            hourglass_in_planes = 16,
        ),
        cost_aggregator=dict(
            type="DeepPruner",
            # the in planes of cost aggregation sub network,
            # 2 * backbone.out_planes + 1 + 2 * patch_match_disparity_sample_number
            in_planes=2*32+2*14+1,
            # the in-planes of hourglass module when cost aggregating
            hourglass_in_planes = 16,
        ),
    ),
    disp_refinement=dict(
        type='DeepPruner',
        # the in planes list of each disparity refinement sub network
        # backbone.out_planes + uniform_matching_disparity_sample_number + 1
        # backbone_out_planes + 1
        in_planes_list=[64+9+1, 32+1],
        # the number of disparity refinement sub network
        num=2,
    ),
    losses=dict(
        l1_loss=dict(
            # the maximum disparity of disparity search range
            max_disp=max_disp,
            # weights for different scale loss
            weights=(1.6, 1.3, 1.0, 0.7, 0.7),
            # weight for l1 loss with regard to other loss type
            weight=1.0,
        ),
        quantile_loss=dict(
            # the maximum disparity of disparity search range
            max_disp=max_disp,
            # weight for quantile loss with regard to other loss type
            weight=1.0,
            # the balancing scalar
            theta=0.05,
        )
    ),
    eval=dict(
        # evaluate the disparity map within (lower_bound, upper_bound)
        lower_bound=0,
        upper_bound=max_disp,
        # evaluate the disparity map in occlusion area and not occlusion
        eval_occlusion=True,
        # return the cost volume after regularization for visualization
        is_cost_return=False,
        # whether move the cost volume from cuda to cpu
        is_cost_to_cpu=True,
    ),
)

# dataset settings
dataset_type = 'SceneFlow'
# data_root = 'datasets/{}/'.format(dataset_type)
# annfile_root = osp.join(data_root, 'annotations')

# root = '/home/youmin/'
root = '/node01/jobs/io/out/youmin/'

data_root = osp.join(root, 'data/StereoMatching/', dataset_type)
annfile_root = osp.join(root, 'data/annotations/', dataset_type)

# If you don't want to visualize the results, just uncomment the vis data
# For download and usage in debug, please refer to DATA.md and GETTING_STATED.md respectively.
vis_data_root = osp.join(root, 'data/visualization_data/', dataset_type)
vis_annfile_root = osp.join(vis_data_root, 'annotations')


img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
data = dict(
    # whether disparity of datasets is sparse, e.g., SceneFLow is not sparse, but KITTI is sparse
    sparse=False,
    imgs_per_gpu=4,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        annfile=osp.join(annfile_root, 'cleanpass_train.json'),
        input_shape=[256, 512],
        use_right_disp=False,
        **img_norm_cfg,
    ),
    eval=dict(
        type=dataset_type,
        data_root=data_root,
        annfile=osp.join(annfile_root, 'cleanpass_test.json'),
        input_shape=[576, 960],
        use_right_disp=False,
        **img_norm_cfg,
    ),
    # If you don't want to visualize the results, just uncomment the vis data
    vis=dict(
        type=dataset_type,
        data_root=vis_data_root,
        annfile=osp.join(vis_annfile_root, 'vis_test.json'),
        input_shape=[576, 960],
        **img_norm_cfg,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        annfile=osp.join(annfile_root, 'cleanpass_test.json'),
        input_shape=[576, 960],
        use_right_disp=False,
        **img_norm_cfg,
    ),
)

optimizer = dict(type='RMSprop', lr=0.001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    gamma=2/3,
    step=[20, 40, 60, 64],
)
checkpoint_config = dict(
    interval=4
)
# every n epoch evaluate
validate_interval = 4

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ]
)

# https://nvidia.github.io/apex/amp.html
apex = dict(
    # whether to use apex.synced_bn
    synced_bn=True,
    # whether to use apex for mixed precision training
    use_mixed_precision=False,
    # the model weight type: float16 or float32
    type="float16",
    # the factor when apex scales the loss value
    loss_scale=16,
)

total_epochs = 64

gpus = 4
dist_params = dict(backend='nccl')
log_level = 'INFO'
validate = True
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = osp.join(root, 'exps/DeepPruner/scene_flow_8x')

# For test
checkpoint = osp.join(work_dir, 'epoch_64.pth')
out_dir = osp.join(work_dir, 'epoch_64')