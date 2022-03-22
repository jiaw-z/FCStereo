import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dmb.modeling.stereo.backbones import build_backbone
from dmb.modeling.stereo.disp_samplers import build_disp_sampler
from dmb.modeling.stereo.cost_processors import build_cost_processor
from dmb.modeling.stereo.cmn import build_cmn
from dmb.modeling.stereo.disp_predictors import build_disp_predictor
from dmb.modeling.stereo.disp_refinement import build_disp_refinement
from dmb.modeling.stereo.losses import make_gsm_loss_evaluator
from dmb.modeling.stereo.losses.contrastive_loss import StereoContrastiveLoss
from dmb.modeling.stereo.losses.ssw_loss import StereoWhiteningLoss


class FCStereoBase(nn.Module):
    """
    A general stereo matching model which fits most methods.

    """
    def __init__(self, cfg):
        super(FCStereoBase, self).__init__()
        self.cfg = cfg.copy()
        self.max_disp = cfg.model.max_disp

        self.whitening = False
        if 'whitening' in cfg.model:
            self.whitening = cfg.model.whitening
        print('whitening', self.whitening)

        self.backbone = build_backbone(cfg)

        self.cost_processor = build_cost_processor(cfg)

        # confidence measurement network
        self.cmn = None
        if 'cmn' in cfg.model:
            self.cmn = build_cmn(cfg)

        self.disp_predictor = build_disp_predictor(cfg)

        self.disp_refinement = None
        if 'disp_refinement' in cfg.model:
            self.disp_refinement = build_disp_refinement(cfg)

        # make general stereo matching loss evaluator
        self.loss_evaluator = make_gsm_loss_evaluator(cfg)

        # loss for feature consistency
        self.loss_stereo_scf = StereoContrastiveLoss(dim=32)
        self.loss_stereo_ssw = StereoWhiteningLoss()

    def forward(self, batch, epoch=-1, cov_list=None):
        ref_img, tgt_img = batch['leftImage'], batch['rightImage']
        target_l = batch['leftDisp'] if 'leftDisp' in batch else None
        target_r = batch['rightDisp'] if 'rightDisp' in batch else None

        x_size = ref_img.size() 

        # extract image feature
        left_fms, right_fms = self.backbone(ref_img, tgt_img)

        if self.whitening:
            l_w_arr, r_w_arr = left_fms[-1], right_fms[-1]
            left_fms, right_fms = left_fms[0], right_fms[0]

        if isinstance(left_fms, list):
            ref_fms = left_fms[0]
            tgt_fms = right_fms[0]
        else:
            ref_fms, tgt_fms = left_fms, right_fms        

        # compute cost volume
        costs = self.cost_processor(ref_fms, tgt_fms)

        # disparity prediction
        disps = [self.disp_predictor(cost) for cost in costs]

        # disparity refinement
        if self.disp_refinement is not None:
            disps = self.disp_refinement(disps, ref_fms, tgt_fms, ref_img, tgt_img) 

        if self.training:
            loss_dict = dict()
            variance = None
            if hasattr(self.cfg.model.losses, 'focal_loss'):
                variance = self.cfg.model.losses.focal_loss.get('variance', None)

            if self.cmn is not None:
                # confidence measurement network
                variance, cm_losses = self.cmn(costs, target_l)
                loss_dict.update(cm_losses)

            loss_args = dict(
                variance = variance,
            )

            
            gsm_loss_dict = self.loss_evaluator(disps, costs, target_l, **loss_args)
            
            loss_dict.update(gsm_loss_dict)

            if epoch >= 0:
                contrast_loss = self.loss_stereo_scf(ref_fms, tgt_fms, target_l, target_r, weight=1.0)
                loss_dict.update(contrast_loss)

            if self.whitening and epoch >= 10:
                assert cov_list is not None
                st_isw_loss = self.loss_stereo_ssw(l_w_arr, cov_list=cov_list, weight=10.0)
                loss_dict.update(st_isw_loss)

            return {}, loss_dict

        else:
            results = dict(
                ref_fms=[ref_fms],
                tgt_fms=[tgt_fms],
                disps=disps,
                costs=costs,
            )

            if self.cmn is not None:
                # confidence measurement network
                variance, confs = self.cmn(costs, target_l)
                results.update(confs=confs)

            return results, {}

    @ torch.no_grad()
    def raw_arr(self, batch):
        ref_img = batch['raw_leftImage'].cuda()
        tgt_img = batch['raw_rightImage'].cuda()

        left_fms, right_fms = self.backbone(ref_img, tgt_img)
        r_w_arr_l = left_fms[-1]
        r_w_arr_r = right_fms[-1]
        
        cov_list = self.loss_stereo_ssw.cal_cov([r_w_arr_l, r_w_arr_r])

        return cov_list
