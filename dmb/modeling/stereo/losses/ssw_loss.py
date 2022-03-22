import torch
import torch.nn as nn
import torch.nn.functional as F
from dmb.modeling.stereo.layers.instance_whitening import instance_whitening_loss
import kmeans1d

class StereoWhiteningLoss(object):
    def __init__(self):
        dim = 1
        self.dim = 1
        self.eps = 1e-5
        self.eye = torch.eye(dim, dim).cuda()

        # print(torch.ones(16, 16).triu(diagonal=1))
        self.reversal_eye = torch.ones(dim, dim).triu(diagonal=1).cuda()
        self.relax_denom = 1.5
        self.clusters = 3
        self.num_off_diagonal = 0
        self.margin = 0


    def __call__(self, l_w_arr, cov_list, weight=0.6):
        wt_loss  = 0
        for idx in range(len(l_w_arr)):
            feats_l = l_w_arr[idx]

            B, c, h, w = feats_l.size()
            self.dim = c

            self.eye = torch.eye(c, c).cuda()
            self.reversal_eye = torch.ones(c, c).triu(diagonal=1).cuda()

            cov_matrix = cov_list[idx]

            var_flatten = cov_matrix.flatten()
            clusters, centroids = kmeans1d.cluster(var_flatten, self.clusters)
            num_sensitive = var_flatten.size()[0] - clusters.count(0)- clusters.count(1)
            values, indices = torch.topk(var_flatten, k=int(num_sensitive))

            mask_matrix = torch.zeros(B, self.dim, self.dim).cuda()
            mask_matrix = mask_matrix.view(B, -1)
            for midx in range(B):
                mask_matrix[midx][indices] = 1
            mask_matrix = mask_matrix.view(B, self.dim, self.dim)
            mask_matrix = mask_matrix * self.reversal_eye
            num_sensitive_sum = torch.sum(mask_matrix)
            f_map = l_w_arr[idx]
            loss = instance_whitening_loss(f_map, self.eye, mask_matrix, num_remove_cov=num_sensitive_sum)
            wt_loss += loss
        
        wt_loss = wt_loss / len(l_w_arr)
        st_isw_loss = dict()
        st_isw_loss['loss_stereo_isw'] = weight * wt_loss

        return st_isw_loss
    
    def cal_cov(self, raw_w_arr):
        cov_list = []
        l_arr_mask = raw_w_arr[0]
        r_arr_mask = raw_w_arr[1]
        for idx in range(len(l_arr_mask)):
            mask_feats_l = l_arr_mask[idx]
            mask_feats_r = r_arr_mask[idx]
            b, c, h, w = mask_feats_l.size()
            self.dim = c
            self.eye = torch.eye(c, c).cuda()
            self.reversal_eye = torch.ones(c, c).triu(diagonal=1).cuda()
            f_map = torch.cat([mask_feats_l.unsqueeze(0), mask_feats_r.unsqueeze(0)], dim=0)
            V, B, C, H, W = f_map.shape
            HW = H * W
            f_map = f_map.contiguous().view(V*B, C, -1)
            f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW - 1) + (self.eps * self.eye)  # VB X C X C / HW
            off_diag_elements = f_cor
            off_diag_elements = off_diag_elements.view(V, B, C, -1)
            f_cor = f_cor.view(V, B, C, -1)
            assert V == 2
            variance_of_covariance = torch.var(off_diag_elements, dim=0)

            cov_list.append(variance_of_covariance)
        
        return cov_list
