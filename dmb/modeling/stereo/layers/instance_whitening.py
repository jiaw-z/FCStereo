import torch
import torch.nn as nn


class InstanceWhitening(nn.Module):

    def __init__(self, dim):
        super(InstanceWhitening, self).__init__()
        self.instance_standardization = nn.InstanceNorm2d(dim, affine=False)

    def forward(self, x):
        x = self.instance_standardization(x)
        w = x.clone()
        return x, w


def instance_whitening_loss(f_map, eye, mask_matrix, num_remove_cov):
    f_cor, B = get_covariance_matrix(f_map, eye=eye)
    f_cor_masked = f_cor * mask_matrix
    off_diag_sum = torch.sum(torch.abs(f_cor_masked), dim=(1,2), keepdim=True) # B X 1 X 1
    loss = torch.clamp(torch.div(off_diag_sum, num_remove_cov), min=0) # B X 1 X 1
    loss = torch.sum(loss)

    return loss


def get_covariance_matrix(f_map, eye=None):
    eps = 1e-5
    B, C, H, W = f_map.shape  # i-th feature size (B X C X H X W)
    HW = H * W
    if eye is None:
        eye = torch.eye(C).cuda()
    f_map = f_map.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
    f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW-1) + (eps * eye)  # C X C / HW

    return f_cor, B


def make_cov_index_matrix(dim):  # make symmetric matrix for embedding index
    matrix = torch.LongTensor()
    s_index = 0
    for i in range(dim):
        matrix = torch.cat([matrix, torch.arange(s_index, s_index + dim).unsqueeze(0)], dim=0)
        s_index += (dim - (2 + i))
    return matrix.triu(diagonal=1).transpose(0, 1) + matrix.triu(diagonal=1)