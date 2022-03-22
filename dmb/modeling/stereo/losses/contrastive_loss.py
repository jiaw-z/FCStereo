import torch
import torch.nn as nn
import torch.nn.functional as F

class StereoContrastiveLoss(nn.Module):
    def __init__(self, dim, K=6000, n_neg=60, T=0.07):
        super(StereoContrastiveLoss, self).__init__()
        self.n_neg = n_neg
        self.T = T
        self.criterion = nn.CrossEntropyLoss()
        self.dim = dim
        self.K = K
        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.projection_q.parameters(), self.projection_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        keys = keys.view(-1, self.dim)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, ref_fms, tgt_fms, target_l, target_r, weight=1):
        low_scale = target_l.shape[-1] / ref_fms.shape[-1]
        disps_lowr = F.interpolate(target_l, scale_factor=1 / low_scale) / low_scale
        disps_lowr_right = F.interpolate(target_r, scale_factor=1 / low_scale) / low_scale
        mask_cont = occ_mask(disps_lowr, disps_lowr_right).squeeze()
        maskl = (disps_lowr > 0.0).float().squeeze()
        b, _, h, w = disps_lowr.size()
        x_base = torch.linspace(0, 1, w).repeat(b, 1, h, 1).type_as(disps_lowr)
        x_shifts = -disps_lowr[:, :, :, :] / w
        maskout = ((x_base + x_shifts) >= 0).squeeze()

        maskl *= maskout
        mask_cont *= maskl
        mask_cont = mask_cont.detach().bool()

        query = F.normalize(ref_fms, dim=1)

        positive = positive_sampler(ref_fms, tgt_fms, disps_lowr, disps_lowr_right)
        positive = F.normalize(positive, dim=1)
        l_pos = torch.einsum('nchw,nchw->nhw', [query, positive]).reshape(b, h, w, 1)   

        negative = negative_sampler(ref_fms, tgt_fms, disps_lowr, disps_lowr_right, self.n_neg, low=1, high=25)
        negative = F.normalize(negative, dim=1)
        l_neg = torch.einsum('nchw,nchwe->nhwe', [query, negative]).reshape(b, h, w, self.n_neg)

        self.queue = F.normalize(self.queue, dim=0)
        l_neg_queue = torch.einsum('nchw,ck->nhwk', [query, self.queue.clone().detach()])
        l_neg = torch.cat([l_neg, l_neg_queue], dim=3)

        logits = torch.cat([l_pos, l_neg], dim=3).permute(3, 0, 1, 2)
        logits *= mask_cont
        logits = logits.permute(1, 2, 3, 0).reshape(b * h * w, 1 + self.n_neg + self.K)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()


        contrast_loss = dict()
        contrast_loss['loss_contrast'] = self.criterion(logits, labels) * weight

        sampler_h = torch.randint(low=0, high=h, size =(1,))
        sampler_w = torch.randint(low=0, high=w, size =(1,))
        sampler_neg = torch.randint(low=0, high=1, size =(1,))
        key = negative.permute(0, 2, 3, 4, 1)
        self._dequeue_and_enqueue(key[:, sampler_h, sampler_w, sampler_neg, :,].squeeze())

        return contrast_loss




# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output



def occ_mask(left_disp, right_disp):
    b, _, h, w = left_disp.size()
    device = left_disp.device
    index = torch.arange(w).float().to(device)
    index = index.repeat(b, 1, h, 1)
    index_l2r = warp(index, right_disp)
    index_l2r2l = warp(index_l2r, -left_disp)

    masko = torch.abs(index - index_l2r2l) < 3.

    return masko.float()


def positive_sampler(left_feat, right_feat, left_disp, right_disp=None):
    b, _, h, w = left_disp.size()
    x_base = torch.linspace(0, 1, w).repeat(b, h, 1).type_as(left_disp)
    y_base = torch.linspace(0, 1, h).repeat(b, w, 1).transpose(1, 2).type_as(left_disp)

    x_shifts = -left_disp[:, 0, :, :] / w
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)

    output = F.grid_sample(right_feat, 2 * flow_field - 1, mode='bilinear', padding_mode='border')

    return output



def negative_sampler(left_feat, right_feat, left_disp, right_disp, n_neg=10, low=1, high=25):
    b, c, h, w = left_feat.size()
    x_base = torch.linspace(0, 1, w).repeat(b, h, 1).type_as(left_feat)
    y_base = torch.linspace(0, 1, h).repeat(b, w, 1).transpose(1, 2).type_as(left_feat)

    if left_disp != None:
        x_shifts = -left_disp[:, 0, :, :] / w
        x_right = (x_base + x_shifts) * w
    else:
        x_right = x_base * w
    y_right = y_base * h

    x_right = x_right.unsqueeze(-1).repeat(1, 1, 1, n_neg)
    y_right = y_right.unsqueeze(-1).repeat(1, 1, 1, n_neg)

    halfn = int(n_neg / 2)
    # Generate random shift for each KeyPoint
    x_random_shift_1 = torch.randint_like(x_right[:, :, :, :halfn], low=int(low), high=high)
    x_random_shift_1 *= torch.sign(torch.rand_like(x_random_shift_1, dtype=torch.float)-0.5).short()   # Random + or - shift
    y_random_shift_1 = torch.randint_like(y_right[:, :, :, :halfn], low=int(low), high=high)
    y_random_shift_1 *= torch.sign(torch.rand_like(y_random_shift_1, dtype=torch.float)-0.5).short()  # Random + or - shift

    x_random_shift_2 = torch.randint_like(x_right[:, :, :, halfn:], low=int(low), high=high)
    x_random_shift_2 *= torch.sign(torch.rand_like(x_random_shift_2, dtype=torch.float)-0.5).short()   # Random + or - shift
    y_random_shift_2 = torch.randint_like(y_right[:, :, :, halfn:], low=int(low), high=high)
    y_random_shift_2 *= torch.sign(torch.rand_like(y_random_shift_2, dtype=torch.float)-0.5).short()  # Random + or - shift

    x_right_shifted = x_right + torch.cat((x_random_shift_1, y_random_shift_1), dim=-1)
    y_right_shifted = y_right + torch.cat((x_random_shift_2, y_random_shift_2), dim=-1)
    flow_field = torch.stack((x_right_shifted, y_right_shifted), dim=-1)

    flow_field %= torch.tensor((w, h), dtype=torch.short, device=flow_field.device)
    flow_field[:, :, :, :, 0] = flow_field[:, :, :, :, 0] / w
    flow_field[:, :, :, :, 1] = flow_field[:, :, :, :, 0] / h

    output = extract_feature_field(right_feat, flow_field)

    return output


def extract_feature_field(right_feat, flow_field, rand_batch=False):
    b, h, w, n, _ = flow_field.size()
    c = right_feat.size(1)
    flow_field = flow_field.permute(3, 0, 1, 2, 4).reshape(n*b, h, w, 2)
    right_feat = right_feat.unsqueeze(-1).repeat(1, 1, 1, 1, n)
    right_feat = right_feat.permute(4, 0, 1, 2, 3).reshape(n*b, c, h, w)
    output = F.grid_sample(right_feat, 2 * flow_field - 1, mode='bilinear', padding_mode='border')
    output = output.reshape(n, b, c, h, w).permute(1, 2, 3, 4, 0)

    return output



def warp(img, disp):
    '''
    Borrowed from: https://github.com/OniroAI/MonoDepth-PyTorch
    '''
    b, _, h, w = img.size()
    device = disp.device
    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, w).repeat(b, h, 1).type_as(img).to(device)
    y_base = torch.linspace(0, 1, h).repeat(b, w, 1).transpose(1, 2).type_as(img).to(device)

    # Apply shift in X direction
    x_shifts = (disp[:, 0, :, :] / w).to(device)
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)

    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='border')

    return output