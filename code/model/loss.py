import torch
from torch import nn
import utils.general as utils
import math
import torch.nn.functional as F
import cv2
import numpy as np

# copy from MiDaS
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


def convert_number_to_digits(x):
    assert isinstance(x, int), 'the input value {} should be int'.format(x)
    v = 2**x
    # convert to 0-1 digits

    


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)
# end copy
    

class MonoSDFLoss(nn.Module):
    def __init__(self, rgb_loss, 
                 eikonal_weight, 
                 smooth_weight = 0.005,
                 depth_weight = 0.1,
                 normal_l1_weight = 0.05,
                 normal_cos_weight = 0.05,
                 end_step = -1):
        super().__init__()
        self.eikonal_weight = eikonal_weight
        self.smooth_weight = smooth_weight
        self.depth_weight = depth_weight
        self.normal_l1_weight = normal_l1_weight
        self.normal_cos_weight = normal_cos_weight
        self.rgb_loss = utils.get_class(rgb_loss)(reduction='mean')
        
        self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)
        
        # print(f"using weight for loss RGB_1.0 EK_{self.eikonal_weight} SM_{self.smooth_weight} Depth_{self.depth_weight} NormalL1_{self.normal_l1_weight} NormalCos_{self.normal_cos_weight}")
        
        self.step = 0
        self.end_step = end_step

    def get_rgb_loss(self,rgb_values, rgb_gt):
        rgb_gt = rgb_gt.reshape(-1, 3)
        rgb_loss = self.rgb_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_smooth_loss(self,model_outputs):
        # smoothness loss as unisurf
        g1 = model_outputs['grad_theta']
        g2 = model_outputs['grad_theta_nei']
        
        normals_1 = g1 / (g1.norm(2, dim=1).unsqueeze(-1) + 1e-5)
        normals_2 = g2 / (g2.norm(2, dim=1).unsqueeze(-1) + 1e-5)
        smooth_loss =  torch.norm(normals_1 - normals_2, dim=-1).mean()
        return smooth_loss
    
    def get_depth_loss(self, depth_pred, depth_gt, mask):
        # TODO remove hard-coded scaling for depth
        depth_length = depth_pred.shape[0]
        return self.depth_loss(depth_pred.reshape(1, 1, depth_length), (depth_gt * 50 + 0.5).reshape(1, 1, depth_length), mask.reshape(1, 1, depth_length))
        
    def get_normal_loss(self, normal_pred, normal_gt):
        normal_gt = torch.nn.functional.normalize(normal_gt[: , :1024, :], p=2, dim=-1)
        normal_pred = torch.nn.functional.normalize(normal_pred[: , :1024, :], p=2, dim=-1)
        l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1).mean()
        cos = (1. - torch.sum(normal_pred * normal_gt, dim = -1)).mean()
        return l1, cos
        
    def forward(self, model_outputs, ground_truth):
        # import pdb; pdb.set_trace()
        rgb_gt = ground_truth['rgb'].cuda()
        # monocular depth and normal
        depth_gt = ground_truth['depth'].cuda()
        normal_gt = ground_truth['normal'].cuda()
        
        depth_pred = model_outputs['depth_values']
        normal_pred = model_outputs['normal_map'][None]
        
        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt)
        
        if 'grad_theta' in model_outputs:
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        else:
            eikonal_loss = torch.tensor(0.0).cuda().float()


        # only supervised the foreground normal
        mask = ((model_outputs['sdf'] > 0.).any(dim=-1) & (model_outputs['sdf'] < 0.).any(dim=-1))[None, :, None]
        # combine with GT
        # mask = (ground_truth['mask'] > 0.5).cuda() & mask

        depth_loss = self.get_depth_loss(depth_pred, depth_gt, mask) if self.depth_weight > 0 else torch.tensor(0.0).cuda().float()
        if isinstance(depth_loss, float):
            depth_loss = torch.tensor(0.0).cuda().float()    
        
        normal_l1, normal_cos = self.get_normal_loss(normal_pred * mask, normal_gt)
        
        smooth_loss = self.get_smooth_loss(model_outputs)
        
        # compute decay weights 
        if self.end_step > 0:
            decay = math.exp(-self.step / self.end_step * 10.)
        else:
            decay = 1.0
            
        self.step += 1

        loss = rgb_loss + \
               self.eikonal_weight * eikonal_loss +\
               self.smooth_weight * smooth_loss +\
               decay * self.depth_weight * depth_loss +\
               decay * self.normal_l1_weight * normal_l1 +\
               decay * self.normal_cos_weight * normal_cos               
        
        output = {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'smooth_loss': smooth_loss,
            'depth_loss': depth_loss * decay * self.depth_weight,
            'normal_l1': normal_l1 * decay * self.normal_l1_weight,
            'normal_cos': normal_cos * decay * self.normal_cos_weight
        }

        return output


class ClusteringSDFLoss(MonoSDFLoss):
    def __init__(self, rgb_loss, 
                 eikonal_weight,
                 semantic_weight = 0.04,
                 edge_weight = 0.001,
                 smooth_weight = 0.005,
                 semantic_loss = torch.nn.CrossEntropyLoss(ignore_index = -1),
                 depth_weight = 0.1,
                 normal_l1_weight = 0.0,
                 normal_cos_weight = 0.0,
                 reg_vio_weight = 0.1,
                 use_obj_opacity = True,
                 bg_reg_weight = 0.1,
                 end_step = -1):
        super().__init__(
                 rgb_loss = rgb_loss, 
                 eikonal_weight = eikonal_weight, 
                 smooth_weight = smooth_weight,
                 depth_weight = depth_weight,
                 normal_l1_weight = normal_l1_weight,
                 normal_cos_weight = normal_cos_weight,
                 end_step = end_step)
        self.semantic_weight = semantic_weight
        self.edge_weight = edge_weight
        self.bg_reg_weight = bg_reg_weight
        self.semantic_loss = torch.nn.CrossEntropyLoss(ignore_index = -1)
        self.edge_loss = torch.nn.MSELoss(reduction='mean')
        self.sem_smooth_mseloss = torch.nn.MSELoss(reduction='mean')
        self.reg_vio_weight = reg_vio_weight
        self.use_obj_opacity = use_obj_opacity

        self.sobel_x = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).cuda()
        self.sobel_y = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0).cuda()

        print(f"[INFO]: using weight for loss RGB_1.0 EK_{self.eikonal_weight} SM_{self.smooth_weight} Depth_{self.depth_weight} NormalL1_{self.normal_l1_weight} NormalCos_{self.normal_cos_weight}\
            Semantic_{self.semantic_weight}, semantic_loss_type_{self.semantic_loss} Use_object_opacity_{self.use_obj_opacity}")


    def get_edge_loss(self, edge_value, edge_gt):
        edge_gt = edge_gt.squeeze()
        edge_value = edge_value.squeeze()
        edge_loss = self.edge_loss(edge_value, edge_gt)
        return edge_loss

    def gumbel_softmax_sample(self, logits, temperature):
        noise = torch.rand_like(logits).clamp_(1e-10, 1).log_().neg_().clamp_(1e-10, 1).log_().neg_()
        return F.softmax((logits + noise) / temperature, dim=1)

    def extract_max_with_gumbel_softmax(self, logits, temperature=1.0):
        gumbel_probas = self.gumbel_softmax_sample(logits, temperature)
        weighted_logits = gumbel_probas * logits
        extracted_max, _ = torch.max(weighted_logits, dim=1)

        return extracted_max


    # violiation loss
    def get_violation_reg_loss(self, sdf_value):
        # turn to vector, sdf_value: [#rays, #objects]
        min_value, min_indice = torch.min(sdf_value, dim=1, keepdims=True)
        input = -sdf_value-min_value.detach() # add the min value for all tensor
        res = torch.relu(input).sum(dim=1, keepdims=True) - torch.relu(torch.gather(input, 1, min_indice))
        loss = res.sum()
        return loss


    def object_distinct_loss(self, sdf_value, min_sdf):
        _, min_indice = torch.min(sdf_value.squeeze(), dim=1, keepdims=True)
        input = -sdf_value.squeeze() - min_sdf.detach()
        res = torch.relu(input).sum(dim=1, keepdims=True) - torch.relu(torch.gather(input, 1, min_indice))
        loss = res.mean()
        return loss

    def object_opacity_loss(self, predict_opacity, gt_opacity, weight=None):
        # normalize predict_opacity
        # predict_opacity = torch.nn.functional.normalize(predict_opacity, p=1, dim=-1)
        target = torch.nn.functional.one_hot(gt_opacity.squeeze(), num_classes=predict_opacity.shape[1]).float()
        if weight is None:
            loss = F.binary_cross_entropy(predict_opacity.clamp(1e-4, 1-1e-4), target)
        return loss


    def smooth_heaviside(self, x, threshold=1.0, delta=1e-1):
        return torch.sigmoid((x - threshold) / delta)

    def temperature_scaled_softmax(self, logits, dim=1, temperature=0.1):
        scaled_logits = logits / temperature
        probas = F.softmax(scaled_logits, dim=dim)
        return probas

    # background regularization loss following the desing in Sec 3.2 of RICO (https://arxiv.org/pdf/2303.08605.pdf)
    def bg_tv_loss(self, depth_pred, normal_pred, gt_mask):
        # the depth_pred and normal_pred should form a patch in image space, depth_pred: [ray, 1], normal_pred: [ray, 3], gt_mask: [1, ray, 1]
        size = int(math.sqrt(gt_mask.shape[1]))
        mask = gt_mask.reshape(size, size, -1) 
        depth = depth_pred.reshape(size, size, -1)
        normal = torch.nn.functional.normalize(normal_pred, p=2, dim=-1).reshape(size, size, -1)
        loss = 0
        for stride in [1, 2, 4]:
            hd_d = torch.abs(depth[:, :-stride, :] - depth[:, stride:, :])
            wd_d = torch.abs(depth[:-stride, :, :] - depth[stride:, :, :])
            hd_n = torch.abs(normal[:, :-stride, :] - normal[:, stride:, :])
            wd_n = torch.abs(normal[:-stride, :, :] - normal[stride:, :, :])
            loss+= torch.mean(hd_d*mask[:, :-stride, :]) + torch.mean(wd_d*mask[:-stride, :, :])
            loss+= torch.mean(hd_n*mask[:, :-stride, :]) + torch.mean(wd_n*mask[:-stride, :, :])
        return loss


    def clustering_loss(self, pred, gt):
        pred_sm = F.softmax(pred, dim=1)
        gt = gt.squeeze()

        unique_labels = torch.unique(gt)
        reg_loss = torch.tensor(0.0).cuda().float()
        for label in unique_labels:
            mask = (gt == label)
            reg_loss += torch.var(pred_sm[mask], dim=1).mean()
        
        reg_loss = reg_loss / len(unique_labels)
        

        diff_cluster_loss = torch.tensor(0.0).cuda().float()
        for label1 in range(len(unique_labels)):
            for label2 in range(label1 + 1, len(unique_labels)):
                mask1 = (gt == unique_labels[label1])
                mask2 = (gt == unique_labels[label2])
                diff_cluster_loss += torch.norm(pred_sm[mask1].mean(dim=0) - pred_sm[mask2].mean(dim=0))

        if len(unique_labels) <= 1:
            diff_cluster_loss = torch.tensor(0.0).cuda().float()
        else:
            diff_cluster_loss = diff_cluster_loss / (len(unique_labels) * (len(unique_labels) - 1))

        return reg_loss, diff_cluster_loss


    def contrastive_loss(self, pred, gt, temperature=1.0, if_softmax=False):
        if if_softmax:
            pred = F.softmax(pred / 0.1, dim=1)
        gt = gt.squeeze()
        bsize = pred.size(0)
        masks = gt.view(-1, 1).repeat(1, bsize).eq_(gt.clone())
        masks = masks.fill_diagonal_(0, wrap=False)
        
        distance_sq = torch.pow(pred.unsqueeze(1) - pred.unsqueeze(0), 2).sum(dim=-1)
        temperature = torch.ones_like(distance_sq) * temperature
        # temperature = torch.where(masks==1, temperature, torch.ones_like(temperature))

        similarity_kernel = torch.exp(-distance_sq/temperature)
        logits = torch.exp(similarity_kernel)

        p = torch.mul(logits, torch.logical_not(masks)).sum(dim=-1)
        p = torch.mul(logits, masks).sum(dim=-1)
        
        Z = logits.sum(dim=-1)

        prob = torch.div(p, Z)
        prob = torch.div(1, p)
        prob_masked = torch.masked_select(prob, prob.ne(0))
        loss = -prob_masked.log().sum()/bsize

        return loss


    def cal_onehot_loss(self, pred, gt):
        non_bg_mask = (gt != 0)
        pred_softmax = F.softmax(pred[non_bg_mask], dim=1)
        max_indices = torch.argmax(pred_softmax, dim=1, keepdim=True)
        one_hot_targets = torch.zeros_like(pred[non_bg_mask]).scatter_(1, max_indices, 1)
        onehot_loss = F.mse_loss(pred[non_bg_mask], one_hot_targets)
        return onehot_loss

        

    def cal_bg_loss(self, pred, gt, non_bg_weight=0.3, if_sem=False):
        gt = gt.squeeze()

        bg_mask = (gt == 0)
        if if_sem:
            bg_loss = torch.tensor(0.0).cuda().float()
            non_bg_loss = torch.mean(torch.abs(pred[:,0]).cuda()) * non_bg_weight
        elif bg_mask.sum() > 0:
            # where the gt is background, the pred should be [1, 0, 0, 0, ...]
            bg_loss = torch.mean(torch.abs(pred[bg_mask][:,0] - torch.tensor([1.0]).cuda()))
            # where the gt is not background, the pred should not be [1, 0, 0, 0, ...]
            non_bg_loss = torch.mean(torch.abs(pred[~bg_mask][:,0]).cuda()) * non_bg_weight
        else:
            bg_loss, non_bg_loss = torch.tensor(0.0).cuda().float(), torch.tensor(0.0).cuda().float()

        return bg_loss, non_bg_loss

    def cal_cross_view_loss(self, pred, pred_multi_view, gt_semantic, extra_semantic_gt):
        # if pixel a and pixel b has different semantic label, then the object opacity should be different
        pred_sm = F.softmax(pred, dim=1)
        pred_multi_view_sm = F.softmax(pred_multi_view, dim=1)

        assert extra_semantic_gt.shape[0] == 4 and extra_semantic_gt.shape[1] == 64
        split_pred_multi_view = torch.split(pred_multi_view_sm, 64, dim=0)

        cross_view_loss = torch.tensor(0.0).cuda().float()
        unique_labels = torch.unique(gt_semantic)
        for label in unique_labels:
            mask = (gt_semantic == label)
            center_cur_view = pred_sm[mask].mean(dim=0)
            for i in range(4):
                for label_multi_view in torch.unique(extra_semantic_gt[i]):
                    if label_multi_view != label:
                        mask_multi_view = (extra_semantic_gt[i] == label_multi_view)
                        center_multi_view = split_pred_multi_view[i][mask_multi_view].mean(dim=0)
                        cross_view_loss += torch.norm(center_cur_view - center_multi_view)

        return torch.exp(-cross_view_loss)


    def sem_loss(self, pred, gt, confidence_gt=None):
        gt = gt.squeeze()

        if confidence_gt is not None:
            confidence_gt = confidence_gt.squeeze()
            bg_area = (gt == 0)
            gt[bg_area] = 21 # set as other properties
            confidence_gt[bg_area] = 0.1

        sem_loss = torch.tensor(0.0).cuda().float()
        unique_labels = torch.unique(gt)
        for label in unique_labels:
            if label == 0:
                continue
            mask = (gt == label)
            if confidence_gt is not None:
                avg_confidence = confidence_gt[mask].mean()
                sem_loss += torch.norm(pred[mask].mean(dim=0) - torch.nn.functional.one_hot(label, num_classes=pred.shape[1]).float().cuda()) * avg_confidence
            else:
                sem_loss += torch.norm(pred[mask].mean(dim=0) - torch.nn.functional.one_hot(label, num_classes=pred.shape[1]).float().cuda())

        sem_loss = sem_loss / len(unique_labels)
        return sem_loss
    

    def forward(self, model_outputs, ground_truth, call_reg=False, clustering_weight=0.0, \
        if_bg_loss=False, if_bg_weight=0.0, non_bg_weight=0.3, if_sem=False, sem_weight=0.15, \
        onehot_weight=0.0, if_onehot=False, sample_sdf_weight=1.0, \
        reg_weight=0.0, diff_cluster_weight=0.0, if_multi_camera_view=False, \
        cross_view_weight=0.0):
        output = super().forward(model_outputs, ground_truth)
        
        semantic_gt = ground_truth['segs'].cuda().long().squeeze()
        reg_loss, diff_cluster_loss = self.clustering_loss(model_outputs['object_opacity'], semantic_gt)
        clustering_loss = reg_loss * reg_weight - diff_cluster_loss * diff_cluster_weight
            
        if if_sem:
            confidence_gt = ground_truth['confidence'].cuda().float()
            sem_loss = self.sem_loss(model_outputs['object_opacity'], semantic_gt, confidence_gt) 
        else:
            sem_loss = torch.tensor(0.0).cuda().float()
        
        if if_onehot:
            semantic_gt = ground_truth['segs'].cuda().long().squeeze()
            onehot_loss = self.cal_onehot_loss(model_outputs['object_opacity'], semantic_gt)
        else:
            onehot_loss = torch.tensor(0.0).cuda().float()

        if if_multi_camera_view:
            semantic_gt = ground_truth['segs_sem'].cuda().long().squeeze()
            extra_semantic_gt = ground_truth['extra_segs_sem'].cuda().long().squeeze()
            cross_view_loss = self.cal_cross_view_loss(model_outputs['object_opacity'], model_outputs['object_opacity_extra'], semantic_gt, extra_semantic_gt)
        else:
            cross_view_loss = torch.tensor(0.0).cuda().float()
            

        if if_bg_loss:
            semantic_gt = ground_truth['segs'].cuda().long().squeeze()
            bg_loss, non_bg_loss = self.cal_bg_loss(model_outputs['object_opacity'], semantic_gt, non_bg_weight=non_bg_weight, if_sem=if_sem)
        else:
            bg_loss, non_bg_loss = torch.tensor(0.0).cuda().float(), torch.tensor(0.0).cuda().float()
        
        
        if "sample_sdf" in model_outputs and call_reg:
            sample_sdf_loss = self.object_distinct_loss(model_outputs["sample_sdf"], model_outputs["sample_minsdf"]) * sample_sdf_weight
        else:
            sample_sdf_loss = torch.tensor(0.0).cuda().float()
            

        output['collision_reg_loss'] = sample_sdf_loss * self.reg_vio_weight
        output['sem_loss'] = sem_loss * sem_weight
        output['onehot_loss'] = onehot_loss * onehot_weight
        output['reg_loss'] = reg_loss * reg_weight
        output['diff_val_loss'] = -diff_cluster_loss * diff_cluster_weight
        output['bg_loss'] = bg_loss * if_bg_weight
        output['non_bg_loss'] = non_bg_loss * if_bg_weight
        output['cross_view_loss'] = cross_view_loss * cross_view_weight
        output['loss'] = output['loss'] + clustering_weight * clustering_loss + self.reg_vio_weight* sample_sdf_loss + \
            onehot_loss * onehot_weight + (bg_loss + non_bg_loss) * if_bg_weight + cross_view_loss * cross_view_weight + sem_loss * sem_weight
        return output