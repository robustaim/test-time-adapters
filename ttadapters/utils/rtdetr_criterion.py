import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision
import torch.distributed as tdist


# from torchvision.ops import box_convert, generalized_box_iou
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from torchvision.ops import complete_box_iou_loss as ciou
# from src.core import register


class SetCriterion(nn.Module):
    """ Computing the loss for DETR.
        Perform Hungarian matching between output_logits and ground truth boxes
        Supervise each matched ground truth-prediction pair
    """
    def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0, eos_coef=1e-4, num_classes=80):
        """ Create the criterion.
        Parameters:
            matcher(nn.Module) : Module for Hungarian matching between model outputs and ground truth.
            num_classes(int): Number of object classes excluding the no-object class.
            weight_dict(dict) : key = loss name, value = weight of the corresponding loss.
            losses(list) : Typically, ['vfl', 'boxes'] are used.
            ['cross_entropy_loss', 'class_error', 'binary_cross_entropy_loss', 'focal_loss', 'varifocal_loss', 'cardinality_matric', 'bbox_loss', 'ciou_loss']
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses 

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef 
        self.register_buffer('empty_weight', empty_weight)

        self.alpha = alpha
        self.gamma = gamma
        
    def pred_indices_aligned(self, indices):
        # Create indices to select matched predictions from each batch element
        batch_idx = torch.cat([torch.full_like(pred, i) for i, (pred, _) in enumerate(indices)])
        pred_idx = torch.cat([pred for (pred, _) in indices])
        return batch_idx, pred_idx

    def assign_matched_gt_classes(self, pred_logits, outputs, targets, indices, num_boxes):
        B, Q, _ = pred_logits.shape
        
        idx = self.pred_indices_aligned(indices)
        
        matched_target_classes = torch.cat([t["labels"][tgt] for t, (_, tgt) in zip(targets, indices)])
        
        all_target_classes = torch.full((B, Q), fill_value=self.num_classes, dtype=torch.int64, device=pred_logits.device)
        all_target_classes[idx] = matched_target_classes
        
        return all_target_classes
    
    def assign_matched_get_boxes(self, pred_logits, outputs, targets, indices):
        idx = self.pred_indices_aligned(indices)
        
        pred_boxes = outputs[idx]
        matched_target_boxes = torch.cat([t["boxes"][tgt] for t, (_, tgt) in zip(targets, indices)], dim=0)
        
        return pred_boxes, matched_target_boxes
    
    def class_loss_CE(self, outputs, targets, indices, num_boxes, log=True):
        pred_logits = outputs['pred_logits']
        target_classes = self.assign_matched_gt_classes(pred_logits, outputs, targets, indices, num_boxes)

        cross_entropy_loss = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'cross_entropy_loss' : cross_entropy_loss}
        
        # if log:
        #     losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        # return losses
    
    def class_loss_BCE(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        target_classes = self.assign_matched_gt_classes(pred_logits, outputs, targets, indices, numboxes)
        
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        binary_cross_entropy_loss = binary_cross_entropy_with_logits(pred_logits, target * 1., reduction='none')
        binary_cross_entropy_loss = binary_cross_entropy_loss.mean(1).sum() * pred_logits.shape[1] / num_boxes
        return {'binary_cross_entropy_loss' : binary_cross_entropy_loss}
    
    def class_loss_Focal(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        target_classes = self.assign_matched_gt_classes(pred_logits, outputs, targets, indices, numboxes)
        
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        focal_loss = torchvision.ops.sigmoid_focal_loss(pred_logits, target, self.alpha, self.gamma, reduction='none')
        focal_loss = focal_loss.mean(1).sum * pred_logits.shape[1] / num_boxes
        return {'focal_loss' : focal_loss}
    
    def class_loss_Varifocal(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        
        pred_boxes, target_boxes = self.assign_matched_get_boxes(pred_logits, outputs, targets, indices)
        ious, _ = box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()
        
        target_classes = self.assign_matched_gt_classes(pred_logits, outputs, targets, indices, numboxes)
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        
        soft_label_map = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        soft_label_map[idx] = ious.to(soft_label_map.dtype)
        soft_targets = soft_label_map.unsqueeze(-1) * target
        
        pred_score = F.sigmoid(pred_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + soft_targets
        
        varifocal_loss = F.binary_cross_entropy_with_logits(pred_logits, soft_targets, weight=weight, reduction='none')
        varifocal_loss = varifocal_loss.mean(1).sum() * pred_logits.shape[1] / num_boxes
        return {'varifocal_loss' : varifocal_loss}
    
    @torch.no_grad()
    def cardinality_matric():
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        target_class = torch.as_tensor([len(tgt['labels']) for tgt in targets], device=device)
        card_pred_object = (pred_logits.argmax(-1) != pred_logits.shape[-1] -1).sum(1)
        card_err = F.l1_loss(card_pred_object.float(), tgt_lengths.float())
        losses = {'cardinality_matric' : card_err}
        return losses 
        
    def boxes_loss(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        
        pred_boxes, target_boxes = self.assign_matched_get_boxes(pred_logits, outputs, targets, indices)
        
        losses = {}
        
        bbox_loss = F.1l_loss(pred_boxes, target_boxes, reduction='none')
        losses['bbox_loss'] = bbox_loss.sum() / num_boxes
        
        ciou_loss 1- ciou(box_cxcywh_to_xyxy(pred_boxes),box_cxcywh_to_xyxy(target_boxes))
        losses['ciou_loss'] = ciou_loss.sum() / num_boxes
        return losses
    
        
    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs(torch.Tensor of dict): Dictionary of outputs from the model.
             targets(list of dicts): len(targets) == batch_size.
        """
        # Only keep 'pred_logits' and 'pred_boxes' for Hungarian matching,
        # as these are the only fields used during the matching process.
        outputs_for_matching = {k : v for k, v in outputs.items() if 'pred' in k}

        # Matching between model outputs and targets.
        indices = self.matcher(outputs_for_matching, targets)

        # In distributed training, sum ground-truth box counts across all nodes to compute the average number of boxes for loss normalization
        num_boxes = sum(len(t["labels"]) for t in targets) # Total number of ground-truth objects in the current batch.
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())))
        if tdist.is_available() and tdist.is_initialized():
            tdist.all_reduce(num_boxes)
            world_size = dist.get_world_size()
        else :
            world_size = 1
        num_boxes = torch.clamp(num_boxes / world_size, min=1).item()

        
        # Compute the requested losses
        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        continue
                    
                    # Disable logging for label loss on intermediate layers
                    log_flag = False if loss_name == 'labels' else True
                    kwargs = {'log': log_flag} if loss_name == 'labels' else {}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of cdn auxiliary losses. For rtdetr
        if 'dn_aux_outputs' in outputs:
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    if loss == 'masks':
                        continue
                    
                    # Logging is enabled only for the last layer.
                    log_flag = False if loss_name == 'labels' else True
                    kwargs = {'log': log_flag} if loss_name == 'labels' else {}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def get_cdn_matched_indices(dn_meta, targets):
        '''get_cdn_matched_indices
        '''
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets] # Number of object categories per image.
        device = targets[0]['labels'].device
        
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                # [0, 1, ..., num_gt-1]
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                # If there are no GTs, append an empty tuple
                empty = torch.zeros(0, dtype=torch.int64, device=device)
                dn_match_indices.append((empty, empty))
        
        return dn_match_indices





# @torch.no_grad()
# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     if target.numel() == 0:
#         return [torch.zeros([], device=output.device)]
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res



