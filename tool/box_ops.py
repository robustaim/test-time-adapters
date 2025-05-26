import torch
from torch import Tensor
from torchvision.ops.boxes import box_area

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1: Tensor, boxes2: Tensor):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

def box_iou_eps(boxes1: Tensor, boxes2: Tensor, eps):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter + eps

    iou = inter / (union)
    return iou, union

def distance_box_iou(boxes1, boxes2, eps=1e-7):
     # calculate iou
     iou, union = box_iou_eps(boxes1, boxes2, eps)
     
     # # The distance between boxes' centers squared
     center1 = (boxes1[:, None, :2] + boxes1[:, None, 2:]) / 2 # [N, 1, 2] <- (cx, cy)
     center2 = (boxes2[None, :, :2] + boxes2[None, :, 2:]) / 2 # [M, 1, 2] <- (cx, cy)
     center_dist = ((center1-center2)**2).sum(-1)
     
     # The diagonal distance
     lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
     rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
     diag = ((rb - lt)**2).sum(-1)
     
     return iou - center_dist / diag

def complete_box_iou(boxes1, boxes2, eps=1e-7):
     diou = distance_box_iou(boxes1, boxes2, eps)
     
     # width and height of boxes
     w_pred = boxes1[:, None, 2] - boxes1[:, None, 0]
     h_pred = boxes1[:, None, 3] - boxes1[:, None, 1]
     w_gt = boxes2[None, :, 2] - boxes2[None, :, 0]
     h_gt = boxes2[None, :, 3] - boxes2[None, :, 1]
     v = (4 / (torch.pi**2)) * torch.pow((torch.atan(w_gt / h_gt) - torch.atan(w_pred / h_pred)), 2)
     with torch.no_grad():
        alpha = v / (1 - diou + v + eps)

     return diou - alpha * v
     