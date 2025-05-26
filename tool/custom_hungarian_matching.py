import torch
import torch.nn.functional as F 

from scipy.optimize import linear_sum_assignment
from torch import nn

from box_ops import box_cxcywh_to_xyxy, generalized_box_iou, distance_box_iou, complete_box_iou


class HungarianMatcher(nn.Module):
    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0, iou="complete_box_iou"):
        """
        weight_dict : Specifies the weights used when computing the cost matrix.
        cost_class: This is the relative weight of the classification error in the matching cost
        cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
        cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        ex) weight_dict: {cost_class: 2, cost_bbox: 5, cost_giou: 2}
        
        use_focal_loss :  
        Used when there is severe class imbalance, 
        especially in cases with a large number of background classes and
        a small number of object classes.
        
        iou : Allows computing the IoU cost using different IoU variants.
        iou = "generalized_box_iou" or "distance_box_iou" or "complete_box_iou"
        """
        super().__init__()
        self.iou = iou
        
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_ciou = weight_dict['cost_ciou']
        
        self.use_focal_loss = use_focal_loss
        # Hyperparameters used in focal loss calculation
        self.alpha = alpha
        self.gamma = gamma
        
        assert self.cost_class !=0 or self.cost_bbox !=0 or self.cost_ciou != 0, "all costs can't be 0"
        
    def forward(self, outputs, targets):
        """
        Params : 
            outputs : This is a dict that contains at least these entries:
                "pred_logits" : Tensor of dim [batch_size, num_queries, num_class] with the classification logits
                "pred_bboxes" : Tensor of dim [batch_size, num_queries, 4]
            
            targets : This is a list of targets. len(targets) = batch_size.
                "labels" : Tensor of dim [num_target_boxes] containing the class labels
                "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
            ex) targets = [
                    {'labels': tensor_1, 'boxes': tensor_2},  # the dictionary corresponding to the first batch sample
                    {'labels': tensor_3, 'boxes': tensor_4},  # the dictionary corresponding to the second batch sample
                    ...
                ]
                
        Returns :
            A list of size batch_size, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["pred_logits"].shape[:2]
        
        # flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1)) # [batch_size * num_queries, num_classes]
        else :
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1) # [batch_size * num_queries, num_classes]
        
        out_bbox = outputs["pred_boxes"].flatten(0, 1) # [batch_size * num_queries, 4]
        
        target_ids = torch.cat([v["labels"] for v in targets]) # A concatenated tensor of object labels from all images in the batch.
        target_bbox = torch.cat([v["boxes"] for v in targets]) # A concatenated tensor of object bbox from all images in the batch.
        
        # Compute the classification cost
        # Hungarian matching depends only on relative ordering, 
        # so it approximates the cost using 1 - Probability[target_class] instead of the more precise but expensive NLL.
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal_loss:
            out_prob = out_prob[:, target_ids]
            # cost of negarivae smaple
            neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            # cost of positive sample
            pos_cost_class = self.slpha * ((1 - out_prob)**self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, target_ids]
            
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, target_bbox, p=1)
        
        # Compute the iou cost between boxes
        output_xyxy_bbox = box_cxcywh_to_xyxy(out_bbox)
        target_xyxy_bbox = box_cxcywh_to_xyxy(target_bbox)
    
        if self.iou == "generalized_box_iou":
            cost_ciou = -generalized_box_iou(output_xyxy_bbox, target_xyxy_bbox)
            
        elif self.iou == "distance_box_iou":
            cost_ciou = -distance_box_iou(output_xyxy_bbox, target_xyxy_bbox)
            
        else :
            cost_ciou = -complete_box_iou(output_xyxy_bbox, target_xyxy_bbox)
            
        # Final cost matrix
        cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_ciou * cost_ciou
        cost = cost.view(batch_size, num_queries, -1).cpu()
        
        sizes = [len(v["boxes"]) for v in targets] # A list of the number of objects (GT boxes) in each image
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        


outputs = {
    "pred_logits": torch.tensor([
        [  # Batch 0
            [1.2, 0.3, -0.5],  # query 0
            [0.1, 2.5, 0.3],   # query 1
            [1.0, -0.2, 1.3],  # query 2
            [0.0, 0.0, 0.0],   # query 3 (low confidence)
            [0.2, 1.1, 1.2]    # query 4
        ],
        [  # Batch 1
            [-0.5, 1.3, 1.5],  # query 0
            [0.7, 0.1, 2.0],   # query 1
            [1.0, 1.0, 1.0],   # query 2
            [0.9, 0.5, 1.7],   # query 3
            [0.0, 0.0, 0.0]    # query 4 (low confidence)
        ]
    ]),  # shape: [2, 5, 3]

    "pred_boxes": torch.tensor([
        [  # Batch 0
            [0.5, 0.5, 0.2, 0.2],  # query 0
            [0.2, 0.3, 0.1, 0.1],  # query 1
            [0.7, 0.6, 0.15, 0.1], # query 2
            [0.1, 0.1, 0.05, 0.05],# query 3
            [0.9, 0.8, 0.1, 0.1]   # query 4
        ],
        [  # Batch 1
            [0.4, 0.5, 0.2, 0.2],
            [0.6, 0.4, 0.15, 0.1],
            [0.1, 0.1, 0.1, 0.1],
            [0.9, 0.9, 0.1, 0.1],
            [0.3, 0.3, 0.2, 0.2]
        ]
    ])  # shape: [2, 5, 4]
}

targets = [
    {  # Batch 0
        "labels": torch.tensor([1, 2, 0]),  # 정답 클래스 3개
        "boxes": torch.tensor([
            [0.21, 0.31, 0.09, 0.1],   # query 1과 비슷
            [0.7, 0.6, 0.14, 0.1],     # query 2와 비슷
            [0.52, 0.5, 0.2, 0.2]      # query 0과 유사
        ])
    },
    {  # Batch 1
        "labels": torch.tensor([2, 0]),  # 정답 클래스 2개
        "boxes": torch.tensor([
            [0.61, 0.41, 0.15, 0.1],   # query 1과 유사
            [0.41, 0.51, 0.2, 0.2]     # query 0과 유사
        ])
    }
]

# 가중치 설정
weight_dict = {"cost_class": 1, "cost_bbox": 5, "cost_ciou": 2}

matcher = HungarianMatcher(weight_dict, use_focal_loss=False, iou="distance_box_iou")

matched_indices = matcher(outputs, targets)

# 결과 보기
for b, (i, j) in enumerate(matched_indices):
    print(f"[Batch {b}] Prediction indices: {i.tolist()}, Target indices: {j.tolist()}")