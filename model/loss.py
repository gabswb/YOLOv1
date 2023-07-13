import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, device, grid_size=7, num_bboxes=2, num_classes=20):
        """ Loss module for Yolo v1.
        Use grid_size, num_bboxes, num_classes information if necessary.

        Args:
            grid_size: (int) size of input grid.
            num_bboxes: (int) number of bboxes per each cell.
            num_classes: (int) number of the object classes.
        """
        super(Loss, self).__init__()
        self.S = grid_size
        self.B = num_bboxes
        self.C = num_classes
        self.device = device

    def compute_iou(self, bbox1, bbox2):
        """ Compute the IoU (Intersection over Union) of two set of bboxes, each bbox format: [x1, y1, x2, y2].
        Use this function if necessary.

        Args:
            bbox1: (Tensor) bounding bboxes, sized [N, 4].
            bbox2: (Tensor) bounding bboxes, sized [M, 4].
        Returns:
            (Tensor) IoU, sized [N, M].
        """
        N = bbox1.size(0)
        M = bbox2.size(0)

        # Compute left-top coordinate of the intersections
        lt = torch.max(
            bbox1[:, :2].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, :2].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Conpute right-bottom coordinate of the intersections
        rb = torch.min(
            bbox1[:, 2:].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Compute area of the intersections from the coordinates
        wh = rb - lt   # width and height of the intersection, [N, M, 2]
        wh[wh < 0] = 0 # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]

        # Compute area of the bboxes
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1]) # [N, ]
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1]) # [M, ]
        area1 = area1.unsqueeze(1).expand_as(inter) # [N, ] -> [N, 1] -> [N, M]
        area2 = area2.unsqueeze(0).expand_as(inter) # [M, ] -> [1, M] -> [N, M]

        # Compute IoU from the areas
        union = area1 + area2 - inter # [N, M, 2]
        iou = inter / union           # [N, M, 2]

        return iou

    def forward(self, pred_tensor, target_tensor):
        """ Compute loss.

        Args:
            pred_tensor (Tensor): predictions, sized [batch_size, S, S, Bx5+C], 5=len([x, y, w, h, conf]).
            target_tensor (Tensor):  targets, sized [batch_size, S, S, Bx5+C].
        Returns:
            loss_xy (Tensor): localization loss for center positions (x, y) of bboxes.
            loss_wh (Tensor): localization loss for width, height of bboxes.
            loss_obj (Tensor): objectness loss.
            loss_noobj (Tensor): no-objectness loss.
            loss_class (Tensor): classification loss.
        """

        batch_size = float(pred_tensor.size(0))

        obj_mask = target_tensor[:, :, :, 4] > 0  # mask for the cells which contain objects. [batch_size, S, S]
        noobj_mask = target_tensor[:, :, :, 4] == 0 # mask for the cells whithout objects. [batch_size, S, S]

        obj_mask = obj_mask.unsqueeze(-1).expand_as(target_tensor) #[batch_size, S, S, B*5+C]
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target_tensor) #[batch_size, S, S, B*5+C]

        # We denote by N the number of cells with an object and M the number of cells without obj
        # we reduce the size of tensors by applying mask 
        obj_pred = pred_tensor[obj_mask].view(-1, self.B*5+self.C)   # [N, B*5+C]
        obj_target = target_tensor[obj_mask].view(-1, self.B*5+self.C) # [N, B*5+C]
        noobj_pred = pred_tensor[noobj_mask].view(-1, self.B*5+self.C) # [M, B*5+C]
        noobj_target = target_tensor[noobj_mask].view(-1, self.B*5+self.C) # [M, B*5+C]

        # Extract confidence
        noobj_pred_conf = noobj_pred[:, 4:self.B*5:5] # [M,B]
        noobj_target_conf = noobj_target[:, 4:self.B*5:5] # [M,B]

        loss_noobj = F.mse_loss(noobj_pred_conf, noobj_target_conf, reduction='sum') / batch_size
        loss_class = F.mse_loss(obj_pred[:, 5*self.B:], obj_target[:, 5*self.B:], reduction='sum') / batch_size

        #no need classes anymore
        obj_pred_bbox = obj_pred[:, :self.B*5].contiguous().view(-1, 5) # [N*self.B, 5]
        obj_target_bbox = obj_target[:, :self.B*5].contiguous().view(-1, 5) # [N*self.B, 5]

        bbox_responsible_mask =  torch.zeros_like(obj_pred_bbox, dtype=bool).to(self.device) # [N*self.B, 5]
        obj_target_conf = torch.zeros_like(obj_target_bbox).to(self.device) # [N*self.B, 5]

        for i in range(0, obj_pred_bbox.size(0), self.B):
          pred_bbox = obj_pred_bbox[i:i+self.B] # [B,5]
          #Compute bottom left corner and top right corner
          pred_bbox_coord = torch.zeros_like(pred_bbox).to(self.device)
          pred_bbox_coord[:, 0] = pred_bbox[:, 0]/self.S - 0.5 * pred_bbox[:, 2]
          pred_bbox_coord[:, 1] = pred_bbox[:, 1]/self.S - 0.5 * pred_bbox[:, 3]
          pred_bbox_coord[:, 2] = pred_bbox[:, 0]/self.S + 0.5 * pred_bbox[:, 2]
          pred_bbox_coord[:, 3] = pred_bbox[:, 1]/self.S + 0.5 * pred_bbox[:, 3]

          #Compute bottom left corner and top right corner
          target_bbox = obj_target_bbox[i].view(-1, 5) # [1,5]
          target_bbox_coord = torch.zeros_like(target_bbox).to(self.device)
          target_bbox_coord[:, 0] = target_bbox[:, 0]/self.S - 0.5 * target_bbox[:, 2]
          target_bbox_coord[:, 1] = target_bbox[:, 1]/self.S - 0.5 * target_bbox[:, 3]
          target_bbox_coord[:, 2] = target_bbox[:, 0]/self.S + 0.5 * target_bbox[:, 2]
          target_bbox_coord[:, 3] = target_bbox[:, 1]/self.S + 0.5 * target_bbox[:, 3]

          iou = self.compute_iou(pred_bbox_coord[:, :4], target_bbox_coord[:, :4])
          max_iou, max_index = iou.max(0)
          bbox_responsible_mask[i+max_index] = 1
          obj_target_conf[i+max_index, 4] = max_iou

        pred_responsible = obj_pred_bbox[bbox_responsible_mask].view(-1, 5)
        target_responsible = obj_target_bbox[bbox_responsible_mask].view(-1, 5)
        target_responsible_conf = obj_target_conf[bbox_responsible_mask].view(-1, 5)
        loss_xy = F.mse_loss(pred_responsible[:, :2], target_responsible[:, :2], reduction='sum') / batch_size
        loss_wh = F.mse_loss(torch.sqrt(pred_responsible[:, 2:4]), torch.sqrt(target_responsible[:, 2:4]), reduction='sum') / batch_size
        loss_obj = F.mse_loss(pred_responsible[:, 4], target_responsible_conf[:, 4], reduction='sum') / batch_size

        return loss_xy, loss_wh, loss_obj, loss_noobj, loss_class