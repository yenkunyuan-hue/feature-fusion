import torch.nn as nn
"""
Implements the Generalized R-CNN framework
"""
from collections import OrderedDict
from typing import Optional, Union

import torch
from torch import nn
#come from https://github.com/pytorch/vision/blob/main/torchvision/models/detection/generalized_rcnn.py
class DualGeneralizedRCNN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        rpn_1: nn.Module,
        rpn_2: nn.Module,
        roi_heads_1: nn.Module,
        roi_heads_2: nn.Module,
        transform: nn.Module,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn_camera = rpn_1
        self.rpn_sonar = rpn_2
        self.roi_heads_sonar = roi_heads_1
        self.roi_heads_camera = roi_heads_2

    def forward(
        self,
        images_camera: list[torch.Tensor],
        images_sonar: list[torch.Tensor],
        targets_camera: Optional[list[dict[str, torch.Tensor]]] = None,
        targets_sonar: Optional[list[dict[str, torch.Tensor]]] = None):
        if self.training:
            if targets_camera is None or targets_sonar is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets_sonar:
                    boxes_sonar = target["boxes"]
                    if isinstance(boxes_sonar, torch.Tensor):
                        torch._assert(
                            len(boxes_sonar.shape) == 2 and boxes_sonar.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes_sonar.shape}.",
                        )
                    else:
                        torch._assert(
                            False,
                            f"Expected target boxes to be of type Tensor, got {type(boxes_sonar)}.",
                        )
                for targets in targets_camera:
                    boxes_camera = targets["boxes"]
                    if isinstance(boxes_camera, torch.Tensor):
                        torch._assert(
                            len(boxes_camera.shape) == 2 and boxes_camera.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes_camera.shape}.",
                        )
                    else:
                        torch._assert(
                            False,
                            f"Expected target boxes to be of type Tensor, got {type(boxes_sonar)}.",
                        )
        original_image_sizes: list[tuple[int, int]] = []
        #sonar and camera are the same in this project
        for img_c in images_camera:
            val = img_c.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img_c.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images_sonar, targets_sonar = self.transform(images_sonar, targets_sonar)
        images_camera, targets_camera = self.transform(images_camera, targets_camera)

        if targets_sonar is not None:
            for target_idx, target in enumerate(targets_sonar):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: list[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )
        if targets_camera is not None:
            for target_idx, target in enumerate(targets_camera):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: list[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        features_camera, features_sonar = self.backbone(images_camera.tensors, images_sonar.tensors)

        if isinstance(features_sonar, torch.Tensor):
            features_sonar = OrderedDict([("0", features_sonar)])
        if isinstance(features_camera, torch.Tensor):
            features_camera = OrderedDict([("0", features_camera)])

        sonar_proposals, sonar_proposal_losses = self.rpn_sonar(images_sonar, features_sonar, targets_sonar)
        sonar_detections, sonar_detector_losses = self.roi_heads_sonar(features_sonar, sonar_proposals, images_sonar.image_sizes, targets_sonar)

        camera_proposals, camera_proposal_losses = self.rpn_camera(images_camera, features_camera, targets_camera)
        camera_detections, camera_detector_losses = self.roi_heads_camera(features_camera, camera_proposals, images_camera.image_sizes, targets_camera)
        camera_detections = self.transform.postprocess(
            camera_detections, images_camera.image_sizes, original_image_sizes
        )  # type: ignore[operator]

        sonar_detections = self.transform.postprocess(
                    sonar_detections, images_sonar.image_sizes, original_image_sizes
                )  # type: ignore[operator]

        losses = {"camera": {},"sonar":{} }
        losses["camera"].update(camera_detector_losses)
        losses["camera"].update(camera_proposal_losses)
        losses["sonar"].update(sonar_detector_losses)
        losses["sonar"].update(sonar_proposal_losses)

        return camera_detections, sonar_detections, losses

