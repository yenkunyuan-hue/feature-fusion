from fusion.dual_CNN import DualGeneralizedRCNN
from fusion.feature_fusion import DualResNet50
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models import resnet50
from torchvision.ops import MultiScaleRoIAlign
import torch.nn as nn

identity = nn.Identity()
identity.out_channels = 512

def assemble_model(class_num, sonar_weight, camera_weight):
    camera_backbone = resnet50(weights = camera_weight)
    sonar_backbone = resnet50(weights = sonar_weight)
    backbone = DualResNet50(model_camera=camera_backbone, 
                            model_sonar=sonar_backbone,
                            out_channel=512)
    camera_f = FasterRCNN(
        backbone=identity,
        num_classes=class_num,
        rpn_anchor_generator=AnchorGenerator(sizes=((32,), (64,), (128,), (256,)),
                                             aspect_ratios=((0.5, 1.0, 2.0),) * 4),
        #（32,) is a tuple, (32) is int, we need tuple here
        box_roi_pool= MultiScaleRoIAlign(featmap_names=["c2", "c3", "c4", "c5"], 
                                         output_size= 7, 
                                         sampling_ratio = 2)
        )
    sonar_f = FasterRCNN(
            backbone=identity,
            num_classes=class_num,
            rpn_anchor_generator=AnchorGenerator(sizes=((32,), (64,), (128,), (256,)),
                                                 aspect_ratios=((0.5, 1.0, 2.0),) * 4),
            #（32,) is a tuple, (32) is int, we need tuple here
            box_roi_pool= MultiScaleRoIAlign(featmap_names=["c2", "c3", "c4", "c5"], 
                                             output_size= 7, 
                                             sampling_ratio = 2)
            )
    model = DualGeneralizedRCNN(backbone=backbone, rpn_1=camera_f.rpn, rpn_2=sonar_f.rpn,
                        roi_heads_1=camera_f.roi_heads,
                        roi_heads_2=sonar_f.roi_heads,
                        transform=camera_f.transform)
    return model
