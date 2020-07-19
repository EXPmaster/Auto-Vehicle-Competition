# -*- coding: UTF-8 -*-
import os
import sys
current_path = os.path.dirname(__file__)
sys.path.append(current_path)
from model_service.pytorch_model_service import PTServingBaseService
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from uts.utils import preprocess, invert_affine
import torch
import numpy as np
from tools import cfg
import yaml
from torchvision.ops import nms
from torchvision.ops.boxes import batched_nms


class PTVisionService(PTServingBaseService):

    def __init__(self, model_name, model_path):
        # 调用父类构造方法
        super(PTVisionService, self).__init__(model_name, model_path)
        # 调用自定义函数加载模型
        checkpoint_file = model_path
        params = yaml.safe_load(open(f'/home/mind/model/projects/{cfg.project}.yml'))
        self.model = EfficientDetBackbone(compound_coef=cfg.compound_coef, num_classes=len(cfg.category),
                                     ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
        self.model.load_state_dict(torch.load(checkpoint_file, map_location=torch.device('cpu')))
        self.model.requires_grad_(False)
        self.model.eval()
        # self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.input_sizes = [512, 896, 768, 896, 1024, 1280, 1280, 1536]
        self.class_dict = dict([val, key] for key, val in cfg.category.items())
        
    def _preprocess(self, data):
        # https两种请求形式
        # 1. form-data文件格式的请求对应：data = {"请求key值":{"文件名":<文件io>}}
        # 2. json格式对应：data = json.loads("接口传入的json体")
        imgs_path = []
        for k, v in data.items():
            for file_name, file_content in v.items():
                imgs_path.append(file_content)

        return imgs_path
    
    def _inference(self, imgs_path):
        results = []
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        for img_path in imgs_path:
            ori_imgs, framed_imgs, framed_metas = preprocess([img_path], max_size=self.input_sizes[cfg.compound_coef])
            x = torch.from_numpy(framed_imgs[0]).float()
            x = x.unsqueeze(0).permute(0, 3, 1, 2)

            features, regression, classification, anchors = self.model(x)
            preds = self._my_postprocess(x,
                                anchors, regression, classification,
                                regressBoxes, clipBoxes,
                                cfg.threshold, cfg.nms_threshold)

            preds = invert_affine(framed_metas, preds)[0]
            scores = preds['scores']
            class_ids = preds['class_ids']
            rois = preds['rois']
            image_result = {
                'detection_classes': [],
                'detection_boxes': [],
                'detection_scores': []
            }
            if rois.shape[0] > 0:
                bbox_score = scores
                
                for roi_id in range(rois.shape[0]):
                    score = float(bbox_score[roi_id])
                    label = int(class_ids[roi_id])
                    box = rois[roi_id, :]
                    image_result['detection_classes'].append(self.class_dict[label+1])
                    image_result['detection_boxes'].append(box.tolist())
                    image_result['detection_scores'].append(score)

            results.append(image_result)

        return results

    def _postprocess(self, data):
        if len(data) == 1:
            return data[0]
        else:
            return data

    def _my_postprocess(self, x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold):
        transformed_anchors = regressBoxes(anchors, regression)
        transformed_anchors = clipBoxes(transformed_anchors, x)
        scores = torch.max(classification, dim=2, keepdim=True)[0]
        scores_over_thresh = (scores > threshold)[:, :, 0]
        out = []
        for i in range(x.shape[0]):
            if scores_over_thresh[i].sum() == 0:
                out.append({
                    'rois': np.array(()),
                    'class_ids': np.array(()),
                    'scores': np.array(()),
                })
                continue

            classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
            transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
            scores_per = scores[i, scores_over_thresh[i, :], ...]
            scores_, classes_ = classification_per.max(dim=0)
            anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

            if anchors_nms_idx.shape[0] != 0:
                classes_ = classes_[anchors_nms_idx]
                scores_ = scores_[anchors_nms_idx]
                boxes_ = transformed_anchors_per[anchors_nms_idx, :]
                boxes_ = boxes_[:,[1,0,3,2]]

                out.append({
                    'rois': boxes_.numpy(),
                    'class_ids': classes_.numpy(),
                    'scores': scores_.numpy(),
                })
            else:
                out.append({
                    'rois': np.array(()),
                    'class_ids': np.array(()),
                    'scores': np.array(()),
                })

        return out

