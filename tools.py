# -*- coding: UTF-8 -*-
import torch


class Config:
    def __init__(self):
        self.data_root = 'labeled_data_backup'
        self.train_test_rate = 0.7
        
        self.threshold = 0.5
        self.nms_threshold = 0.5
        
        self.class_num = 6
        self.useCUDA = torch.cuda.is_available()
        self.device = 'cuda:0' if self.useCUDA else 'cpu'
        self.WD = 5e-5
        self.project = 'auto'
        self.saved_path = 'logs/'
        self.log_path = 'logs/'
        self.batch_size = 8
        self.num_workers = 0
        self.data_path = 'data/'
        self.compound_coef = 1
        self.load_weights = 'weights/efficientdet-d1.pth'
        self.head_only = False
        self.debug = False
        self.optim = 'adamw'
        self.lr = 1e-5
        self.num_epochs = 30
        self.save_interval = 500
        self.val_interval = 1
        self.es_min_delta = 0.0
        self.es_patience = 0 
        self.category = {'red_stop':1, 'green_go':2, 'yellow_back':3, 'pedestrian_crossing':4, 'speed_limited':5, 'speed_unlimited':6}
        
        
cfg = Config()


class Metric:
    def __init__(self):
        self.reset()
    
    def update(self, pred, gt):
        pass
    
    def reset(self):
        pass