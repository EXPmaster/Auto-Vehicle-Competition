# -*- coding: UTF-8 -*-
import torch


class Config:
    def __init__(self):
        self.data_root = './labeled_data_backup'
        self.train_test_rate = 0.7
        
        self.F1_thresh = 0.5
        
        self.batchSize = 64
        self.class_num = 6
        self.epoch = 20
        self.useCUDA = torch.cuda.is_available()
        self.device = 'cuda:0' if self.useCUDA else 'cpu'
        self.LR = 1e-3
        self.WD = 5e-5
        
        
cfg = Config()


class Metric:
    def __init__(self):
        self.reset()
    
    def update(self, pred, gt):
        pass
    
    def reset(self):
        pass