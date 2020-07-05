import torch


class Config:
    def __init__(self):
        self.train_path = './parsed_data/train_data_label.pkl'
        self.val_path = './parsed_data/eval_data_label.pkl'
        
        self.pkt_len = 100
        self.pkt_num_per_flow = 3
        self.F1_thresh = 0.5
        
        self.batchSize = 64
        self.class_num = 4
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