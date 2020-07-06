# -*- coding: UTF-8 -*-
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import xml.dom.minidom
from Utils import cfg
import os


def get_train_test():
    """
    划分训练集与验证集的idx（以名称为索引）
    return: dict
    """
    def listdir(path):
        for item in os.listdir(path):
            if not item.startswith('.'):
                yield item
    
    data_list = [file for file in listdir(cfg.data_root)
                  if file[-3:] == 'xml']
    data_list.sort(key=lambda x: int(x.replace('.xml', '')))
    
    train_test_dict = {}
    data_array = np.array(data_list)
    idx_array = np.zeros_like(data_array, dtype=bool)
    idx = np.random.choice(len(data_list), int(len(data_list)*cfg.train_test_rate), replace=False)
    idx_array[idx] = True
    
    train_test_dict['train'] = data_array[idx_array]
    train_test_dict['val'] = data_array[~idx_array]
    
    return train_test_dict
        


class ImageDataset(Dataset):
    """
    读取数据，为dataloader准备
    """
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transforms.ToTensor() if transform == None else transform
    
    def __getitem__(self, idx):
        label_path = os.path.join(cfg.data_root, self.file_list[idx])
        img_name, parsed_label = self._xml_parser(label_path)
        img_path = os.path.join(cfg.data_root, img_name)
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError('Cannot find the image')
        
        return self.transform(img), parsed_label
    
    def __len__(self):
        return len(self.file_list)

    def _xml_parser(self, file_name):
        """
        用于XML解析的函数
        input: file_name
        output: str, dict
        """
        DOMTree = xml.dom.minidom.parse(file_name)
        annotation = DOMTree.documentElement
        pic_name = annotation.getElementsByTagName('filename')[0]
        pic_name = pic_name.childNodes[0].data
        objects = annotation.getElementsByTagName('object')
        coords_key = ['xmin', 'ymin', 'xmax', 'ymax']
        parsed = {'detection_classes': [], 'detection_boxes': []}
        for obj in objects:
            name = obj.getElementsByTagName('name')[0].childNodes[0].data
            parsed['detection_classes'].append(name)
            bboxes = obj.getElementsByTagName('bndbox')[0]
            tmp_list = [bboxes.getElementsByTagName(key)[0].childNodes[0].data 
                        for key in coords_key]
            parsed['detection_boxes'].append(tmp_list)

        return pic_name, parsed

        
if __name__ == '__main__':
    # fname = os.path.join(cfg.data_root, '4434.xml')
    # xml_parser(fname)
    train_val_dict = get_train_test()
    trainset = ImageDataset(train_val_dict['train'])
    print(next(iter(trainset)))
        
        
        