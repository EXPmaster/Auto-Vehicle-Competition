import os
import torch
import numpy as np
import xml.dom.minidom
from torch.utils.data import Dataset, DataLoader
#from pycocotools.coco import COCO
import cv2
from tools import cfg


class CocoDataset(Dataset):
    def __init__(self, root_dir, set='trainset', transform=None):

        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform

        #self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = os.listdir(self.set_name)

        #self.load_classes()

    # def load_classes(self):

    #     # load class names (name -> label)
    #     categories = self.coco.loadCats(self.coco.getCatIds())
    #     categories.sort(key=lambda x: x['id'])

    #     self.classes = {}
    #     for c in categories:
    #         self.classes[c['name']] = len(self.classes)

    #     # also load the reverse (label -> name)
    #     self.labels = {}
    #     for key, value in self.classes.items():
    #         self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        #print('before transform')
        #print(sample['annot'])
        if self.transform:
            sample = self.transform(sample)
        #print('after transform')
        #print(sample['annot'])
        return sample
    
    def _xml_parser(self, file_name):
        """
        用于XML解析的函数
        input: file_name
        output: str, dict
        """
        DOMTree = xml.dom.minidom.parse(file_name)
        annotation = DOMTree.documentElement
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

        return parsed
    
    def load_image(self, image_index):
        path = os.path.join(self.set_name, self.image_ids[image_index])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        # get ground truth annotations
        #annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))
        filename = str(self.image_ids[image_index]).replace('.jpg', '.xml')
        filename = os.path.join(cfg.data_root, filename)
        annot = self._xml_parser(filename)
        # some images appear to miss annotations
        classes = annot['detection_classes']
        bboxes = annot['detection_boxes']
        #print(annot)
        #print(classes)
        #print(bboxes)
        for class_sample, bbox in zip(classes, bboxes):
            annotation = np.zeros((1, 5))
            #print(a)
            bbox = list(map(float, bbox))
            annotation[0, :4] = bbox
            annotation[0, 4] = cfg.category[class_sample] - 1
            #print(annotation)
            annotations = np.append(annotations, annotation, axis=0)

        return annotations


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)
    #print(max_num_annots)

    if max_num_annots > 0:
        #print(annots)
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
        #print(annot_padded)
        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
        #print(annot_padded)
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=640):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        #print('before')
        #print(sample['annot'])
        #print(height)
        #print(width)
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale
        #print('after')
        #print(annots)

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            #print(sample['annot'])
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}
            #print(sample['annot'])

        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        #print('Normalizer')
        #print(sample['annot'])

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}

