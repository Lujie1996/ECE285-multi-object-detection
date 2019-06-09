import os
import numpy as np 
import torchvision as tv
from PIL import Image
import xml.etree.ElementTree as ET
import torch
import torch.utils.data as td 

#dataset_root_dir = '/datasets/ee285f-public/PascalVOC2012'

class VOCDataset(td.Dataset):

    def __init__(self, root_dir, mode='train', image_size=(448, 448), S=7): 
        super(VOCDataset, self).__init__()
        self.mode = mode
        self.image_size = image_size
        self.S = S
        if mode == 'train':
            self.list_file = os.path.join(root_dir, 'ImageSets/Main/train.txt')
            #self.list_file = 'train.txt'
        else:
            self.list_file = os.path.join(root_dir, 'ImageSets/Main/val.txt')
            #self.list_file = 'val.txt'
        
        self.annot_dir = os.path.join(root_dir, 'Annotations')
        
        with open(self.list_file) as f:
            lines = f.readlines()    
        
        self.image_names = []
        for line in lines:
            self.image_names.append(line[:11])
        
        self.images_dir = os.path.join(root_dir, 'JPEGImages')

    def __len__(self):
        return len(self.image_names)

    def __repr__(self):
        return "VOCDataset(mode={}, image_size={})". \
            format(self.mode, self.image_size)

    def __getitem__(self, idx):
        iname = self.image_names[idx]
        img_path = os.path.join(self.images_dir, iname+'.jpg') 
        
        # Read annotations from xml
        tree = ET.parse(os.path.join(self.annot_dir, iname+'.xml'))
        boxes = []
        labels = []
        for obj in tree.iter(tag='object'):
            bbox = [int(obj.find('bndbox').find('xmax').text),\
                    int(obj.find('bndbox').find('ymax').text),\
                    int(obj.find('bndbox').find('xmin').text),\
                    int(obj.find('bndbox').find('ymin').text)]
            label = obj.find('name').text
            boxes.append(bbox)
            labels.append(label)
        isize = (int(tree.find('size').find('width').text), int(tree.find('size').find('height').text))
        boxes = torch.FloatTensor(boxes)
        
        # Read images and perform random processing and normalization
        img = Image.open(img_path).convert('RGB')
        if self.mode=='train':
            transform = tv.transforms.Compose([
                tv.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
                tv.transforms.Resize(self.image_size),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((.5, .5,.5),(.5, .5, .5))
                ])
        else:
            transform = tv.transforms.Compose([
                tv.transforms.Resize(self.image_size),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((.5, .5,.5),(.5, .5, .5))
                ])
        img = transform(img)
        
        target = self.encoder(boxes, labels, isize)
        
        # Return img(3x448x448 torch tensor), target(7x7x30 torch tensor)
        return img, target
    
    def encoder(self, boxes, labels, isize):
        '''
        Encode boxes and labels to 7x7x30 tensor. For each area, the 30 len tensor has such structure:
        [ 20(class label) | 1(C) | 1(C) | 4(width, height, center_w, center_h, and all are ratio) | 4(the same) ]
        '''
        S = self.S
        class_dict = {'aeroplane':0, 'bicycle':1, 'bird':2, 'boat':3, 'bottle':4, 'bus':5, 'car':6, 'cat':7, \
                     'chair':8, 'cow':9, 'diningtable':10, 'dog':11, 'horse':12, 'motorbike':13, 'person':14, \
                      'pottedplant':15, 'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19}
        target = torch.zeros((S, S, 30))
        wh = boxes[:, :2]-boxes[:, 2:]
        cxcy = (boxes[:, :2]+boxes[:, 2:])/2
        
        # [x,y,w,h,c | x,y,w,h,c | C     ] <- [C      | c1 | c2 | w, h, x, y | w, h, x, y]
        #  0 1 2 3 4   5 6 7 8 9   10..29      0....19  20   21   22 23 24 25  26 27 28 29
        
        for i in range(cxcy.size()[0]):
            center = cxcy[i]
            loc = (int(center[0]/(isize[0]/S)), int(center[1]/(isize[1]/S)))
            target[loc[1], loc[0], 4] = 1
            target[loc[1], loc[0], 9] = 1
            target[loc[1], loc[0], class_dict[labels[i]]+10] = 1
            normalized_wh = torch.tensor([wh[i,0]/isize[0], wh[i,1]/isize[1]])
            normalized_center = torch.tensor([center[0]/isize[0], center[1]/isize[1]])
            xy = torch.tensor([loc[0]/S, loc[1]/S]) 
            delta_xy = (normalized_center -xy)*S
            target[loc[1], loc[0], 2:4] = normalized_wh
            target[loc[1], loc[0], :2] = delta_xy
            target[loc[1], loc[0], 7:9] = normalized_wh
            target[loc[1], loc[0], 5:7] = delta_xy
            
        return target