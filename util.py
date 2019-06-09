import numpy as np
import torch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def myimshow(image, ax=plt):
    image = image.to('cpu').numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1]) 
    image = (image + 1) / 2
    image[image < 0] = 0
    image[image > 1] = 1 
    h = ax.imshow(image) 
    ax.axis('off') 
    return h

def compute_iou(box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            # [N,2] -> [N,1,2] -> [N,M,2]
            box1[:, :2].unsqueeze(1).expand(N, M, 2),
            # [M,2] -> [1,M,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),
        )

        rb = torch.min(
            # [N,2] -> [N,1,2] -> [N,M,2]
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),
            # [M,2] -> [1,M,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2]-box1[:, 0]) * (box1[:, 3]-box1[:, 1])  # [N,]
        area2 = (box2[:, 2]-box2[:, 0]) * (box2[:, 3]-box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou


def decoder(y):
    '''
    y (tensor): 7x7x30
    return (tensor): box[[xmin,ymin,xmax,ymax]] label[...] prob
    '''
    class_dict = {0:'aeroplane', 1:'bicycle', 2:'bird', 3:'boat', 4:'bottle', 5:'bus', 6:'car', 7:'cat', \
                     8:'chair', 9:'cow', 10:'diningtable', 11:'dog', 12:'horse', 13:'motorbike', 14:'person', \
                      15:'pottedplant', 16:'sheep', 17:'sofa', 18:'train', 19:'tvmonitor'}
    grid_num = 7
    y = y.data
    
    # Pick valid gridcells
    contain = torch.cat((y[:,:,4].unsqueeze(2),y[:,:,9].unsqueeze(2)),2)
    mask = ((contain > 0.3)+(contain==contain.max())).gt(0)
    
    # Loop over all 
    boxes=[]
    cls_indexs=[]
    probs = []
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                if mask[i,j,b] == 1:
                    box = y[i,j,b*5:b*5+4]
                    contain_prob = y[i,j,b*5+4]
                    #contain_prob = torch.FloatTensor([y[i,j,b*5+4]])
                    #print('c',contain_prob)
                    xy = torch.FloatTensor([j,i])/grid_num 
                    xy = xy.to(device)
                    box[:2] = box[:2]/grid_num + xy # return cxcy relative to image
                    box_xy = torch.FloatTensor(box.shape)#[cx,cy,w,h] to [x1,y1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5*box[2:]
                    box_xy[2:] = box[:2] + 0.5*box[2:]
                    max_cls_prob,cls_index = torch.max(y[i,j,10:],0)
                    #print('m',contain_prob*max_cls_prob)
                    #if float((contain_prob*max_cls_prob)[0]) > 0.1:
                    if contain_prob*max_cls_prob > 0.1:
                        boxes.append(box_xy.view(1,4))
                        cls_indexs.append(cls_index)
                        #probs.append(contain_prob*max_cls_prob)
                        probs.append(torch.FloatTensor([contain_prob*max_cls_prob]).to(device))
    if boxes:
        boxes = torch.cat(boxes,0) #(n,4)
        probs = torch.cat(probs,0) #(n,)
        cls_indexs = torch.stack(cls_indexs,0) #(n,)
    else:
        boxes = torch.zeros((1,4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
        
    valid = nms(boxes,probs)
    cls_indexs_valid = cls_indexs[valid].tolist()
    labels = [class_dict[i] for i in cls_indexs_valid]
    return boxes[valid],labels,probs[valid]


def nms(bboxes,scores,threshold=0.5):
    '''
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1) * (y2-y1)
    

    _,order = scores.sort(0,descending=True)
    keep = []
    while order.numel() > 0:
        if order.nelement()==1:
            i = order.item()
        else:
            i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break
            
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)


def show_img_with_boxes(image, boxes, labels, mode):
    # boxes: [box]
    # box: [x_of_top_left, y_of_top_left, x_of_bottom_right, y_of_bottom_right]
    #  im = np.array(Image.open('stinkbug.png'), dtype=np.uint8)
    image = image.to('cpu').numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1]) 
    image = (image + 1) / 2
    image[image < 0] = 0
    image[image > 1] = 1 
    
    # Create figure and axes
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    for i in range(len(boxes)):
        box = boxes[i]
        x_of_top_left = box[0]*448
        y_of_top_left = box[1]*448
        width = box[2]*448 - box[0]*448
        height = box[3]*448 - box[1]*448
        
        r = lambda: random.randint(0,255)
        
        label_color = dict((k,'') for k in labels)
        
        for label in labels:
            label_color[label] = '#%02X%02X%02X' % (r(),r(),r())
        
        # Create a Rectangle patch
        rect = patches.Rectangle((x_of_top_left,y_of_top_left),width,height,linewidth=1,edgecolor=label_color[labels[i]],facecolor='none', label='1212')
        # Rectangle((x_of_top_left,y_of_top_left),width,hight,linewidth=1,edgecolor='r',facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
        
        plt.text(x_of_top_left, y_of_top_left, labels[i], color=label_color[labels[i]])
        ax.set_xlabel(mode)
    
    plt.show()


def plot(exp, fig, axes):
    axes[0].clear()
    axes[1].clear()
    if exp.history:
        print(exp.history[-1])
    axes[0].plot([exp.history[k][0]['loss'] for k in range(exp.epoch)], label='training loss')
    axes[1].plot([exp.history[k][0]['mAP'] for k in range(exp.epoch)], label='training mAP')
    axes[0].plot([exp.history[k][1]['loss'] for k in range(exp.epoch)], label='evaluation loss')
    axes[1].plot([exp.history[k][1]['mAP'] for k in range(exp.epoch)], label='evaluation mAP')
    axes[0].legend()
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[1].legend()
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mAP')
    plt.tight_layout()
    fig.canvas.draw()