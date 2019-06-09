import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import util
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ImageOutput:
    def __init__(self, y, S, B, C):
        self.cells = list()
        for i in range(S):
            for j in range(S):
                t = y[i][j]
                label_for_this_cell = dict()
                #  t is a vector with length 30
                
                label_for_this_cell['boxes'] = list()
                label_for_this_cell['boxes'].append(dict())
                label_for_this_cell['boxes'].append(dict())
                
                label_for_this_cell['boxes'][0]['x'] = t[0]
                label_for_this_cell['boxes'][0]['y'] = t[1]
                label_for_this_cell['boxes'][0]['w'] = t[2]
                label_for_this_cell['boxes'][0]['h'] = t[3]
                label_for_this_cell['boxes'][0]['p_obj'] = t[4]
                
                label_for_this_cell['boxes'][1]['x'] = t[5]
                label_for_this_cell['boxes'][1]['y'] = t[6]
                label_for_this_cell['boxes'][1]['w'] = t[7]
                label_for_this_cell['boxes'][1]['h'] = t[8]
                label_for_this_cell['boxes'][1]['p_obj'] = t[9]
                
                label_for_this_cell['C'] = t[10:]
                self.cells.append(label_for_this_cell)


class yoloLoss_alt(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(yoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj
    
    def forward(self, y, d):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,30)
        '''
        batch_size = len(y)
    
        sum_loss = 0
    
        for i in range(batch_size):
            yy = y[i]
            dd = d[i]
            # yy: 7 x 7 x 30
            # dd: 7 x 7 x 25, since dd has only one sub-box for one cell
            sum_loss += self.get_loss_for_one_image(yy, dd)
        return sum_loss / (1.0 * batch_size)
    
    def get_loss_for_one_image(self, y, d):
        # y is the network output for one image, and d is the corresponding label.
        loss = 0
        S = 7  # S x S cells in one image
        B = 2  # B sub-boxex in one cell
        C = 20  # C classes to classify
        yy = ImageOutput(y, S, B, C)
        dd = ImageOutput(d, S, 1, C)
    
        for i in range(S*S):
            loss += self.get_loss_for_one_cell(yy.cells[i], dd.cells[i])
    
        return loss
    
    def get_loss_for_one_cell(self, y_cell, d_cell):
        # Here is a detailed explanation for loss:
        # https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088
    
        lambda_coord = 5
        lambda_noobj = 0.5
        classification_loss = 0
        localization_loss = 0
        confidence_loss = 0
    
        has_object_in_this_cell = False
        if d_cell['boxes'][0]['p_obj'] == 1:
            has_object_in_this_cell = True
    
        responsible_box = self.get_responsible_box(y_cell['boxes'], d_cell['boxes'])
    
        # localization loss
        if has_object_in_this_cell:
            x_diff = responsible_box['x'] - d_cell['boxes'][0]['x']
            y_diff = responsible_box['y'] - d_cell['boxes'][0]['y']
            localization_loss += lambda_coord * (x_diff ** 2 + y_diff ** 2)
    
            sqrt_w_diff = math.sqrt(max(0, responsible_box['w'])) -  math.sqrt(max(0, d_cell['boxes'][0]['w']))
            sqrt_h_diff =  math.sqrt(max(0, responsible_box['h'])) -  math.sqrt(max(0, d_cell['boxes'][0]['h']))
            localization_loss += lambda_coord * (sqrt_w_diff ** 2 + sqrt_h_diff ** 2)
    
        # confidence loss
        if has_object_in_this_cell:
            tuple_responsible = (responsible_box['x'], responsible_box['y'], responsible_box['w'], responsible_box['h'])
            tuple_groundtruth = (d_cell['boxes'][0]['x'], d_cell['boxes'][0]['y'],
                                 d_cell['boxes'][0]['w'], d_cell['boxes'][0]['h'])
            iou = utils.compute_iou(tuple_responsible, tuple_groundtruth)
            C_diff = responsible_box['p_obj'] * iou - d_cell['boxes'][0]['p_obj']
            # confidence = p_obj * iou
            # for ground-truth, the confidence is equal to its p_obj
            confidence_loss += C_diff ** 2
        else:
            # Question here!!! for boxes that don't contain objects, do we need to compute confidence as P_obj * IOU?
            # If so, the confidence for those boxes is always zero, which doesn't make sense for optimization
    
            # Here, we sum up the confidence loss of all boxes.
            for box in y_cell['boxes']:
                # C_diff = box['p_obj'] - d_cell['boxes'][0]['p_obj'], with d_cell['boxes'][0]['p_obj'] = 0
                C_diff = box['p_obj']
                confidence_loss += lambda_noobj * (C_diff ** 2)
    
        # classification loss
        if has_object_in_this_cell:
            C = len(y_cell['C'])
            for i in range(C):
                pc_diff = y_cell['C'][i] - d_cell['C'][i]
                classification_loss += (pc_diff ** 2)
    
        return localization_loss + confidence_loss + classification_loss
    
    def get_responsible_box(self, y_boxes, d_boxes):
        max_iou = -1
        max_iou_box = None
        d_box = (d_boxes[0]['x'], d_boxes[0]['y'], d_boxes[0]['w'], d_boxes[0]['h'])
        for box in y_boxes:
            this_box = (box['x'], box['y'], box['w'], box['h'])
            iou = utils.compute_iou(this_box, d_box)
            if iou > max_iou:
                max_iou = iou
                max_iou_box = box
        return max_iou_box
          

class yoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(yoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def get_responsible_box(self, box_pred, box_target):
        contain_obj_response_mask = torch.cuda.ByteTensor(box_target.size())
        contain_obj_response_mask.zero_()

        contain_obj_not_response_mask = torch.cuda.ByteTensor(box_target.size())
        contain_obj_not_response_mask.zero_()

        box_target_iou = torch.zeros(box_target.size()).cuda()

        for i in range(0, box_target.size()[0], 2):  # choose the best iou box
            box1 = box_pred[i:i+2]
            box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            box1_xyxy[:, :2] = box1[:, :2]/7. - 0.5*box1[:, 2:4]
            box1_xyxy[:, 2:4] = box1[:, :2]/7. + 0.5*box1[:, 2:4]
            box2 = box_target[i].view(-1,5)
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:, :2] = box2[:, :2]/7. - 0.5*box2[:, 2:4]
            box2_xyxy[:, 2:4] = box2[:, :2]/7. + 0.5*box2[:, 2:4]
            iou = util.compute_iou(box1_xyxy[:, :4], box2_xyxy[:, :4])  # [2,1]
            max_iou, max_index = iou.max(0)
            max_index = max_index.data.cuda()
            contain_obj_response_mask[i+max_index] = 1
            contain_obj_not_response_mask[i+1-max_index] = 1
            box_target_iou[i+max_index, torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()
            box_target_iou = Variable(box_target_iou).cuda()
            return box_target_iou, contain_obj_response_mask, contain_obj_not_response_mask

    def forward(self, pred_tensor, target_tensor):
        # pred_tensor: (tensor) size: (Batchsize, S, S, Bx5+20=30) [x,y,w,h,c]
        # target_tensor: (tensor) size: (Batchsize, S, S, 30)
        # S = 7, B = 2

        N = pred_tensor.size()[0]

        # Find gridcells with C=1, create mask
        contain_obj_mask = target_tensor[:, :, :, 4] > 0
        no_obj_mask = target_tensor[:, :, :, 4] == 0
        contain_obj_mask = contain_obj_mask.unsqueeze(-1).expand_as(target_tensor)
        no_obj_mask = no_obj_mask.unsqueeze(-1).expand_as(target_tensor)

        # Pre-processing pred_tensor
        contain_obj_pred = pred_tensor[contain_obj_mask].view(-1, 30) 
        box_pred = contain_obj_pred[:,:10].contiguous().view(-1, 5)  # box[x1,y1,w1,h1,c1,loch,locw]
        class_pred = contain_obj_pred[:, 10:]  # [x2,y2,w2,h2,c2]
        
        # Pre-processing target_tensor
        contain_obj_target = target_tensor[contain_obj_mask].view(-1, 30)
        box_target = contain_obj_target[:,:10].contiguous().view(-1, 5)  # [2n,5]
        class_target = contain_obj_target[:, 10:]  # [x2,y2,w2,h2,c2]

        # compute the mask for not contain obj boxes
        no_obj_pred = pred_tensor[no_obj_mask].view(-1, 30)
        no_obj_target = target_tensor[no_obj_mask].view(-1, 30)
        no_obj_pred_mask = torch.cuda.ByteTensor(no_obj_pred.size())
        no_obj_pred_mask.zero_()
        no_obj_pred_mask[:, 4] = 1
        no_obj_pred_mask[:, 9] = 1
        no_obj_pred_c = no_obj_pred[no_obj_pred_mask]
        no_obj_target_c = no_obj_target[no_obj_pred_mask]
        
        box_target_iou, contain_obj_response_mask, contain_obj_not_response_mask = self.get_responsible_box(box_pred, box_target)
       
        # get response boxes for prediction and target
        box_pred_response = box_pred[contain_obj_response_mask].view(-1, 5)
        box_pred_not_response = box_pred[contain_obj_not_response_mask].view(-1, 5)

        box_target_response = box_target[contain_obj_response_mask].view(-1, 5)
        box_target_response_iou = box_target_iou[contain_obj_response_mask].view(-1, 5)
        box_target_not_response = box_target[contain_obj_not_response_mask].view(-1, 5)
        box_target_not_response[:, 4] = 0

        # 1. Confidence loss
        contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response_iou[:, 4], reduction='sum')
        not_contain_loss = F.mse_loss(box_pred_not_response[:, 4], box_target_not_response[:, 4], reduction='sum')
        nooobj_loss = F.mse_loss(no_obj_pred_c, no_obj_target_c, reduction='sum')
        confidence_loss = 2*contain_loss + not_contain_loss + self.l_noobj*nooobj_loss

        # 2. Localization loss
        zeros = torch.zeros(box_pred_response.shape, device=device)
        x_y_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], reduction='sum')
        w_h_loss = F.mse_loss(torch.sqrt(torch.max(zeros[:, 2:4], box_pred_response[:, 2:4])), torch.sqrt(torch.max(zeros[:, 2:4], box_target_response[:, 2:4])), reduction='sum')
        localization_loss = x_y_loss + w_h_loss

        # 3. Classification loss
        classification_loss = F.mse_loss(class_pred, class_target, reduction='sum')

        return (confidence_loss + self.l_coord*localization_loss + classification_loss) / N