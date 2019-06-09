import numpy as np 
import torch
import nntools as nt
from statistics import mean
from collections import defaultdict
from collections import Counter
import util

class ClassificationStatsManager(nt.StatsManager):

    def __init__(self):
        super(ClassificationStatsManager, self).__init__()

    def init(self):
        super(ClassificationStatsManager, self).init()
        self.running_accuracy = 0
        self.mAP = []
        self.AP = {}

    def accumulate(self, loss, x, y, d, cnt, eval_mode):
        super(ClassificationStatsManager, self).accumulate(loss, x, y, d)
        # _, l = torch.max(y, 1)
        # self.running_accuracy += torch.mean((l == d).float())
        if eval_mode=='train':
            if cnt != 0 and cnt % 10 == 0:
                mAP, _ = get_mAP(y,d)
                self.mAP.append(mAP)
        else:
            mAP, AP_dict = get_mAP(y,d)
            self.mAP.append(mAP)
            self.AP = Counter(self.AP) + Counter(AP_dict)

    def summarize(self, eval_mode):
        loss = super(ClassificationStatsManager, self).summarize()
        MAP = mean(self.mAP)
        self.mAP = []
        if eval_mode=='train':
            return {'loss': loss, 'mAP': MAP}
        else:
            AP = {}
            for key in self.AP:
                AP[key] = self.AP[key]/self.number_update
            self.AP = {}
            return {'loss': loss, 'mAP': MAP, 'AP':AP}

def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre

def get_mAP(y, d):
    temp = d[0,:,:,:]
    batchsize = d.shape[0]
    class_mAP = {'aeroplane':[], 'bicycle':[], 'bird':[], 'boat':[], 'bottle':[], 'bus':[], 'car':[], 'cat':[], \
                     'chair':[], 'cow':[], 'diningtable':[], 'dog':[], 'horse':[], 'motorbike':[], 'person':[], \
                      'pottedplant':[], 'sheep':[], 'sofa':[], 'train':[], 'tvmonitor':[]}
    result = {}
    threshold = [0.1,0.2,0.3,0.4,0.5] #thrd for prob to be selected for valid
    thresIOU = 0.5
    mAP = []
    for i in range(batchsize):
        #boxD: Tensor of xmin, ymin, xmax ymax N*4
        #label:list o
        #probs:Tensor of N, prob

        rec = defaultdict(list)
        prec = defaultdict(list)


        boxD,labelsD,probsD = util.decoder(d[i])
        boxY,labelsY,probsY = util.decoder(y[i])
        APs = []
        for thred in threshold:
            predlabel = defaultdict(list)
            baselabel = defaultdict(list)
            for t in range(len(labelsY)):
                if(probsY[t] > thred):
                    predlabel[labelsY[t]].append(boxY[t])

            for t in range(len(labelsD)):
                baselabel[labelsD[t]].append(boxD[t])

            for key, value in baselabel.items():
                if (len(value) == 0):
                    continue
                #print(value)
                listbas = value
                if not(key in predlabel):
                    continue
                #listpre = predlabel[key]
                 #print(value)
                #print(value)
                listbas = torch.stack(value, 0)
                #print("listbas : " + str(listbas))
#                 print(predlabel[key])
                listpre = torch.stack(predlabel[key])
                if(len(listpre) == 0):
                    rec[key].append(0)
                    prec[key].append(0)
                    class_mAP[key].append(0)
                    continue
                iou = util.compute_iou (listpre, listbas)
                
                #print(iou)
                numofPre= 0
                for idx in range(iou.size()[0]):
                    #print(iou[idx])
                    #print("mx:"+str(max(iou[idx]).float()))
                    if max(iou[idx]).float() > thresIOU:
                        numofPre += 1
                rec[key].append((float(numofPre))/len(value))
                prec[key].append((float(numofPre))/len(predlabel[key]))
            for key, value in predlabel.items():
                if key not in baselabel:
                    rec[key].append(0)
                    prec[key].append(0)
                    class_mAP[key].append(0)
        for key, value in rec.items():
            ap,_,_ = voc_ap(rec[key], prec[key])
            APs.append(ap)
            class_mAP[key].append(ap)
        for key, value in class_mAP.items():
            if(len(class_mAP[key]) > 0):
                result[key] = mean(class_mAP[key])
        if(len(APs) == 0):
            continue
        mAP.append(mean(APs))
    return mean(mAP), result


    