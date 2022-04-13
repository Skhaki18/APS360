from typing import ClassVar
import numpy as np
# from copy import deepcopy
# import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import cv2

#NOTE: THE BELOW FUNCIONS INCLUDE CALCULATION BASED METRICS FROM REPUTABLE SOURCES, PARTICULARLY THE
#IOU AND PRECISION RECALL METRICS IN THIS FILE COME FROM  PYTORCH AND KAGGLE. THESE ARE REPUTABLE SOURCES
# AND WE ACKNOWLEDGE THE CONTRIBUTION IN SOME OF THE BELOW FUNCTIONS, HOWEVER, WE HAVE MADE SERIOUS
# MODIFICATIONS TO ALL THE FUNCTIONS AND ACTUALLY DON'T USE HALF OF THE KAGGLE BASED FUNCTIONS
# THEY ARE SIMPLY INCLUDED FOR COMPLETENESS OF WHAT WE TESTED. IN PARTICULAR
# IOU AND WEIGHTED IOU WERE RE-WRITTEN FROM SCRATCH BY US TO SOLVE OUR PROBLEMS AND GENERATE
# BETTER METRICS - THE PRECISION AND RECALL RESULTS COME FROM THE TRAIN ONE EPOCH ENGINE.


def calc_iou( gt_bbox, pred_bbox):
    '''
    This function takes the predicted bounding box and ground truth bounding box and 
    return the IoU ratio
    '''
    x_topleft_gt, y_topleft_gt, x_bottomright_gt, y_bottomright_gt= gt_bbox
    x_topleft_p, y_topleft_p, x_bottomright_p, y_bottomright_p= pred_bbox
    
    if (x_topleft_gt > x_bottomright_gt) or (y_topleft_gt> y_bottomright_gt):
        raise AssertionError("Ground Truth Bounding Box is not correct")
    if (x_topleft_p > x_bottomright_p) or (y_topleft_p> y_bottomright_p):
        raise AssertionError("Predicted Bounding Box is not correct",x_topleft_p, x_bottomright_p,y_topleft_p,y_bottomright_gt)
        
         
    #if the GT bbox and predcited BBox do not overlap then iou=0
    if(x_bottomright_gt< x_topleft_p):
        # If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox
        
        return 0.0
    if(y_bottomright_gt< y_topleft_p):  # If bottom right of y-coordinate  GT  bbox is less than or above the top left of y coordinate of  the predicted BBox
        
        return 0.0
    if(x_topleft_gt> x_bottomright_p): # If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox
        
        return 0.0
    if(y_topleft_gt> y_bottomright_p): # If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox
        
        return 0.0
    
    
    GT_bbox_area = (x_bottomright_gt -  x_topleft_gt + 1) * (  y_bottomright_gt -y_topleft_gt + 1)
    Pred_bbox_area =(x_bottomright_p - x_topleft_p + 1 ) * ( y_bottomright_p -y_topleft_p + 1)
    
    x_top_left =np.max([x_topleft_gt, x_topleft_p])
    y_top_left = np.max([y_topleft_gt, y_topleft_p])
    x_bottom_right = np.min([x_bottomright_gt, x_bottomright_p])
    y_bottom_right = np.min([y_bottomright_gt, y_bottomright_p])
    
    intersection_area = (x_bottom_right- x_top_left + 1) * (y_bottom_right-y_top_left  + 1)
    
    union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)
   
    return intersection_area/union_area


def calc_precision_recall(image_results):
    """Calculates precision and recall from the set of images
    Args:
        img_results (dict): dictionary formatted like:
            {
                'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'img_id2': ...
                ...
            }
    Returns:
        tuple: of floats of (precision, recall)
    """
    true_positive=0
    false_positive=0
    false_negative=0
    for img_id, res in image_results.items():
        true_positive +=res['true_positive']
        false_positive += res['false_positive']
        false_negative += res['false_negative']
        try:
            precision = true_positive/(true_positive+ false_positive)
        except ZeroDivisionError:
            precision=0.0
        try:
            recall = true_positive/(true_positive + false_negative)
        except ZeroDivisionError:
            recall=0.0
    return (precision, recall)


def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """
    all_pred_indices= range(len(pred_boxes))
    all_gt_indices=range(len(gt_boxes))
    if len(all_pred_indices)==0:
        tp=0
        fp=0
        fn=0
        return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
    if len(all_gt_indices)==0:
        tp=0
        fp=0
        fn=0
        return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
    
    gt_idx_thr=[]
    pred_idx_thr=[]
    ious=[]
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou= calc_iou(gt_box, pred_box)
            
            if iou >iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)
    iou_sort = np.argsort(ious)[::1]
    if len(iou_sort)==0:
        tp=0
        fp=0
        fn=0
        return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
    else:
        gt_match_idx=[]
        pred_match_idx=[]
        for idx in iou_sort:
            gt_idx=gt_idx_thr[idx]
            pr_idx= pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if(gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp= len(gt_match_idx)
        fp= len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)
    return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}



# ClassA = [[226,307, 412, 426],[571, 293, 625, 340]]
# ClassB = [[212.53, 303.5, 195.94+212.53, 124.4+ 303.5],[
# 568.63, 294.18, 50.2+568.63, 45.86+294.18]]


# We wrote the below from scratch !!!!!!!

# with specefc paramters we turn IOU to weighted and vice versa below!
def IOU(detection, ids, labels, img):
    IOUList = []
    TotalArea = 0
    count = 0
    # print(detection)
    detection["boxes"] = detection["boxes"].tolist()
    detection["labels"] = detection["labels"].tolist()
    detection["scores"] = detection["scores"].tolist()
    for i in labels:
        if(i["iscrowd"] == 1): continue
        # if(i["category_id"] in ids):
        (startX, startY, endX, endY) = i["bbox"]
        TotalArea += endX*endY
        count += 1

    # if (TotalArea < 0.15*(img.shape[0] * img.shape[1])):
    #     return -1
    for i in labels:
        if(i["iscrowd"] == 1): continue
        # if(i["category_id"] in ids):
            
        MaxIou = 0
        # GT Bound Box
        (startX, startY, endX, endY) = i["bbox"]
        validation = [int(startX), int(startY), int(startX+endX), int(startY+endY)]
        CurrentArea = 100*abs((endX) *(endY))

        
        for j in range(0, len(detection["boxes"])):
            confidence = detection["scores"][j]
            if confidence > 0.5:
                idx = int(detection["labels"][j])
                # if(idx == i["category_id"]):
                    # Detections
                box = detection["boxes"][j]
                (startX, startY, endX, endY) = box#.astype("int")
                prediction = [startX, startY, endX, endY]

                
                # IOU Weight Sum
                localIOU = calc_iou(validation, prediction)
        
                MaxIou = max(localIOU, MaxIou)
                    

        IOUList.append(MaxIou*(CurrentArea/TotalArea))



    # Another iteration of our IOU metric!
    # for i in range(0, len(detection["boxes"])):
        
    #     confidence = detection["scores"][i]
    #     if confidence > 0.5:
    #         idx = int(detection["labels"][i])
    #         if idx in ids:
                
    #             box = detection["boxes"][i].detach().cpu().numpy()
    #             (startX, startY, endX, endY) = box.astype("int")
    #             # print(startX, startY, endX, endY)
    #             prediction = [startX, startY, endX, endY]
    #             CurrentArea = 100*abs((startX-endX) *(startY-endY))
    #             MaxIou = 0
    #             for i in labels:
    #                 if(i["category_id"] in ids):
    #                     (startX, startY, endX, endY) = i["bbox"]
    #                     validation = [int(startX), int(startY), int(startX+endX), int(startY+endY)]
    #                     localIOU = calc_iou(validation, prediction)
    #                     MaxIou = max(localIOU, MaxIou)

    #             IOUList.append(MaxIou *(CurrentArea/TotalArea))

    # Factoring in non-& miss detections
    # if len(IOUList) != count:
    #     for i in range(0, abs(count-len(IOUList))):
    #         IOUList.append(0)
    return np.sum(IOUList)/100
    # return np.mean(IOUList)
