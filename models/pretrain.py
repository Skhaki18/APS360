import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import cv2
torch.manual_seed(88)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Basic Model Creations - Note this was changed repeatedly and may no longer represent initial state.

def pretrainedResNet(classes, trainstat):

    if(trainstat):
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(DEVICE)
        model.eval()
        return model, DEVICE

def pretrainedMobileNetLarge(classes, trainstat):
    

    if(trainstat):
        print("MobileNet")
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True).to(DEVICE)
        model.eval()
        return model, DEVICE

def pretrainedMobileLowRes(classes, trainstat):

    if(trainstat):
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True).to(DEVICE)
        model.eval()
        return model, DEVICE



def fine_tuningResNet(classes=5, trainstat=False):
    print("In HERE Fine Tuning Resnet")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(DEVICE) 


    in_features = model.roi_heads.box_predictor.cls_score.in_features 
  
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, classes) 

    return model, DEVICE

def fine_tuningMobileNet(classes=5, trainstat=False):
    print("In HERE Fine Tuning Resnet")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True).to(DEVICE) 

    
    in_features = model.roi_heads.box_predictor.cls_score.in_features 
 
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, classes) 

    return model, DEVICE


