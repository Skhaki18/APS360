
import argparse
from train import train, Livetrain
import yaml
import numpy as np
import cv2
from data import AnnotationLive, DynamicAnnotation
import os
import torchvision.transforms as transforms
# from util.engine import train_one_epoch, evaluate
from test import IOU, get_single_image_results
import matplotlib.pyplot as plt
import torch
global args

def input_arg(parser):
    parser.add_argument(
        '--config', dest='config',
        default='./config/Base.yaml', type=str,
        help="Default: Base.yaml")

    parser.add_argument(
        '--arch', dest='arch',
        default='ResNet', type=str,
        help="Default: ResNet")

    parser.add_argument(
        '--mode', dest='mode',
        default="Test" #Train
        , type=str,
        help="Default: Train"
    )

    
    return parser
     


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process CLI')
    parser = input_arg(parser)
    args = parser.parse_args()
    print(args.config)
    with open(args.config) as f:
        args.config = yaml.load(f, Loader=yaml.FullLoader)

    if(args.mode == "Train"):
        # Modified train to load and visualize data!
        model, ValidationSet, DEVICE = train(args)
        #Visualize output from train!


        imageOG,label, base = ValidationSet[6]
        
        image = imageOG.to(DEVICE)
        
        detections = model([image])
        print(detections)
        print(label)

        # evaluate(model, dataloader, device=DEVICE)
        AnnotationLive(detections[0], base, args.config["class_ids"], label)

    elif(args.mode == "Test"):
        # Loading in pretrained or Fine Tuned train
        model, ValidationSet, DEVICE = Livetrain(args, Live=True)
    

       
        imageOG,label, base = ValidationSet[23]
        

        image = imageOG.to(DEVICE)
        # print(model)
        
        detections = model([image])
        # print(detections[1][0])
        # Note detections could be [0] or [1][0] depending on the model!
        print(IOU(detections[0], args.config["class_ids"], label, base))

        AnnotationLive(detections[0], base, args.config["class_ids"], label)

        

    else:

        # Supports pretrained and Fine TUning models
        model, ValidationSet, DEVICE = Livetrain(args)
        #Various items tested below!
        
        # transform = transforms.Compose([transforms.ToTensor()])
        # img = transform(cv2.cvtColor(cv2.imread("carFrames/frame"+str(6)+".jpg"), cv2.COLOR_BGR2RGB))
        
        # detections = model([img])
        # print(detections)

        # OG: 6, Jeep: 23, Surferes: 3, 1, Bad: 22 
        # print(len(ValidationSet))
        # Train: 1, 3, 12, 14, 10
        imageOG,label, base = ValidationSet[96]

        image = imageOG.to(DEVICE)
        print(imageOG)
        
        detections = model([image])
    
        BoxValid = []
        ScorePred = []
        
        for i in label:
            idx = int(i["category_id"])
            if(idx in args.config["class_ids"]):
                BoxValid.append(i["bbox"])


        
        print(detections[0])
        BoxPred = []
        ScorePred = []
        for i in range(len(detections[0]["boxes"])):
            if (detections[0]["labels"][i].numpy() in args.config["class_ids"]):
                BoxPred.append(detections[0]["boxes"][i].detach().numpy().tolist())
                ScorePred.append(detections[0]["scores"][i].detach().numpy().tolist())
       

        print(BoxPred)
        print(ScorePred)
        pred_boxes = {"boxes":BoxPred, "scores":ScorePred}

    
        print(IOU(detections[0], args.config["class_ids"], label, base))
        AnnotationLive(detections[0], base, args.config["class_ids"], label)
        
        TempIOU = []

        
        for i in range(2000):
            
            imageOG,label, base = ValidationSet[i]
            image = imageOG.to(DEVICE)    
            detections = model([image])
            temp = IOU(detections[0], args.config["class_ids"], label, base)
            if (temp == 0):
                print("0 case", "Epoch", str(i))
                continue
            if (temp == -1): 
                print("-1 case", "Epoch", str(i))
                continue

            if (temp <= 0.01):
                np.savetxt("LowestPercentages", temp, delimiter=',')
            
            TempIOU.append(temp)
            print("Epoch", str(i), str(TempIOU[-1]))
            AnnotationLive(detections[0], base, args.config["class_ids"], label)

        print(TempIOU)
        print(np.mean(TempIOU))
        
        # Video Demonstration Below
       
        
        
        
        
        vidcap = cv2.VideoCapture('cars.mp4')
        frame_width = int(vidcap.get(3))
        frame_height = int(vidcap.get(4))
        frame_size = (frame_width,frame_height)
        fps = 20
        output = cv2.VideoWriter('t2est.mp4', cv2.VideoWriter_fourcc(*'XVID'), 10, frame_size)
        success,image = vidcap.read()
        count = 0
        transform = transforms.Compose([transforms.ToTensor()])
        os.mkdir("carFrames")
        # for i in range(90):
        success,image = vidcap.read()
        while success:
            print(image.shape)
           
            cv2.imwrite("carFrames/frame"+str(count)+".jpg", image)     # save frame as JPEG file   
            
            
            img = transform(cv2.cvtColor(cv2.imread("carFrames/frame"+str(count)+".jpg"), cv2.COLOR_BGR2RGB))
            detections = model([img])
            # print(detections)
            
            image = DynamicAnnotation(detections[0], image, args.config["class_ids"])
            cv2.imwrite("carFrames/frame"+str(count)+".jpg", image)

            success,image = vidcap.read()
            print('Read a new frame: ', success)
            count += 1
            
      

        for i in range(299):
            print(i)
            frame = cv2.imread("carFrames/frame"+str(i)+".jpg")
            output.write(frame)
        output.release()


    
      