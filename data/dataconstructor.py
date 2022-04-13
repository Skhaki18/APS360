import fiftyone as fo
import fiftyone.zoo as foz


import torchvision.transforms as transforms

from PIL import Image
import os
import os.path
from pycocotools.coco import COCO
import torch.utils.data as data
import cv2

import torch
import matplotlib.pyplot as plt




class CocoDetection(data.Dataset):
    """`MS Coco Captions <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    
    def __init__(self, root, annFile, args, Live = False, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.filterClasses = args.config["class_type"]
        self.CatIds = self.coco.getCatIds(catNms=self.filterClasses)
        if(args.config["class_ids"] == []):
            temp = self.coco.getCatIds(catNms=args.config["scanner"])
            args.config["class_ids"] = list(temp)
        self.ids = []
        for i in self.CatIds:
            subset = self.coco.getImgIds(catIds=[i])
            for i in range(len(subset)):
                mid = subset[i]
                self.ids.append(mid)
        self.ids = list(set(self.ids)) 
        print("CurrentSelfIDS", self.ids)
        self.transform = transform
        self.target_transform = target_transform
        self.Live = Live

    def __getitem__(self, index):
        
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        img_id = self.ids[index]
        path = coco.loadImgs(img_id)[0]['file_name']

        base = plt.imread(os.path.join(self.root, path))
        img = Image.open(os.path.join(self.root, path))
        if self.transform is not None:
            img = self.transform(img)
        
    
        if self.target_transform is not None:
            target = self.target_transform(target)

        if(self.Live): #Handles Live Annotations!!!
            return img, target , base

        boxes = []
        labels = []
        # Define Indices: [0:car, 1: truck, 2: bus, 3: motorcycle, 4:person]
        # annotation file
        Set = [3]#, 8, 6, 4, 1]
        area = 0
        for i in labels:
            if(i["category_id"] in Set):
                (startX, startY, endX, endY) = i["bbox"]
                area += endX*endY

        for i in target:
            id = i["category_id"]
            if i["category_id"] in Set:
                labels.append(Set.index(id))
                (startX, startY, endX, endY) = i["bbox"]
                boxes.append([round(startX), round(startY), round(startX+endX), round(startY+endY)])
       
        # Standard One Epoch Preprocessing PyTorch Supported!
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        
        labels = torch.as_tensor(labels, dtype=torch.int64)



        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([index])
        target["image_id"] = image_id


        return img, target

    def __len__(self):
        return len(self.ids)


def downloadData():
    dataset = foz.load_zoo_dataset(
    "coco-2017",
    label_types=["detections"],
    classes=["person", "car", "truck", "motorcycle", "bus"],
    seed = 420,
    shuffle = True,
    include_id = True
    )
    dataset.persistent = True

    print(dataset.info)


def constructData(BasePath, args, Live):

    transform = transforms.Compose([transforms.Resize((480, 640)),transforms.RandomRotation(10), transforms.ToTensor()])
    Train = CocoDetection(root = BasePath+"train/data",annFile = BasePath+"train/labels.json",  args=args, transform=transform, Live=Live)
    Validation = CocoDetection(root = BasePath+"validation/data",annFile = BasePath+"validation/labels.json", args=args, transform=transform, Live=Live)
    # Test currently disabled
    # Test = CocoDetection(root = BasePath+"test/data",annFile = BasePath+"test/labels.json", args=args, transform=transform)
    # Change here for test/validation
    return Train, Validation, 1


def samplingImages(dataset):
    print("sampled!")
    # Sample Visualization curretly ammended


    return


def Data_Driver(args, Live=False):
    BasePath = args.config["datapath"]
    Train, Validation, Test = constructData(BasePath, args, Live=Live)
    print(len(Validation))
    samplingImages(Validation)

    return Train, Validation, Test


