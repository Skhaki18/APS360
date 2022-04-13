import numpy as np
import cv2
import matplotlib.pyplot as plt

#We use standard cv2 bounding box techniques

def AnnotationLive(detection, image, ids, labels):


    for i in range(0, len(detection["boxes"])):
        
        confidence = detection["scores"][i]
        if confidence > 0.5:
            idx = int(detection["labels"][i])
            Set = [3, 8, 6, 4, 1]
            if idx in Set:
                
      
                box = detection["boxes"][i]
                temp = np.round(box[0:4].detach()).detach().numpy().astype("int")
               
                # print(temp)
                (startX, startY, endX, endY)  = temp
               
                print(idx)
                cv2.rectangle(image, (startX, startY), (endX, endY),
                    (255,0,0), 2)


    for i in labels:
        if(i["iscrowd"] == 1): continue
        Set = [3, 8, 6, 4, 1]
        if(i["category_id"] in Set):
            
            (startX, startY, endX, endY) = i["bbox"]
           
    
            cv2.rectangle(image, (round(startX), round(startY)), (round(startX+endX), round(startY+endY)), (0,255,0), 2)

    # Write to Disk Feature currently off
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)       
    # image = (image*255).astype(np.uint8)
    # print(image)
    # cv2.imwrite('Output.jpg', image)
    plt.imshow(image)
    plt.show()
    return


def DynamicAnnotation(detection, image, ids):

    for i in range(0, len(detection["boxes"])):
    
        confidence = detection["scores"][i]
        if confidence > 0.5:
            idx = int(detection["labels"][i])
            print(ids)
            if idx in ids:
                box = detection["boxes"][i].detach().cpu().numpy()
                (startX, startY, endX, endY) = box.astype("int")
                
                cv2.rectangle(image, (startX, startY), (endX, endY),
                    [255,0,0], 2)
        
    return image
