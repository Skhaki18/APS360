from email.mime import image
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from PIL import Image


# Visualize Data Transformations

BasePath = "/Users/samirkhaki/fiftyone/coco-2017/validation/data/000000001532.jpg"
image = Image.open(BasePath)
image2 = Image.open(BasePath)
# 
transform = transforms.Compose([transforms.RandomRotation(5)])
transform2 = transforms.Compose([transforms.RandomRotation(0)])

# dataset = datasets.ImageFolder(BasePath, transform=transform)
imageNew = transform(image)


fig, (ax1,ax2) = plt.subplots(1,2)

ax1.imshow(image2)
ax1.set(title='Base Image')
ax2.imshow(imageNew)
ax2.set(title='Rotated Image')
plt.show()

# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)