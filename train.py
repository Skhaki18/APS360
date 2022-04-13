# from util.utils import collate_fn
from data import Data_Driver
# from util.engine import train_one_epoch, evaluate
import torch
import torchvision
from models import *
import matplotlib.pyplot as plt
import util.utils as utils
import numpy as np
from util.engine import train_one_epoch, evaluate


# Base batch shuffling
def batch_and_shuffle(minibatch, trainData, trainTarget):

    #gets number of required batches
    required_batches = trainTarget.size / minibatch

    train_batch = []
    target_batch = []

    #shuffle the data using functions from loadData()
    randIndx = np.arange(len(trainTarget))
    np.random.shuffle(randIndx)
    data_temp, target_temp = trainData[randIndx], trainTarget[randIndx]

    #creates the batches
    for b in range(int(required_batches)):
        
        #b*minibatch:(b+1)*minibatch ensures each batch is has 'minibatch' number of data points
        train_batch.append(data_temp[b*minibatch:(b+1)*minibatch])

        target_batch.append(target_temp[b*minibatch:(b+1)*minibatch])

    return train_batch, target_batch, required_batches


# Argmaxin predictions --- Not used in final implementation
def accruacy(pred, labels):

    class_choice_pred = np.argmax(pred)
    class_choice_labels = np.argmax(labels)

    correct = 0

    for i in range(len(class_choice_pred)):
        if class_choice_pred[i] == class_choice_labels[i]:
            correct+=1

    return correct/len(class_choice_labels)

# Handles custom and Pretrained model loading

def Livetrain(args, Live=False):
    
    Train, Validation, Test = Data_Driver(args, Live=Live)

    model, DEVICE = get_model(args.config["model"],classes =args.config["classes"],trainstat=args.config["trainstat"])

    print("Replacing with Fine TUned Model!")
    # Supports pretrained and Fine TUning Replaceents !!!
    # model = torch.jit.load('FasterRCNN_Resnet50_FT.pt')


    return model, Validation, DEVICE


# Training Procedure partly supported by PYTORCH ENGINE!

def train(args):
    
    Train, Validation, Test = Data_Driver(args)

    model, DEVICE = get_model(args.config["model"],classes =args.config["classes"],trainstat=args.config["trainstat"])
    
    
    torch.manual_seed(1)
    indices = torch.randperm(len(Train)).tolist()

    train_split = 500
    TrainDataset = torch.utils.data.Subset(Train, indices[:train_split])


    data_loader = torch.utils.data.DataLoader(
    TrainDataset, batch_size=4, shuffle=True,
    collate_fn=utils.collate_fn)

    data_loader_Valid = torch.utils.data.DataLoader(
    Validation, batch_size=4, shuffle=True,
    collate_fn=utils.collate_fn)

    # model = torch.jit.load('FasterRCNN_Resnet50_FT.pt')

    model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)


    num_epochs = 10
    # model.train()
    # EpochValid = evaluate(model, data_loader_Valid, device=DEVICE)
    # print(EpochValid.coco_eval["bbox"])
    # ValidFile = open("FasterRCNN_Resnet50_FT_Valid.txt", 'a')
    # ValidFile.write("\n")
    # ValidFile.write(str(EpochValid.coco_eval["bbox"]))
    # return
    for epoch in range(num_epochs):
        # training for one epoch
        EpochUpdates = train_one_epoch(model, optimizer, data_loader, DEVICE, epoch, print_freq=1)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
      

        EpochValid = evaluate(model, data_loader_Valid, device=DEVICE)
        # evaluate(model, data_loader_Valid, device=DEVICE)

        TrainFile = open("FasterRCNN_Mobile_FT_Train.txt", 'a')
        TrainFile.write("\n")
        TrainFile.write(str(EpochUpdates))


        ValidFile = open("FasterRCNN_Resnet50_FT_Valid.txt", 'a')
        ValidFile.write("\n")
        ValidFile.write(str(EpochValid))

    torch.save(model.state_dict(), "FasterRCNN_Resnet50_FT")
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save('FasterRCNN_Mobile_FT.pt') # Save

    return model, Validation, DEVICE


    # Previous Iteration on Train Epochs

    """
    data_loader = torch.utils.data.DataLoader(
    Validation,batch_size=args.config["mini_batch_size"], shuffle=True, collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        Validation, batch_size=args.config["mini_batch_size"], shuffle=True, collate_fn=collate_fn)

    TrainingBatchSample = next(iter(data_loader))
    print(TrainingBatchSample[0])

    # TrainingBatchSample = next(iter(TrainLoader))
    Sampler = 32
    fig = plt.figure(figsize=(15, 2))

    for i in range(0, Sampler):
        ax = plt.subplot(2, int(Sampler/2), i + 1)

        image = TrainingBatchSample[0][i].cpu().numpy().transpose((1, 2, 0))
        # label = ClassLabes[TrainingBatchSample[1][i]]
        plt.imshow(image)
        # plt.title(label)
        plt.axis("off")


    plt.show()
    # return model, Validation, DEVICE#, data_loader
    train_loss = []
    valid_loss = []
    #test_loss = []
    train_acc = []
    valid_acc = []
    #test_acc =[]

    TrainingBatchSample = next(iter(data_loader))
    return
    # print(TrainingBatchSample)

    ##is base the same as labels?????
    #image,_, base = ValidationSet[:]
    # trainData, trainTarget, train_base = Train[:] 
    # trainData = trainData.to(DEVICE)
    # trainTarget = trainTarget.to(DEVICE)

    # validData, validTarget, valid_base = Validation[:]
    # validData = validData.to(DEVICE)
    # validTarget = validTarget.to(DEVICE)

    # testData, testTarget = Test[:]
    # testData = testData.to(DEVICE)
    # testTarget = testTarget.to(DEVICE)
    """
    """

    epochs = args.config["max_epoch"]

    lr = args.config["init_lr"]


    beta = args.config["beta"]

    # if args.config["loss"] == "cross_entropy":
    #     criterion = nn.CrossEntropyLoss()

    if args.config["optim_method"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif args.config["optim_method"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, beta1=beta)

    # learning rate updates -step
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)


    for epoch in range(epochs):
        # training for one epoch
        print(epoch)
        train_one_epoch(model, optimizer, data_loader, DEVICE, epoch, print_freq=1)
        print("STEP")
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=DEVICE)


    """

    #if(args.config["trainstat"]):
        #return model, Validation, DEVICE

    #train_loader = torch.utils.data.DataLoader(Train, batch_size=model.mini_batch_size, shuffle=True)
    #valid_loader = torch.utils.data.DataLoader(Validation, batch_size=model.mini_batch_size, shuffle=True)
    #test_loader = torch.utils.data.DataLoader(Test, batch_size=model.mini_batch_size, shuffle=True)
""" for ep in epochs:

        train_batch, target_batch, required_batches = batch_and_shuffle(args.config["mini_batch_size"], trainData, trainTarget)
        
        for b in range(required_batches):
        #for data, labels in iter(train_loader):    

            #might have to be done elemnt wise
            images = train_batch[b].to(DEVICE)
            
            #can take batch
            detections = model(images)

            loss = criterion(detections, target_batch[b])

            loss.backward()
            optimizer.step()
            # zero the parameter gradients
            optimizer.zero_grad()

        #Get train and validation loss per epoch
        pred = model(trainData)
        train_loss.append(criterion(pred, trainTarget))
        train_acc.append(accruacy(pred, trainTarget))

        pred = model(validData)
        valid_loss.append(criterion(pred, validTarget))
        valid_acc.append(accruacy(pred, validTarget))

        # pred = model(testData)
        # test_loss.append(criterion(pred, testTarget))

        

        return epochs, train_loss, valid_loss, train_acc, valid_acc#, test_loss, test_acc


"""
   