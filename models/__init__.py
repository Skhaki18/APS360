from models.pretrain import pretrainedResNet, pretrainedMobileNetLarge,pretrainedMobileLowRes, fine_tuningResNet, fine_tuningMobileNet

def get_model(model_type, **kwargs):
    models = {
        # 'ResNet': pretrainedResNet,
        'resnet50': pretrainedResNet,
        'mobile1': pretrainedMobileNetLarge,
        'mobile2': pretrainedMobileLowRes,
        'finetuneRN': fine_tuningResNet,
        'finetuneMobile': fine_tuningMobileNet
    }
    return models[model_type](**kwargs)