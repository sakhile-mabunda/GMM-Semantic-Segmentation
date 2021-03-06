from torchvision import transforms

from data_loader.segmentation_dataset import SegmentationDataset
from data_loader.transform import Rescale, RandomHorizontalFlip,ToTensor
from trainer import Trainer
from predict import *
from unet import *
import all_models
from util.logger import Logger


train_images = r'dataset/cityspaces/images/train'
val_images = r'dataset/cityspaces/images/val'
test_images = r'dataset/cityspaces/images/test'

train_labled = r'dataset/cityspaces/labeled/train'
val_labeled = r'dataset/cityspaces/labeled/val'
test_labeled = r'dataset/cityspaces/labeled/test'

if __name__ == '__main__':
    model_name = "unet_vgg16"
    device = 'cuda'
    batch_size = 4
    n_classes = 2
    num_epochs = 100
    image_axis_minimum_size = 200
    pretrained = True
    fixed_feature = False

    logger = Logger(model_name=model_name, data_name='example')

    ### Loader
    compose = transforms.Compose([
        Rescale(image_axis_minimum_size),
        RandomHorizontalFlip(0.7),
        ToTensor()
         ])

    train_datasets = SegmentationDataset(train_images, train_labled, n_classes, compose)
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, drop_last=True)

    val_datasets = SegmentationDataset(val_images, val_labeled, n_classes, compose)
    val_loader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True, drop_last=True)

    test_datasets = SegmentationDataset(test_images, test_labeled, n_classes, compose)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True, drop_last=True)

    ### Model
    model = all_models.model_from_name[model_name](n_classes, batch_size,
                                                   pretrained=pretrained,
                                                   fixed_feature=fixed_feature)
    model.to(device)

    ##Load model
    ##please check the foloder: (.segmentation/test/runs/models)
    # logger.load_model(model, 'epoch_15')

    ### Optimizers
    if pretrained and fixed_feature: #fine tunning
        params_to_update = model.parameters()
        print("Params to learn:")
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
        optimizer = torch.optim.Adadelta(params_to_update)
    else:
        optimizer = torch.optim.Adadelta(model.parameters())

    ### Train
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    trainer = Trainer(model, optimizer, logger, num_epochs, train_loader, val_loader)
    trainer.train()


    #### Writing the predict result.
    predict(model, r'dataset/cityspaces/input.png',
             r'dataset/cityspaces/output.png')


