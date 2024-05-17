import torch
import torch.nn as nn
from glob import glob
import os
import numpy as np
import argparse
import cv2
# from unet import UNET
import albumentations as A
import torch.nn.functional as F
import math
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import pandas as pd

from utils import BCEDiceLoss
import utils as ut
from dataset import Dataset
from collections import OrderedDict

from src.kan_unet import KANU_Net
from src.unet import U_Net


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model train')
    parser.add_argument('--model', type=str, default='KANU_Net', choices=['KANU_Net', 'U_Net'])
    parser.add_argument('--dataset', type=str, default='BUSI', choices=['BUSI'])
    parser.add_argument('--fold', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--model_dir', type=str, default='experiences')
    parser.add_argument('--loss', type=str, default='bcediceloss')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--early_stopping_rounds', type=int, default=50)

    args = parser.parse_args()

    model_name = args.model
    fold = args.fold
    dataset = args.dataset
    model_dir = args.model_dir
    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.epochs
    img_size = args.img_size
    loss = args.loss
    gpu = args.gpu
    early_stopping = args.early_stopping_rounds

    if dataset == 'BUSI':
        
        num_classes = 1
        num_channels = 3

        DATA_DIR = r'data\splitted\train'
        DATA_DIR_TEST = r'data\splitted\test'
        BUS_DATA_PATH = r'data\BUS'

        imagesListTrain = []
        makskListTrain = []

        for idx in range(5):
            if idx == fold:
                imagesListValid = glob(f"{DATA_DIR}/split{fold}/images"+'/*.png')
                maskListValid = glob(f"{DATA_DIR}/split{fold}/masks"+'/*.png')
            else:

                for img in glob(f'{DATA_DIR}/split{idx}/images'+'/*.png'):
                    imagesListTrain.append(img)
                for msk in glob(f'{DATA_DIR}/split{idx}/masks'+'/*.png'):
                    makskListTrain.append(msk)

        imagesListTest = glob(os.path.join(DATA_DIR_TEST, 'images', '*.png'))
        makskListTest = glob(os.path.join(DATA_DIR_TEST, 'masks', '*.png'))

        BUS_test_images = glob(os.path.join(BUS_DATA_PATH, 'original', '*.png'))
        BUS_test_masks = glob(os.path.join(BUS_DATA_PATH, 'GT', '*.png'))

        print(f"Number of training images : {len(imagesListTrain)}, Number of training masks: {len(makskListTrain)}")
        print(f"Number of images of validation images : {len(imagesListValid)}, number of validation masks: {len(maskListValid)}")
        print(f"Number of testing images : {len(imagesListTest)}, Number of testing masks : {len(makskListTest)}")
        print(f"Number of testing images on BUS dataset: {len(BUS_test_images)}, Number of testing masks : {len(BUS_test_masks)}")

        num_masks_test = len(makskListTest)

        def mask_load_test(dir_path, imgs_list, masks_array):
            for i in range(len(imgs_list)):
                # tmp_img = Image.open(os.patzh.join(dir_path, imgs_list[i])).resize((256, 256))
                img = cv2.imread(os.path.join(dir_path, imgs_list[i]))
                img = cv2.resize(img, (224, 224))
                img = np.array(img)
                img = img.transpose(2, 0, 1)
                masks_array[i] = img[0,:,:]/255.0

            # Expand the dimensions of the arrays
            masks_array = np.expand_dims(masks_array, axis=3)
            return masks_array

        masksTest = np.zeros((num_masks_test, 224, 224))
        # masksTest = mask_load_test(os.path.join(DATA_DIR_TEST, 'masks'), makskListTest, masksTest)

        print("Test masks", masksTest.shape)

        transform = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Transpose(p=0.5),
                    #A.ElasticTransform(p=0.5),
                    #A.GridDistortion(p=0.5),
                    #A.OpticalDistortion(p=0.5),
                    #A.ShiftScaleRotate(p=0.5),
                ])

        # Create the dataloader for training and validation
        train_dataset = Dataset(imagesListTrain, makskListTrain, transform=transform)
        valid_dataset = Dataset(imagesListValid, maskListValid)
        test_dataset = Dataset(imagesListTest, makskListTest)
        bus_test_dataset = Dataset(BUS_test_images, BUS_test_masks)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        bus_test_loader = torch.utils.data.DataLoader(bus_test_dataset, batch_size=batch_size, shuffle=False)


    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    if model_name == 'U_Net':
        model = U_Net(n_channels=num_channels, n_classes=num_classes).to(DEVICE)
    elif model_name == 'KANU_Net':
        model = KANU_Net(n_channels=num_channels, n_classes=num_classes, device=DEVICE).to(DEVICE)


    # define the scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # define the optimizer and the scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-7)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs)

    BCE_dice_loss = ut.BCEDiceLoss().to(DEVICE)

    if not os.path.exists(os.path.join(model_dir, model_name, dataset)):
        os.makedirs(os.path.join(model_dir, model_name, dataset))

    # Run the training and validation for the specified number of epochs
    train_loss_list = []
    val_loss_list = []

    best_dice = 0
    trigger = 0
                                                                
    log = OrderedDict([
            ('epoch', []),
            ('loss', []),
            ('iou', []),
            ('dice', []),
            ('val_loss', []),
            ('val_iou', []),
            ('val_dice', []),
        ])

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch+1}/{num_epochs}")
        train_log = ut.train_fn(train_loader, model, optimizer, BCE_dice_loss, scaler)
        val_log = ut.val_fn(valid_loader, model,  BCE_dice_loss)
        # train_loss_list.append(train_loss)
        # val_loss_list.append(val_loss)

        print('loss %.4f - iou %.4f - dice %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f - val_SE %.4f - val_PC %.4f - val_F1 %.4f - val_SP %.4f - val_ACC %.4f'
                % (train_log['loss'], train_log['iou'], train_log['dice'], val_log['loss'], val_log['iou'], val_log['dice'], val_log['SE'],
                val_log['PC'], val_log['F1'], val_log['SP'], val_log['ACC']))

        scheduler.step(val_log['loss'])

        log['epoch'].append(epoch)
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['dice'].append(train_log['dice'])

        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv(f'{os.path.join(model_dir, model_name, dataset)}_{fold}.csv', index=False)

        trigger += 1

        if val_log['dice'] > best_dice:
            torch.save(model.state_dict(), f'{os.path.join(model_dir, model_name, dataset)}_{fold}.pth')
            best_dice = val_log['dice']
            print("=> saved best model. Best dice: {}".format(best_dice))
            trigger = 0

            # early stopping
            
            if trigger >= early_stopping:
                print("=> early stopping")
                break
            torch.cuda.empty_cache()

    print('testing...')
    print('from memory...')

    # Create a function to test the model on the test dataset, using the metrics we created above
    def test(model, test_loader, device, loss_fn):
        model.eval()

        avg_meters = {'loss': ut.AverageMeter(),
                    
                    'iou': ut.AverageMeter(),
                    'dice': ut.AverageMeter(),
                    'SE':ut.AverageMeter(),
                    'PC':ut.AverageMeter(),
                    'F1':ut.AverageMeter(),
                    'SP':ut.AverageMeter(),
                    'ACC':ut.AverageMeter()
                        }

        with torch.no_grad():
            for image, target in test_loader:
                image, target = image.to(device=device, dtype=torch.float), target.to(device=DEVICE, dtype=torch.float)
                output = model(image)
                test_loss = loss_fn(output, target).item()
                iou, dice, SE, PC, F1, SP, ACC = ut.iou_score(output, target)

                avg_meters['loss'].update(test_loss, image.size(0))
                avg_meters['iou'].update(iou, image.size(0))
                avg_meters['dice'].update(dice, image.size(0))
                avg_meters['SE'].update(SE, image.size(0))
                avg_meters['PC'].update(PC, image.size(0))
                avg_meters['F1'].update(F1, image.size(0))
                avg_meters['SP'].update(SP, image.size(0))
                avg_meters['ACC'].update(ACC, image.size(0))
                
        return (OrderedDict([('loss', avg_meters['loss'].avg),
                            ('iou', avg_meters['iou'].avg),
                            ('dice', avg_meters['dice'].avg),
                            ('SE', avg_meters['SE'].avg),
                            ('PC', avg_meters['PC'].avg),
                            ('F1', avg_meters['F1'].avg),
                            ('SP', avg_meters['SP'].avg),
                            ('ACC', avg_meters['ACC'].avg)
                            ]), avg_meters)
    test_log, dicr = test(model, test_loader, DEVICE, BCE_dice_loss)
    print('test_loss %.4f - test_iou %.4f - test_dice %.4f -test_SE %.4f - test_PC %.4f - test_F1 %.4f - test_SP %.4f - test_ACC %.4f'
                % (test_log['loss'], test_log['iou'], test_log['dice'], test_log['SE'],
                test_log['PC'], test_log['F1'], test_log['SP'], test_log['ACC']))
    if dataset == 'BUSI':
        bus_test_log, dicr= test(model, bus_test_loader, DEVICE, BCE_dice_loss)
        print('test_loss %.4f - test_iou %.4f - test_dice %.4f -test_SE %.4f - test_PC %.4f - test_F1 %.4f - test_SP %.4f - test_ACC %.4f'
                    % (bus_test_log['loss'], bus_test_log['iou'], bus_test_log['dice'], bus_test_log['SE'],
                    bus_test_log['PC'], bus_test_log['F1'], bus_test_log['SP'], bus_test_log['ACC']))


    print('from saved...')
    model.load_state_dict(torch.load(f'{os.path.join(model_dir, model_name, dataset)}_{fold}.pth',  map_location=torch.device('cuda')), strict=True)
    # Create a function to test the model on the test dataset, using the metrics we created above
    def test(model, test_loader, device, loss_fn):
        model.eval()

        avg_meters = {'loss': ut.AverageMeter(),
                    
                    'iou': ut.AverageMeter(),
                    'dice': ut.AverageMeter(),
                    'SE':ut.AverageMeter(),
                    'PC':ut.AverageMeter(),
                    'F1':ut.AverageMeter(),
                    'SP':ut.AverageMeter(),
                    'ACC':ut.AverageMeter()
                        }

        with torch.no_grad():
            for image, target in test_loader:
                image, target = image.to(device=device, dtype=torch.float), target.to(device=DEVICE, dtype=torch.float)
                output = model(image)
                test_loss = loss_fn(output, target).item()
                iou, dice, SE, PC, F1, SP, ACC = ut.iou_score(output, target)

                avg_meters['loss'].update(test_loss, image.size(0))
                avg_meters['iou'].update(iou, image.size(0))
                avg_meters['dice'].update(dice, image.size(0))
                avg_meters['SE'].update(SE, image.size(0))
                avg_meters['PC'].update(PC, image.size(0))
                avg_meters['F1'].update(F1, image.size(0))
                avg_meters['SP'].update(SP, image.size(0))
                avg_meters['ACC'].update(ACC, image.size(0))
                
        return (OrderedDict([('loss', avg_meters['loss'].avg),
                            ('iou', avg_meters['iou'].avg),
                            ('dice', avg_meters['dice'].avg),
                            ('SE', avg_meters['SE'].avg),
                            ('PC', avg_meters['PC'].avg),
                            ('F1', avg_meters['F1'].avg),
                            ('SP', avg_meters['SP'].avg),
                            ('ACC', avg_meters['ACC'].avg)
                            ]), avg_meters)
    test_log, dicr = test(model, test_loader, DEVICE, BCE_dice_loss)
    print('test_loss %.4f - test_iou %.4f - test_dice %.4f -test_SE %.4f - test_PC %.4f - test_F1 %.4f - test_SP %.4f - test_ACC %.4f'
                % (test_log['loss'], test_log['iou'], test_log['dice'], test_log['SE'],
                test_log['PC'], test_log['F1'], test_log['SP'], test_log['ACC']))

    if dataset == 'BUSI':
        bus_test_log, dicr= test(model, bus_test_loader, DEVICE, BCE_dice_loss)
        print('test_loss %.4f - test_iou %.4f - test_dice %.4f -test_SE %.4f - test_PC %.4f - test_F1 %.4f - test_SP %.4f - test_ACC %.4f'
                    % (bus_test_log['loss'], bus_test_log['iou'], bus_test_log['dice'], bus_test_log['SE'],
                    bus_test_log['PC'], bus_test_log['F1'], bus_test_log['SP'], bus_test_log['ACC']))
        



    

