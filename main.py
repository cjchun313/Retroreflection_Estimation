import argparse
import numpy as np

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch import nn

from torch.utils.data import DataLoader

from evaluation import compute_iou
from models.classification_modern_model import cnn_model
from models.segmentation_unet import UNet
from data_loader import ImagevDatasetForClassi, ImagevDatasetForSegment

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

CLASSI_MODEL_PATH = '../pth/classi/classi_'
SEGMENT_MODEL_PATH = '../pth/segment/segment_'

def train(model, train_loader, optimizer):
    model.train()
    correct = 0

    criterion1 = nn.CrossEntropyLoss().to(DEVICE)
    criterion2 = nn.MSELoss().to(DEVICE)

    train_loss = 0
    for batch_idx, samples in enumerate(train_loader):
        data, target1, target2 = samples
        #print(data.shape, target.shape)

        data = data.to(DEVICE)
        target1 = target1.to(DEVICE)
        target2 = target2.to(DEVICE)

        output = model(data)
        #loss = criterion(m(output), target)
        loss1 = criterion1(output[:,0:2], target1)
        loss2 = criterion2(output[:,2], target2)
        loss = loss1 + loss2
        train_loss += loss.item()

        classi_output = output[:, 0:2]
        prediction = classi_output.max(1, keepdim=True)[1]
        correct += prediction.eq(target1.view_as(prediction)).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Batch Index: {}\tLoss: {:.6f}\tLoss1: {:.6f}\tLoss2: {:.6f}'.format(batch_idx, loss.item(), loss1.item(), loss2.item()))
    train_loss /= len(train_loader.dataset)
    train_acc = 100. * correct / len(train_loader.dataset)

    return train_loss, train_acc

def train2(model, train_loader, optimizer):
    model.train()
    train_loss = 0
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    for batch_idx, samples in enumerate(train_loader):
        data, target = samples
        #print(data.shape, target.shape)

        data = data.to(DEVICE)
        target = target.to(DEVICE)

        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Batch Index: {}\tLoss: {:.6f}'.format(batch_idx, loss.item()))
    train_loss /= len(train_loader.dataset)

    return train_loss

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    criterion1 = nn.CrossEntropyLoss().to(DEVICE)
    criterion2 = nn.MSELoss().to(DEVICE)

    with torch.no_grad():
        for samples in test_loader:
            data, target1, target2 = samples

            data = data.to(DEVICE)
            target1 = target1.to(DEVICE)
            target2 = target2.to(DEVICE)

            output = model(data)
            loss1 = criterion1(output[:, 0:2], target1)
            loss2 = criterion2(output[:, 2], target2)
            loss = loss1 + loss2
            test_loss += loss.item()

            classi_output = output[:, 0:2]
            prediction = classi_output.max(1, keepdim=True)[1]
            correct += prediction.eq(target1.view_as(prediction)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    return test_loss, test_acc

def evaluate2(model, test_loader):
    model.eval()
    test_loss = 0
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    total_ious = []
    with torch.no_grad():
        for samples in test_loader:
            data, target = samples

            data = data.to(DEVICE)
            target = target.to(DEVICE)

            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

            ious = compute_iou(output, target)
            total_ious.append(ious)

    test_loss /= len(test_loader.dataset)
    total_ious = np.mean(np.array(total_ious), axis=0)

    return test_loss, total_ious


def save_model(modelpath, model, optimizer, scheduler):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(state, modelpath)

    print('model saved')


def load_model(modelpath, model, optimizer=None, scheduler=None):
    state = torch.load(modelpath, map_location=torch.device(DEVICE))
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])

    print('model loaded')


def main(args):
    # train
    if args.mode == 'train':
        train_dataset = ImagevDatasetForClassi(mode='train')
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=0)

        val_dataset = ImagevDatasetForClassi(mode='val')
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        print(train_dataloader, val_dataloader)

        modelpath = CLASSI_MODEL_PATH
        if args.model == 'resnet18':
            modelpath += 'resnet18.pth'
        elif args.model == 'vgg16':
            modelpath += 'vgg16.pth'
        elif args.model == 'alexnet':
            modelpath += 'alexnet.pth'
        elif args.model == 'mobilenet':
            modelpath += 'mobilenet.pth'
        elif args.model == 'shufflenet':
            modelpath += 'shufflenet.pth'

        model = cnn_model(args.model)
        if torch.cuda.device_count() > 1:
            print('multi gpu used!')
            model = nn.DataParallel(model)
        model = model.to(DEVICE)

        # set optimizer
        optimizer = Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=args.lr)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
        #modelpath = CLASSI_MODEL_PATH
        #load_model(modelpath, model, optimizer, scheduler)

        acc_prev = 0.0
        for epoch in range(args.epoch):
            # train set
            loss, acc = train(model, train_dataloader, optimizer)
            # validate set
            val_loss, val_acc = evaluate(model, val_dataloader)

            print('Epoch:{}\tTrain Loss:{:.6f}\tTrain Acc:{:2.4f}'.format(epoch, loss, acc))
            print('Val Loss:{:.6f}\tVal Acc:{:2.4f}'.format(val_loss, val_acc))

            if val_acc > acc_prev:
                acc_prev = val_acc
                #modelpath = '../pth/20200726/model-{:d}-{:.6f}-{:2.4f}.pth'.format(epoch, val_loss, val_acc)
                save_model(modelpath, model, optimizer, scheduler)

            # scheduler update
            scheduler.step()
    # evaluate
    elif args.mode == 'evaluate':
        val_dataset = ImagevDatasetForClassi(mode='val')
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        print(val_dataloader)

        model = cnn_model(args.model)
        if torch.cuda.device_count() > 1:
            print('multi gpu used!')
            model = nn.DataParallel(model)
        model = model.to(DEVICE)

        modelpath = CLASSI_MODEL_PATH
        if args.model == 'resnet18':
            modelpath += 'resnet18.pth'
        elif args.model == 'vgg16':
            modelpath += 'vgg16.pth'
        elif args.model == 'alexnet':
            modelpath += 'alexnet.pth'
        elif args.model == 'mobilenet':
            modelpath += 'mobilenet.pth'
        elif args.model == 'shufflenet':
            modelpath += 'shufflenet.pth'
        load_model(modelpath, model)

        test_loss, test_acc = evaluate(model, val_dataloader)
        print('Test Loss:{:.6f}\tTest Acc:{:2.4f}'.format(test_loss, test_acc))

def main2(args):
    # train
    if args.mode == 'train':
        train_dataset = ImagevDatasetForSegment(mode='train')
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=0)

        val_dataset = ImagevDatasetForSegment(mode='val')
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        print(train_dataloader, val_dataloader)

        model = UNet()
        if torch.cuda.device_count() > 1:
            print('multi gpu used!')
            model = nn.DataParallel(model)
        model = model.to(DEVICE)

        modelpath = SEGMENT_MODEL_PATH
        modelpath += 'unet.pth'
        # set optimizer
        optimizer = Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=args.lr)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
        # modelpath = CLASSI_MODEL_PATH
        # load_model(modelpath, model, optimizer, scheduler)

        iou_prev = 0.0
        for epoch in range(args.epoch):
            # train set
            loss = train2(model, train_dataloader, optimizer)
            train_loss, train_ious = evaluate2(model, train_dataloader)
            train_mean_ious = np.mean(train_ious)
            # validate set
            val_loss, val_ious = evaluate2(model, val_dataloader)
            val_mean_iou = np.mean(val_ious)

            print('Epoch:{}\tTrain Loss:{:.6f}'.format(epoch, loss))
            print('Train Loss:{:.6f}\tTrain Mean IoU:{:2.4f}\tTrain IoU1:{:2.4f}\tTrain IoU2:{:2.4f}\tTrain IoU3:{:2.4f}'.format(train_loss, train_mean_ious, train_ious[0], train_ious[1], train_ious[2]))
            print('Val Loss:{:.6f}\tVal Mean IoU:{:2.4f}\tVal IoU1:{:2.4f}\tVal IoU2:{:2.4f}\tVal IoU3:{:2.4f}'.format(val_loss, val_mean_iou, val_ious[0], val_ious[1], val_ious[2]))

            if val_mean_iou > iou_prev:
                iou_prev = val_mean_iou
                # modelpath = '../pth/20200726/model-{:d}-{:.6f}-{:2.4f}.pth'.format(epoch, val_loss, val_acc)
                save_model(modelpath, model, optimizer, scheduler)

            # scheduler update
            scheduler.step()
    # evaluate
    elif args.mode == 'evaluate':
        val_dataset = ImagevDatasetForSegment(mode='val')
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        print(val_dataloader)

        model = UNet()
        if torch.cuda.device_count() > 1:
            print('multi gpu used!')
            model = nn.DataParallel(model)
        model = model.to(DEVICE)

        modelpath = SEGMENT_MODEL_PATH
        modelpath += 'unet.pth'
        load_model(modelpath, model)

        val_loss, val_ious = evaluate2(model, val_dataloader)
        val_mean_iou = np.mean(val_ious)
        print('Val Loss:{:.6f}\tVal Mean IoU:{:2.4f}\tVal IoU1:{:2.4f}\tVal IoU2:{:2.4f}\tVal IoU3:{:2.4f}'.format(val_loss, val_mean_iou, val_ious[0], val_ious[1], val_ious[2]))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch',
        default=240,
        type=int)
    parser.add_argument(
        '--epoch',
        help='the number of training iterations',
        default=50,
        type=int)
    parser.add_argument(
        '--lr',
        help='learning rate',
        default=0.001,
        type=float)
    parser.add_argument(
        '--shuffle',
        help='True, or False',
        default=True,
        type=bool)
    parser.add_argument(
        '--mode',
        help='train, or evaluate',
        default='train',
        type=str)
    parser.add_argument(
        '--model',
        help='resnet18, vgg16, alexnet, mobilenet, or shufflenet',
        default='resnet18',
        type=str)
    parser.add_argument(
        '--type',
        help='classification, or segmentation',
        default='classification',
        type=str)

    args = parser.parse_args()
    print(args)

    if args.type == 'classification':
        main(args)
    elif args.type == 'segmentation':
        main2(args)