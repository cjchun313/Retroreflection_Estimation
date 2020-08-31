import pandas as pd
import glob
import tqdm
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

class ImagevDatasetForClassi(Dataset):
    def __init__(self, mode='train', transform=None, width=640, height=480):
        self.mode = mode
        self.transform = transform
        self.width = width
        self.height = height

        path = '../db/'
        if self.mode == 'train':
            path += 'train/'
        else:
            path += 'val/'

        # positive files
        self.positive_imgfiles = sorted(glob.glob(path + '*/ok/*.csv'))
        #self.positive_hfiles = sorted(glob.glob(path + '*/npy/*.npy'))

        # negative files
        self.negative_imgfiles = sorted(glob.glob(path + '*/not_ok/*.csv'))

        self.x_data, self.y_data = [], []
        for imgfile in tqdm.tqdm(self.positive_imgfiles):
            npyfile = imgfile.replace('/ok', '/npy')
            npyfile = npyfile.replace('csv', 'npy')
            df_org = pd.read_csv(imgfile)
            ok_h = np.load(npyfile)
            self.x_data.append(df_org.values[:,1:641])
            self.y_data.append([1, ok_h / 480.0])

        for imgfile in tqdm.tqdm(self.negative_imgfiles):
            df_org = pd.read_csv(imgfile)
            self.x_data.append(df_org.values[:,1:641])
            self.y_data.append([0, 0])

        self.x_data = np.array(self.x_data)
        self.y_data = np.array(self.y_data)
        print(self.x_data.shape, self.y_data.shape)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        sample = torch.FloatTensor(self.x_data[idx]).view(1, self.height, self.width), torch.as_tensor(self.y_data[idx][0], dtype=torch.long), torch.as_tensor(self.y_data[idx][1])
        if self.transform:
            sample = self.transform(sample)

        return sample

class ImagevDatasetForSegment(Dataset):
    def __init__(self, mode='train', transform=None, width=640, height=192):
        self.mode = mode
        self.transform = transform
        self.width = width
        self.height = height

        path = '../db/'
        if self.mode == 'train':
            path += 'train/'
        else:
            path += 'val/'

        # positive files
        self.positive_imgfiles = sorted(glob.glob(path + '*/ok/*.csv'))
        self.positive_lblfiles = sorted(glob.glob(path + '*/labeled/*.png'))

        self.x_data = []
        for imgfile in tqdm.tqdm(self.positive_imgfiles):
            npyfile = imgfile.replace('/ok', '/npy')
            npyfile = npyfile.replace('csv', 'npy')
            df_org = pd.read_csv(imgfile)
            ok_h = np.load(npyfile)
            self.x_data.append(df_org.values[ok_h:ok_h+self.height, 1:641])

        self.y_data = []
        for lblfile in tqdm.tqdm(self.positive_lblfiles):
            img = np.array(cv2.imread(lblfile, cv2.IMREAD_GRAYSCALE))
            #img = img.reshape(1, self.height, self.width)
            self.y_data.append(img)

        self.x_data = np.array(self.x_data)
        self.y_data = np.array(self.y_data)
        print(self.x_data.shape, self.y_data.shape)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        sample = torch.FloatTensor(self.x_data[idx]).view(1, self.height, self.width), torch.LongTensor(self.y_data[idx]).view(self.height, self.width)
        if self.transform:
            sample = self.transform(sample)

        return sample


class ImagevDatasetForRetroEsti(Dataset):
    def __init__(self, mode='val2', transform=None, width=640, height=480):
        self.mode = mode
        self.transform = transform
        self.width = width
        self.height = height

        path = '../db/'
        path += 'val2/'

        # iamge files
        self.imgfiles = sorted(glob.glob(path + 'pixels/*.csv'))

        self.x_data, self.y_data = [], []
        for imgfile in tqdm.tqdm(self.imgfiles):
            jpgfile = imgfile.replace('pixels', 'original')
            jpgfile = jpgfile.replace('csv', 'jpg')

            df_org = pd.read_csv(imgfile)
            self.x_data.append(df_org.values[:,1:641])
            img = np.array(cv2.imread(jpgfile))
            self.y_data.append(img)

        self.x_data = np.array(self.x_data)
        self.y_data = np.array(self.y_data)
        print(self.x_data.shape, self.y_data.shape)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        sample = torch.FloatTensor(self.x_data[idx]).view(1, self.height, self.width), torch.LongTensor(self.y_data[idx]).view(3, self.height, self.width)
        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    '''
    train_dataset = ImagevDatasetForClassi(mode='val')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=0)
    for batch_idx, samples in enumerate(train_loader):
        data, target1, target2 = samples
        print(data.shape, target1.shape, target2.shape)

        break
    '''

    train_dataset = ImagevDatasetForRetroEsti(mode='val2')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    for batch_idx, samples in enumerate(train_loader):
        data, target = samples
        print(data.shape, target.shape)
        print(torch.max(target), torch.min(target))

        break
