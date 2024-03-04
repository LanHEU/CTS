import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
import torchvision.utils as vutils
import cv2 

import numpy as np

def normalize_pixel_value(pixel_value, min_value, max_value):
    if{min_value == max_value}:
        return pixel_value
    else:
        normalized_value = (pixel_value - min_value) / (max_value - min_value )
        normalized_value = (normalized_value * 2) - 1
        return normalized_value

def picture2nor(pixel_values):
    pixel_values = pixel_values.numpy()
    # 计算像素值的最大值和最小值
    min_value = np.min(pixel_values)
    max_value = np.max(pixel_values)

    # 对每个像素值进行归一化
    normalized_pixel_values = np.array([[normalize_pixel_value(pixel_value, min_value, max_value) for pixel_value in row] for row in pixel_values])
    normalized_pixel_values = torch.from_numpy(normalized_pixel_values)
    return normalized_pixel_values



class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform, test_flag=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('_')[3]
                    datapoint[seqtype] = os.path.join(root, f)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            path=filedict[seqtype]
            out.append(torch.tensor(nib_img.get_fdata()))
        out = torch.stack(out)
        if self.test_flag:
            image=out
            image = image[..., 8:-8, 8:-8]     #crop to a size of (224, 224)
            if self.transform:
                image = self.transform(image)
            return (image, image, path)
        else:

            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            image = image[..., 8:-8, 8:-8]      #crop to a size of (224, 224)
            label = label[..., 8:-8, 8:-8]
            label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
            if self.transform:
                state = torch.get_rng_state()
                image = self.transform(image)
                torch.set_rng_state(state)
                label = self.transform(label)
            return (image, label, path)

    def __len__(self):
        return len(self.database)

class BRATSDataset3D(torch.utils.data.Dataset):
    def __init__(self, directory, transform, test_flag=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    # print (f)
                    seqtype = f.split('.')[0]
                    datapoint[seqtype] = os.path.join(root, f)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)
    
    def __len__(self):
        return len(self.database) * 155

    def __getitem__(self, x):
        out = []
        n = x // 155
        slice = x % 155
        filedict = self.database[n]
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            path=filedict[seqtype]
            o = torch.tensor(nib_img.get_fdata())[:,:,slice]
            # o = picture2nor(o)
            # if seqtype != 'seg':
            #     o = o / o.max()
            out.append(o)
        out = torch.stack(out)
        if self.test_flag:
            image=out
            # image = image[..., 8:-8, 8:-8]     #crop to a size of (224, 224)
            image[0,:,:] = picture2nor(out[0])
            image[1,:,:] = picture2nor(out[1])
            image[2,:,:] = picture2nor(out[2])
            image[3,:,:] = picture2nor(out[3])
            
            
            label = out[-1, ...][None, ...]
            label = torch.squeeze(label)
            label_one_hot_1 =  torch.zeros(label.size())
            label_one_hot_2 =  torch.zeros(label.size())
            label_one_hot_3 =  torch.zeros(label.size())
            label_one_hot_4 =  torch.zeros(label.size())
            
            for i in range(label.size(0)):
                for j in range(label.size(1)):
                    if label[i, j] == 1:
                        label_one_hot_1[i, j] = 1
                    if label[i, j] == 2:
                        label_one_hot_2[i, j] = 1
                    if label[i, j] == 3:
                        label_one_hot_3[i, j] = 1
                    if label[i, j] == 4:
                        label_one_hot_4[i, j] = 1
            label_one_hot_1 = torch.unsqueeze(label_one_hot_1, dim=0)
            label_one_hot_2 = torch.unsqueeze(label_one_hot_2, dim=0)
            label_one_hot_3 = torch.unsqueeze(label_one_hot_3, dim=0)
            label_one_hot_4 = torch.unsqueeze(label_one_hot_4, dim=0)
            label_one_hot = torch.cat((label_one_hot_1, label_one_hot_2, label_one_hot_3, label_one_hot_4), dim=0)
            label = label_one_hot
            if self.transform:
                image = self.transform(image)
            return (image, image, path.split('.nii')[0] + "_slice" + str(slice)+ ".nii") # virtual path
        else:
            
            # #得到LOG下的滤波
            # out_log = out[:-1, ...]
            # out_log = out_log.numpy()
            # # out_log = out[:-1, ...]
            # out_log[0,:,:] = cv2.Laplacian(out_log[0], cv2.CV_16S, ksize=5)
            # out_log[1,:,:] = cv2.Laplacian(out_log[1], cv2.CV_16S, ksize=5)
            # out_log[2,:,:] = cv2.Laplacian(out_log[2], cv2.CV_16S, ksize=5)
            # out_log[3,:,:] = cv2.Laplacian(out_log[3], cv2.CV_16S, ksize=5)
            # out_log = torch.tensor(out_log)
            
            # out = torch.tensor(out)
            image = out[:-1, ...]
            image[0,:,:] = picture2nor(out[0])
            image[1,:,:] = picture2nor(out[1])
            image[2,:,:] = picture2nor(out[2])
            image[3,:,:] = picture2nor(out[3])
            
            
            label = out[-1, ...][None, ...]
            label = torch.squeeze(label)
            label_one_hot_1 =  torch.zeros(label.size())
            label_one_hot_2 =  torch.zeros(label.size())
            label_one_hot_3 =  torch.zeros(label.size())
            label_one_hot_4 =  torch.zeros(label.size())
            
            for i in range(label.size(0)):
                for j in range(label.size(1)):
                    if label[i, j] == 1:
                        label_one_hot_1[i, j] = 1
                    if label[i, j] == 2:
                        label_one_hot_2[i, j] = 1
                    if label[i, j] == 3:
                        label_one_hot_3[i, j] = 1
                    if label[i, j] == 4:
                        label_one_hot_4[i, j] = 1
            label_one_hot_1 = torch.unsqueeze(label_one_hot_1, dim=0)
            label_one_hot_2 = torch.unsqueeze(label_one_hot_2, dim=0)
            label_one_hot_3 = torch.unsqueeze(label_one_hot_3, dim=0)
            label_one_hot_4 = torch.unsqueeze(label_one_hot_4, dim=0)
            label_one_hot = torch.cat((label_one_hot_1, label_one_hot_2, label_one_hot_3, label_one_hot_4), dim=0)
            label = label_one_hot
                    
            
            # label = label.numpy()
            # image = image[..., 8:-8, 8:-8]      #crop to a size of (224, 224)
            # label = label[..., 8:-8, 8:-8]
            # label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
            if self.transform:
                state = torch.get_rng_state()
                image = self.transform(image)
                # out_log = self.transform(out_log)
                torch.set_rng_state(state)
                label = self.transform(label)
            return (image, label,  path.split('.nii')[0] + "_slice" + str(slice)+ ".nii") # virtual path



