import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import random


def is_image_file(filename):
    """Check if the filename has a valid image extension."""
    return filename.lower().endswith(('jpeg', 'jpg', 'png', 'gif'))

class DataLoaderTrain(Dataset):
    def __init__(self, dir_path, patch_size):
        super(DataLoaderTrain, self).__init__()
        
        assert os.path.exists(dir_path)
        
        input_files = sorted(os.listdir(os.path.join(dir_path, 'input')))
        target_files = sorted(os.listdir(os.path.join(dir_path, 'gt')))

        self.input_filenames = [os.path.join(dir_path, 'input', x) for x in input_files if is_image_file(x)]
        self.target_filenames = [os.path.join(dir_path, 'gt', x) for x in target_files if is_image_file(x)]

        self.sizex = len(self.target_filenames)  # get the size of target

        self.patch_size = patch_size
    
    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        input_path = self.input_filenames[index_]
        target_path = self.target_filenames[index_]

        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')

        # Resize images 
        # input_img = input_img.resize((1024, 1024), Image.BICUBIC)
        # target_img = target_img.resize((1024, 1024), Image.BICUBIC)

        w, h = target_img.size
        padw = self.patch_size - w if w < self.patch_size else 0
        padh = self.patch_size - h if h < self.patch_size else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            input_img = TF.pad(input_img, (0, 0, padw, padh), padding_mode='reflect')
            target_img = TF.pad(target_img, (0, 0, padw, padh), padding_mode='reflect')

        input_img = TF.to_tensor(input_img)
        target_img = TF.to_tensor(target_img)

        hh, ww = target_img.shape[1], target_img.shape[2]

        rr = random.randint(0, hh - self.patch_size)
        cc = random.randint(0, ww - self.patch_size)
        # aug = random.randint(0, 8)

        # Crop patch
        input_img = input_img[:, rr:rr + self.patch_size, cc:cc + self.patch_size]
        target_img = target_img[:, rr:rr + self.patch_size, cc:cc + self.patch_size]

        # Data Augmentations
        # if aug == 1:
        #     input_img = input_img.flip(1)
        #     target_img = target_img.flip(1)
        # elif aug == 2:
        #     input_img = input_img.flip(2)
        #     target_img = target_img.flip(2)
        # elif aug == 3:
        #     input_img = torch.rot90(input_img, dims=(1, 2))
        #     target_img = torch.rot90(target_img, dims=(1, 2))
        # elif aug == 4:
        #     input_img = torch.rot90(input_img, dims=(1, 2), k=2)
        #     target_img = torch.rot90(target_img, dims=(1, 2), k=2)
        # elif aug == 5:
        #     input_img = torch.rot90(input_img, dims=(1, 2), k=3)
        #     target_img = torch.rot90(target_img, dims=(1, 2), k=3)
        # elif aug == 6:
        #     input_img = torch.rot90(input_img.flip(1), dims=(1, 2))
        #     target_img = torch.rot90(target_img.flip(1), dims=(1, 2))
        # elif aug == 7:
        #     input_img = torch.rot90(input_img.flip(2), dims=(1, 2))
        #     target_img = torch.rot90(target_img.flip(2), dims=(1, 2))

        filename = os.path.splitext(os.path.split(target_path)[-1])[0]

        return target_img, input_img, filename



class DataLoaderValidation(Dataset):
    def __init__(self, dir_path, patch_size):
        super(DataLoaderValidation, self).__init__()

        input_files = sorted(os.listdir(os.path.join(dir_path, 'input')))
        target_files = sorted(os.listdir(os.path.join(dir_path, 'gt')))

        self.input_filenames = [os.path.join(dir_path, 'input', x) for x in input_files if is_image_file(x)]
        self.target_filenames = [os.path.join(dir_path, 'gt', x) for x in target_files if is_image_file(x)]

        self.patch_size = patch_size
        self.sizex = len(self.target_filenames)  # get the size of target

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        

        input_path = self.input_filenames[index_]
        target_path = self.target_filenames[index_]

        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')

        # Resize images to 256x256
        # inp_img = inp_img.resize((512, 512), Image.BICUBIC)
        # tar_img = tar_img.resize((512, 512), Image.BICUBIC)

        # Validate on center crop
        w, h = target_img.size
        padw = self.patch_size - w if w < self.patch_size else 0
        padh = self.patch_size - h if h < self.patch_size else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            input_img = TF.pad(input_img, (0, 0, padw, padh), padding_mode='reflect')
            target_img = TF.pad(target_img, (0, 0, padw, padh), padding_mode='reflect')

        input_img = TF.to_tensor(input_img)
        target_img = TF.to_tensor(target_img)

        hh, ww = target_img.shape[1], target_img.shape[2]

        rr = random.randint(0, hh - self.patch_size)
        cc = random.randint(0, ww - self.patch_size)
        # aug = random.randint(0, 8)

        # Crop patch
        input_img = input_img[:, rr:rr + self.patch_size, cc:cc + self.patch_size]
        target_img = target_img[:, rr:rr + self.patch_size, cc:cc + self.patch_size]
        
        filename = os.path.splitext(os.path.split(target_path)[-1])[0]

        return target_img, input_img, filename