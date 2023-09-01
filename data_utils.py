from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose , RandomCrop , ToTensor , ToPILImage ,CenterCrop ,Resize
import torch
torch.autograd.set_detect_anomaly(True)

def is_image_file (filename):
    return any(filename.endswith(extension) for extension in ['.png' , '.jpg' , '.jpeg'])

def calculate_valid_crop_size(crop_size , upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])

def train_lr_transform(crop_size , upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor , interpolation=Image.BICUBIC),
        ToTensor()
    ])

def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])

class TrainDatasetFromFolder(Dataset):
    def __init__(self , dataset_dir , crop_size , upscale_factor):
        super(TrainDatasetFromFolder , self).__init__()
        self.image_filenames = [join(dataset_dir , x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size , upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size , upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image , hr_image
    
    def __len__(self):
        return len(self.image_filenames)

class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)
    
    def __len__(self):
        return len(self.image_filenames)
