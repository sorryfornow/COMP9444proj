import os
import torch
import glob
import numpy as np
from skimage import io, transform
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

# Mask Color if Multi-class
# COLOR_DICT = np.array([
#     [128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128], [60, 40, 222],
#     [128, 128, 0], [192, 128, 128], [64, 64, 128], [64, 0, 128], [64, 64, 0],
#     [0, 128, 192], [0, 0, 0]
# ])


# [[maybe_unused]]
def geneTrainNumpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjust_data(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr, mask_arr


# Data adjustments [[maybe_unused]]
def adjust_data(img, mask, flag_multi_class, num_class):

    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i, i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return img, mask


# Custom dataset
class MembraneDataset(Dataset):
    def __init__(self, root, image_folder, mask_folder, transform=None):
        self.image_folder = os.path.join(root, image_folder)
        self.mask_folder = os.path.join(root, mask_folder)
        self.transform = transform
        self.image_names = glob.glob(os.path.join(self.image_folder, "*.png"))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        mask_name = img_name.replace(self.image_folder, self.mask_folder).replace("image", "mask")
        image = io.imread(img_name, as_gray=True)
        mask = io.imread(mask_name, as_gray=True)
        if self.transform:
            image, mask = adjust_data(image, mask, False, 2)
            image = image.astype(np.float32)
            image = np.clip(image, 0, 1)
            image = self.transform(image)

            mask = mask.astype(np.float32)
            mask = np.clip(mask, 0, 1)
            mask = self.transform(mask)

        return image, mask


# Data loader
def train_generator(batch_size, train_path, image_folder, mask_folder, aug_dict, target_size=(256,256)):
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=aug_dict.get('horizontal_flip', 0.5)),
        # Add other augmentations based on aug_dict
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    dataset = MembraneDataset(train_path, image_folder, mask_folder, transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# Test generator
def test_generator(test_path, num_image=30, target_size=(256,256)):
    image_folder = os.path.join(test_path, "image")
    test_image_names = glob.glob(os.path.join(image_folder, "*.png"))
    num_image = min(num_image, len(test_image_names))
    for i in range(num_image):
        img = io.imread(test_image_names[i], as_gray=True)
        img = transform.resize(img, target_size)
        img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        yield img


# Save results
def save_result(save_path, npyfile, flag_multi_class=False, num_class=2):
    for i, item in enumerate(npyfile):
        # img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
        img = item[:, :, 0] if not flag_multi_class else item
        io.imsave(os.path.join(save_path, "%d_predict.png" % i), img)


# second generator

class MembraneImage(Dataset):
    def __init__(self, root, image_folder, transform=None):
        self.image_folder = os.path.join(root, image_folder)
        self.transform = transform
        self.image_names = glob.glob(os.path.join(self.image_folder, "*.png"))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        label = int(img_name[-5])
        # label is 0 or 1
        label = torch.tensor(label)
        image = io.imread(img_name, as_gray=True)
        if self.transform:
            image = image.astype(np.float32)
            image = np.clip(image, 0, 1)
            image = self.transform(image)
        return image, label


def train_generator_2(batch_size, train_path, image_folder, aug_dict, target_size=(256, 256)):
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=aug_dict.get('horizontal_flip', 0.5)),
        # Add other augmentations based on aug_dict
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    dataset = MembraneImage(train_path, image_folder, transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def evaluateRes(test_path, results):
    image_folder = os.path.join(test_path, "image")
    test_image_names = glob.glob(os.path.join(image_folder, "*.png"))
    num_image = len(test_image_names)
    correct = 0
    # [array([[0.5238405]], dtype=float32), array([[0.5245774]], dtype=float32), array([[0.52511436]], dtype=float32),]
    for i in range(num_image):
        img = io.imread(test_image_names[i], as_gray=True)
        label = test_image_names[i][-5]
        label = int(label)
        # sigmoid results
        t = 0
        if results[i][0][0].any() > 0.5:
            t = 1
        if label == t:
            correct += 1
    return correct/(num_image+1)

def test_generator2(test_path, num_image=30, target_size=(256,256)):
    image_folder = os.path.join(test_path, "label")
    test_image_names = glob.glob(os.path.join(image_folder, "*.png"))
    num_image = min(num_image, len(test_image_names))
    for i in range(num_image):
        img = io.imread(test_image_names[i], as_gray=True)
        img = transform.resize(img, target_size)
        img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        yield img