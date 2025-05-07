import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from utils.utils_func import natural_key

class TumorDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images = sorted([f for f in os.listdir(os.path.join(data_dir, 'images')) if f.endswith('.tif')], key=natural_key)
        self.masks = sorted([f for f in os.listdir(os.path.join(data_dir, 'masks')) if f.endswith('.tif')], key=natural_key)

        self.image_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, 'images', self.images[idx])
        mask_path = os.path.join(self.data_dir, 'masks', self.masks[idx])

        image = Image.open(img_path)
        mask = Image.open(mask_path)

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        
        return image, mask