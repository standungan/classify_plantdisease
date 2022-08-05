from torchvision.datasets import ImageFolder
from torchvision import transforms

import config as cfg

# default image data transforms
data_transforms = {
    'train' : transforms.Compose(
        [
            transforms.Resize((cfg.img_size,cfg.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.255])
        ]),
    'test' : transforms.Compose(
        [
            transforms.Resize((cfg.img_size,cfg.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.255])
        ])
}

# implementasi dataset type = Image Folder
def dset_imageFolder():
    train_dataset = ImageFolder(root=cfg.train_dir, transform=data_transforms['train'])
    
    valid_dataset = ImageFolder(root=cfg.valid_dir, transform=data_transforms['test'])

    test_dataset = ImageFolder(root=cfg.test_dir, transform=data_transforms['test'])

    return train_dataset, valid_dataset, test_dataset


# implementasi dataset type = CSV files
def dset_imageCSV():
    pass