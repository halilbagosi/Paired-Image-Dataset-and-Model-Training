import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class FujiIphoneDataset(Dataset):
    def __init__(self, root_dir, phase='train', crop_size=256):
        """
        Args:
            root_dir (str): Root directory containing 'train', 'val', and 'test' subdirectories,
                            each with 'fuji' and 'iphone' folders.
                            Expected structure:
                                root_dir/train/fuji/
                                root_dir/train/iphone/
                                root_dir/val/fuji/
                                root_dir/val/iphone/
                                root_dir/test/fuji/
                                root_dir/test/iphone/
            phase (str): 'train', 'val', or 'test'
            crop_size (int): Size of the crop to be applied if needed
        """
        self.root_dir  = root_dir
        self.phase     = phase
        self.crop_size = crop_size

        self.fuji_dir   = os.path.join(root_dir, phase, 'fuji')
        self.iphone_dir = os.path.join(root_dir, phase, 'iphone')

        if not os.path.isdir(self.fuji_dir):
            raise ValueError(f"Fuji directory not found: {self.fuji_dir}")
        if not os.path.isdir(self.iphone_dir):
            raise ValueError(f"iPhone directory not found: {self.iphone_dir}")

        self.image_filenames = sorted([
            f for f in os.listdir(self.fuji_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        if len(self.image_filenames) == 0:
            raise ValueError(f"No image files found in {self.fuji_dir}")

        print(f"[{phase.upper()}] Found {len(self.image_filenames)} images.")

        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]

        fuji_path   = os.path.join(self.fuji_dir,   img_name)
        iphone_path = os.path.join(self.iphone_dir, img_name)

        fuji_img   = Image.open(fuji_path).convert('RGB')
        iphone_img = Image.open(iphone_path).convert('RGB')

        if self.phase == 'train':
            # Joint augmentation: same transform on both images
            if torch.rand(1) < 0.5:
                fuji_img   = fuji_img.transpose(Image.FLIP_LEFT_RIGHT)
                iphone_img = iphone_img.transpose(Image.FLIP_LEFT_RIGHT)
            if torch.rand(1) < 0.5:
                fuji_img   = fuji_img.transpose(Image.FLIP_TOP_BOTTOM)
                iphone_img = iphone_img.transpose(Image.FLIP_TOP_BOTTOM)

        fuji_tensor   = transforms.ToTensor()(fuji_img)
        iphone_tensor = transforms.ToTensor()(iphone_img)

        # Center-crop so both dimensions are divisible by 32 (required by U-Net / PyNET)
        _, h, w = fuji_tensor.shape
        new_h   = h - (h % 32)
        new_w   = w - (w % 32)

        if new_h != h or new_w != w:
            crop          = transforms.CenterCrop((new_h, new_w))
            fuji_tensor   = crop(fuji_tensor)
            iphone_tensor = crop(iphone_tensor)

        fuji_tensor   = self.normalize(fuji_tensor)
        iphone_tensor = self.normalize(iphone_tensor)

        return {'fuji': fuji_tensor, 'iphone': iphone_tensor, 'name': img_name}


def get_dataloaders(root_dir, batch_size=8, num_workers=4,
                    pin_memory=False, persistent_workers=False,
                    prefetch_factor=2):
    train_dataset = FujiIphoneDataset(root_dir, phase='train')
    val_dataset   = FujiIphoneDataset(root_dir, phase='val')

    # prefetch_factor: each worker pre-loads this many batches into RAM ahead of time
    # so the GPU never stalls waiting for data. Only valid when num_workers > 0.
    _prefetch = prefetch_factor if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=_prefetch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=_prefetch,
    )

    return train_loader, val_loader


def get_test_dataloader(root_dir, batch_size=1, num_workers=4):
    test_dataset = FujiIphoneDataset(root_dir, phase='test')
    test_loader  = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return test_loader