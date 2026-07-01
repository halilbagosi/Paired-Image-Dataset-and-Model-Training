import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from torchvision.io import read_image, write_jpeg
from pathlib import Path
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure
import logging
import time

logging.basicConfig(level=logging.INFO, 
                    format='%(processName)s - %(asctime)s - %(message)s',
                    handlers=[
                        logging.FileHandler("preprocessing.log"),
                        logging.StreamHandler()
                    ])

class PatchDataset(Dataset):
    def __init__(self, iphone_dir, fuji_dir):
        self.iphone_dir = Path(iphone_dir)
        self.fuji_dir = Path(fuji_dir)
        self.patch_names = [f.name for f in self.iphone_dir.iterdir() if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]

    def __len__(self):
        return len(self.patch_names)

    def __getitem__(self, idx):
        patch_name = self.patch_names[idx]
        iphone_path = self.iphone_dir / patch_name
        fuji_path = self.fuji_dir / patch_name
        
        # Read image to uint8 tensor (C, H, W)
        iphone_img = read_image(str(iphone_path))
        fuji_img = read_image(str(fuji_path))
        
        return iphone_img, fuji_img, patch_name

def process_and_filter():
    start_filter_time = time.time()
    raw_iphone_dir = Path("output/raw_patches/iphone")
    raw_fuji_dir = Path("output/raw_patches/fuji")
    
    final_iphone_dir = Path("output/final_dataset/iphone")
    final_fuji_dir = Path("output/final_dataset/fuji")
    
    final_iphone_dir.mkdir(parents=True, exist_ok=True)
    final_fuji_dir.mkdir(parents=True, exist_ok=True)
    
    if not raw_iphone_dir.exists() or not raw_fuji_dir.exists():
        logging.error("Raw patches directories do not exist. Run preprocess.py first.")
        return

    # MPS acceleration for Apple Silicon
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using Apple Silicon MPS for hardware acceleration.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using CUDA for hardware acceleration.")
    else:
        device = torch.device("cpu")
        logging.warning("Hardware acceleration not found. Falling back to CPU.")

    dataset = PatchDataset(raw_iphone_dir, raw_fuji_dir)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
    # SSIM Metric initialized with reduction='none' to get scores for each patch in the batch
    # data_range is 1.0 because we will scale the images to [0, 1] for SSIM calculation
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none').to(device)
    
    ssim_threshold = 0.4
    accepted_count = 0
    discarded_count = 0
    
    logging.info(f"Starting SSIM filtering for {len(dataset)} patches...")
    
    for iphone_batch, fuji_batch, patch_names in tqdm(dataloader, desc="Filtering Batches"):
        # PyTorch image tensors are [0, 255] uint8. We convert to float32 [0.0, 1.0] for SSIM
        iph_float = iphone_batch.to(device).float() / 255.0
        fuj_float = fuji_batch.to(device).float() / 255.0
        
        # Calculate SSIM
        ssim_scores = ssim_metric(iph_float, fuj_float)
        
        for i in range(len(patch_names)):
            score = ssim_scores[i].item()
            if score >= ssim_threshold:
                accepted_count += 1
                base_name = patch_names[i]
                name_no_ext = os.path.splitext(base_name)[0]
                
                iph_img = iphone_batch[i]
                fuj_img = fuji_batch[i]
                
                write_jpeg(iph_img, str(final_iphone_dir / f"{name_no_ext}.jpg"), quality=100)
                write_jpeg(fuj_img, str(final_fuji_dir / f"{name_no_ext}.jpg"), quality=100)
            else:
                discarded_count += 1
                
    logging.info("--- Filtering Complete ---")
    logging.info(f"Accepted Patches: {accepted_count}")
    logging.info(f"Discarded Patches (SSIM < {ssim_threshold}): {discarded_count}")
    logging.info(f"Total Usable Patches: {accepted_count}")
    
    elapsed = time.time() - start_filter_time
    logging.info(f"GPU Filtering complete in {elapsed:.2f} seconds.")

if __name__ == '__main__':
    process_and_filter()