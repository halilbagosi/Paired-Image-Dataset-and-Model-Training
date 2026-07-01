import os
import random
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(processName)s - %(asctime)s - %(message)s',
                    handlers=[
                        logging.FileHandler("preprocessing.log"),
                        logging.StreamHandler()
                    ])

def split_dataset():
    logging.info("--- Phase 3: Train/Val/Test Split (by Pair) ---")
    
    input_iphone_dir = Path("output/final_dataset/iphone")
    input_fuji_dir = Path("output/final_dataset/fuji")
    
    if not input_iphone_dir.exists() or not input_fuji_dir.exists():
        logging.error("final_dataset directories do not exist. Run filter.py first.")
        return

    output_dir = Path("output/split_dataset")
    splits = ['train', 'val', 'test']
    
    for split in splits:
        (output_dir / split / 'iphone').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'fuji').mkdir(parents=True, exist_ok=True)
        
    # Find all unique pair IDs
    # Patches are named like pair1_p0_orig.jpg
    iphone_files = [f.name for f in input_iphone_dir.iterdir() if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    
    if not iphone_files:
        logging.warning("No files found in final_dataset to split. They might have already been moved.")
        return

    pair_ids = set()
    for f in iphone_files:
        if f.startswith("pair"):
            pair_id = f.split('_')[0][4:] # extract number after 'pair'
            pair_ids.add(pair_id)
            
    pair_ids = sorted(list(pair_ids), key=lambda x: int(x) if x.isdigit() else x)
    
    # Shuffle for random split
    random.seed(42) # fixed seed for reproducibility
    shuffled_pairs = list(pair_ids)
    random.shuffle(shuffled_pairs)
    
    total_pairs = len(shuffled_pairs)
    train_count = int(0.8 * total_pairs)
    val_count = int(0.1 * total_pairs)
    
    train_pairs = set(shuffled_pairs[:train_count])
    val_pairs = set(shuffled_pairs[train_count:train_count+val_count])
    test_pairs = set(shuffled_pairs[train_count+val_count:])
    
    logging.info(f"Total pairs found: {total_pairs}")
    logging.info(f"Train pairs: {len(train_pairs)}")
    logging.info(f"Val pairs: {len(val_pairs)}")
    logging.info(f"Test pairs: {len(test_pairs)}")
    
    # Move files
    moved_count = 0
    for f in iphone_files:
        if not f.startswith("pair"):
            continue
        pair_id = f.split('_')[0][4:]
        
        if pair_id in train_pairs:
            split = 'train'
        elif pair_id in val_pairs:
            split = 'val'
        else:
            split = 'test'
            
        # Move files to split directory
        shutil.move(str(input_iphone_dir / f), str(output_dir / split / 'iphone' / f))
        
        # Move corresponding fuji file
        if (input_fuji_dir / f).exists():
            shutil.move(str(input_fuji_dir / f), str(output_dir / split / 'fuji' / f))
            moved_count += 1
            
    # Clean up empty directories
    try:
        input_iphone_dir.rmdir()
        input_fuji_dir.rmdir()
        input_iphone_dir.parent.rmdir()
    except OSError:
        pass # Directory not empty or doesn't exist
            
    logging.info(f"Successfully moved {moved_count} file pairs to train/val/test splits.")

if __name__ == '__main__':
    split_dataset()
