import cv2
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO, 
                    format='%(processName)s - %(asctime)s - %(message)s',
                    handlers=[
                        logging.FileHandler("preprocessing.log"),
                        logging.StreamHandler()
                    ])

def crop_black_borders(img1, img2):
    # Create a mask of non-black pixels from the warped image (img2 is usually the warped one)
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Find bounding box of non-black pixels
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        # Crop both images
        img1_cropped = img1[y:y+h, x:x+w]
        img2_cropped = img2[y:y+h, x:x+w]
        return img1_cropped, img2_cropped
    return img1, img2

def save_patch(patch_args):
    """
    Consumer Thread task to save the patch to SSD.
    Bypasses GIL for I/O bound tasks.
    """
    img_patch, save_path = patch_args
    cv2.imwrite(str(save_path), img_patch, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return True

def process_image_pair(pair_id, iphone_path, fuji_path, output_dir):
    try:
        start_time = time.time()
        logging.info(f"Processing pair {pair_id}")
        iphone_img = cv2.imread(str(iphone_path))
        fuji_img = cv2.imread(str(fuji_path))
        
        if iphone_img is None or fuji_img is None:
            logging.error(f"Failed to load images for pair {pair_id}")
            return False

        # 1. Feature Extraction (SIFT)
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(fuji_img, None)
        kp2, des2 = sift.detectAndCompute(iphone_img, None)
        
        # 2. Matching & RANSAC
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                
        if len(good_matches) < 10:
            logging.error(f"Not enough good matches for pair {pair_id}")
            return False

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # RANSAC Outlier Rejection
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            logging.error(f"Homography failed for pair {pair_id}")
            return False

        # 3. Perspective Warping
        h, w = iphone_img.shape[:2]
        warped_fuji = cv2.warpPerspective(fuji_img, H, (w, h))
        
        # Cropping black borders
        iphone_cropped, fuji_cropped = crop_black_borders(iphone_img, warped_fuji)
        
        # 4. Overlapping Tiling
        patch_size = 256
        stride = 128
        
        ch, cw = iphone_cropped.shape[:2]
        
        iphone_out_dir = Path(output_dir) / 'iphone'
        fuji_out_dir = Path(output_dir) / 'fuji'
        
        iphone_out_dir.mkdir(parents=True, exist_ok=True)
        fuji_out_dir.mkdir(parents=True, exist_ok=True)
        
        patch_tasks = []
        patch_count = 0
        
        for y in range(0, ch - patch_size + 1, stride):
            for x in range(0, cw - patch_size + 1, stride):
                iph_patch = iphone_cropped[y:y+patch_size, x:x+patch_size]
                fuj_patch = fuji_cropped[y:y+patch_size, x:x+patch_size]
                
                # Check if patch is mostly black (from warping edge cases)
                if np.mean(fuj_patch) < 10:
                    continue
                    
                iph_path = iphone_out_dir / f"pair{pair_id}_p{patch_count}.jpg"
                fuj_path = fuji_out_dir / f"pair{pair_id}_p{patch_count}.jpg"
                
                patch_tasks.append((iph_patch, iph_path))
                patch_tasks.append((fuj_patch, fuj_path))
                patch_count += 1

        # Consumers (I/O Multithreading)
        # Using a ThreadPoolExecutor inside the CPU worker process to parallelize SSD writes.
        with ThreadPoolExecutor(max_workers=8) as thread_pool:
            list(thread_pool.map(save_patch, patch_tasks))
            
        elapsed_time = time.time() - start_time
        logging.info(f"Pair {pair_id} completed in {elapsed_time:.2f} seconds: generated {patch_count} patches.")
        return True

    except Exception as e:
        logging.error(f"Error processing pair {pair_id}: {e}")
        return False

def main():
    input_dir_iphone = Path("input/iphone")
    input_dir_fuji = Path("input/fuji")
    output_dir = Path("output/raw_patches")
    
    input_dir_iphone.mkdir(parents=True, exist_ok=True)
    input_dir_fuji.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Master Process: Scan and Pair
    jobs = []
    # Assuming images are named 1.jpg, 2.jpg ... up to 61.jpg (or similar extensions)
    # Let's dynamically find them based on the iphone dir
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    
    iphone_files = sorted(
        [f for f in input_dir_iphone.iterdir() if f.suffix.lower() in valid_extensions],
        key=lambda f: f.stem
    )
    
    # Build a case-insensitive lookup for Fuji files by stem
    fuji_lookup = {f.stem.lower(): f for f in input_dir_fuji.iterdir() 
                   if f.suffix.lower() in valid_extensions}
    
    for iph_file in iphone_files:
        pair_id = iph_file.stem  # e.g., '1', '2'
        fuji_file = fuji_lookup.get(iph_file.stem.lower())
        
        if fuji_file is not None:
            jobs.append((pair_id, iph_file, fuji_file, output_dir))
        else:
            logging.warning(f"Matching file for {iph_file.name} not found in fuji folder.")

    if not jobs:
        logging.error("No image pairs found. Please place images 1-61 in input/iphone and input/fuji.")
        return

    logging.info(f"Found {len(jobs)} pairs. Starting Producer-Consumer processing.")

    # Producers (CPU Multiprocessing)
    # Limit max_workers to 4 to prevent 4K uncompressed image arrays from overflowing unified memory
    start_total = time.time()
    with ProcessPoolExecutor(max_workers=4) as process_pool:
        futures = []
        for job in jobs:
            futures.append(process_pool.submit(process_image_pair, *job))
            
        for future in futures:
            future.result() # Wait for completion

    total_elapsed = time.time() - start_total
    logging.info(f"Preprocessing complete in {total_elapsed:.2f} seconds! Patches saved to output/raw_patches.")

if __name__ == '__main__':
    main()
