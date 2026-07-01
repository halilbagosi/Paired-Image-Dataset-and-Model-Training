import logging
from preprocess import main as run_preprocessing
from filter import process_and_filter as run_filtering
from split import split_dataset as run_splitting
import time

logging.basicConfig(level=logging.INFO, 
                    format='%(processName)s - %(asctime)s - %(message)s',
                    handlers=[
                        logging.FileHandler("preprocessing.log"),
                        logging.StreamHandler()
                    ])

def main():
    logging.info("===================================================")
    logging.info("  Starting iPhone-to-Fuji Preprocessing Pipeline   ")
    logging.info("===================================================")
    
    start_total_time = time.time()
    
    # Step 1: CPU-based preprocessing (Alignment, Slicing)
    logging.info("\n--- Phase 1: CPU-Accelerated Alignment & Tiling ---")
    run_preprocessing()
    
    # Step 2: GPU-based filtering (SSIM)
    logging.info("\n--- Phase 2: GPU-Accelerated MPS SSIM Filtering ---")
    run_filtering()
    
    # Step 3: Train/Val/Test Split
    logging.info("\n--- Phase 3: Train/Val/Test Split ---")
    run_splitting()
    
    end_total_time = time.time()
    total_elapsed = end_total_time - start_total_time
    
    logging.info("\n===================================================")
    logging.info(f"    Pipeline Completed Successfully in {total_elapsed:.2f}s!    ")
    logging.info("===================================================")

if __name__ == '__main__':
    main()