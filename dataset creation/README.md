# iPhone to Fuji Preprocessing Pipeline

This project implements a high-performance image preprocessing and data augmentation pipeline to pair, align, and extract patches from 4K images captured on an iPhone and a Fujifilm camera. 

The pipeline is heavily optimized for Apple Silicon (M1 Pro and newer), utilizing a custom **Memory-Constrained Producer-Consumer Model** to prevent unified memory overflow and maximize computational throughput.

## Parallel Programming Techniques Implemented

### 1. Memory-Constrained Producer-Consumer Model (CPU)

Working with 4K uncompressed images rapidly consumes RAM. To avoid extreme SSD swap degradation on macOS Unified Memory, we implement a constrained architecture.

*   **The Master Process (Queue Manager)**: The main thread loops over the input directory, dynamically scanning for matching file pairs (images 1-61) and submitting them to a multiprocessing queue.
*   **The Producers (CPU Workers)**: We use Python's `concurrent.futures.ProcessPoolExecutor(max_workers=4)`. By capping the workers at 4, we pin the intensive OpenCV mathematical operations (SIFT Feature Extraction, FLANN/BFMatching, RANSAC, and Homography Warping) strictly to the M1 Pro's Performance Cores, maintaining a predictable memory footprint.

### 2. I/O Bypassing the Global Interpreter Lock (GIL)

Once the homography aligns the 4K images and slices them into thousands of 256x256 patches, writing these arrays back to the SSD becomes heavily I/O-bound.

*   **The Consumers (I/O Threads)**: Inside *each* of the 4 CPU worker processes, we spawn an embedded `ThreadPoolExecutor`.
*   Since I/O operations release the Python Global Interpreter Lock (GIL), multiple threads can write JPEG files to the NVMe SSD concurrently without blocking the worker from continuing its matrix calculations for the next patch.

### 3. GPU-Accelerated Dynamic Scene Filtering (PyTorch MPS)

To ensure the neural network only trains on perfectly static scenes (ignoring trees blowing in the wind or people walking), we filter out mismatched patches using the Structural Similarity Index (SSIM).

*   **Apple Metal Performance Shaders (MPS)**: SSIM filtering is a convolution-heavy task. Instead of bottlenecking the CPU, `filter.py` loads the raw patches into PyTorch.
*   **Batched Execution**: The patches are pushed to the Apple Silicon GPU via `device=torch.device("mps")` in batches of 128.
*   **In-Memory Augmentation**: Patches that score $\ge 0.4$ SSIM are kept in VRAM. The GPU then natively applies 90°, 180°, and 270° rotations along with horizontal flips to massively multiply the dataset size before writing the final arrays back to the SSD.

## Installation & Execution

This pipeline requires natively compiled ARM64 binaries to leverage Apple Silicon. We recommend using `miniforge3`.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your images
# Place 61 iPhone images in: input/iphone/
# Place 61 Fuji images in: input/fuji/

# 3. Run the Full Pipeline
# This will execute the CPU alignment followed by the GPU SSIM filtering automatically.
python main.py
```
