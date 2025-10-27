# Image & Video Frame Quality Comparison Tool

This Python script provides a comprehensive, command-line tool for comparing the quality of two images or two specific video frames. It calculates key quality metrics and generates a detailed visual report summarizing the findings.

The primary metrics used are:
* **SSIM (Structural Similarity Index)**
* **PSNR (Peak Signal-to-Noise Ratio)**
* **Sharpness** (via Laplacian Variance)

## Features

* **Dual Mode:** Compare two static images (e.g., `.png`, `.jpg`) OR two specific video frames (e.g., from `.mp4`, `.avi`).
* **Comprehensive Metrics:** Calculates SSIM, PSNR, and relative sharpness.
* **Visual Report:** Generates a `matplotlib` graphic showing the images side-by-side with their scores.
* **Interpretations:** Provides plain-English interpretations for SSIM ("Nearly Identical", "Similar", etc.) and PSNR ("Excellent", "Good", etc.).
* **Robust Handling:** Automatically resizes images to matching dimensions for comparison if they differ.
* **Efficient Video Reading:** Uses `decord` for fast and accurate video frame extraction.

## Installation

This script requires Python 3 and several external libraries.

1.  Clone or download the repository.
2.  Install the required packages using `pip`:

    ```bash
    pip install opencv-python numpy matplotlib scikit-image decord PyQt6
    ```

## Usage

The script is run from the command line. You must provide **either** two images **or** two videos to compare.

### 1. Compare Two Images

Use the **-i1** and **-i2** flags to specify the paths to your two images.

```bash
python compare.py -i1 "path/to/original_image.png" -i2 "path/to/compressed_image.jpg"
```

**Optional Flag:**

  * **-o <filename>**: Specify the output filename for the resulting comparison image (e.g., `-o "diff.png"`).

-----

### 2. Compare Two Videos

Use the **-v1** and **-v2** flags to specify the paths to your two videos. By default, this compares the average of all frames.

```bash
python compare.py -v1 "path/to/original_video.mp4" -v2 "path/to/compressed_video.avi"
```

#### Compare Specific Frames

To compare a single, specific frame from each video, use the **-s** flag along with **-f1** and **-f2**.

```bash
python compare.py -v1 "path/to/video1.mp4" -f1 150 -v2 "path/to/video2.mp4" -f2 100 -s
```

**Optional Video Flags:**

  * **-s**: Compare specific frames only. Requires **-f1** and **-f2** to be set.
  * **-f1 <number>**: The frame number to extract from video 1 (used with **-s**).
  * **-f2 <number>**: The frame number to extract from video 2 (used with **-s**).
  * **-o <filename>**: Specify the output filename when comparing specific frames (e.g., `-o "frame_comp.png"`).