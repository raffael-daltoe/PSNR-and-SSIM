import cv2, argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_metric
from decord import VideoReader, cpu

# Define a default number of samples for full video analysis
DEFAULT_SAMPLES = 10

# create_comparison_graphic, get_..._interpretation, plot_..._scale
# are all UNCHANGED from the previous version. I've omitted them
# for brevity, but they are still required.

def create_comparison_graphic(img1_path, img2_path, ssim_score, psnr_score, 
                            sharpness, output_path="comparison_results.png",
                            title1="Image 1", title2="Image 2", metric_label=""):
    """
    Create a comprehensive graphic comparing two images with SSIM and PSNR metrics
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image  
        ssim_score: SSIM score
        psnr_score: PSNR score
        sharpness: Relative sharpness percentage
        output_path: Path to save the output graphic
        title1: Title for the first image
        title2: Title for the second image
        metric_label: Label to add to metric names (e.g., "(Average)")
    """
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # Check if images loaded correctly
    if img1 is None:
        raise FileNotFoundError(f"Could not load image from {img1_path} for graphic.")
    if img2 is None:
        raise FileNotFoundError(f"Could not load image from {img2_path} for graphic.")
        
    # Convert BGR to RGB for matplotlib
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 8))
    
    # Main title
    fig.suptitle('Image Quality Comparison Analysis', fontsize=16, fontweight='bold', y=0.95)
    
    # Plot original images
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=1)
    ax1.imshow(img1_rgb)
    ax1.set_title(title1, fontsize=10, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot2grid((3, 3), (0, 2), colspan=2)
    ax2.imshow(img2_rgb)
    ax2.set_title(title2, fontsize=10, fontweight='bold')
    ax2.axis('off')
    
    # Metrics display
    ax3 = plt.subplot2grid((10, 3), (0, 0), colspan=3)
    ax3.axis('off')
    
    # Create metrics text
    metrics_text = f"""
    QUALITY METRICS COMPARISON
    {'='*40}
    
    STRUCTURAL SIMILARITY INDEX (SSIM) {metric_label}
    Score: {ssim_score:.4f}
    Interpretation: {get_ssim_interpretation(ssim_score)}
    
    PEAK SIGNAL-TO-NOISE RATIO (PSNR) {metric_label}
    Score: {psnr_score:.2f} dB
    Interpretation: {get_psnr_interpretation(psnr_score)}
    
    RELATIVE SHARPNESS
    (Image 2 vs Image 1 or Video 2 vs Video 1 Avg.)
    Score: {sharpness:.2f}%
    
    Image 1 Dimensions: {img1.shape[1]} x {img1.shape[0]}
    Image 2 Dimensions: {img2.shape[1]} x {img2.shape[0]}
    """
    
    ax3.text(0.35, 1.4, metrics_text, fontfamily='monospace', fontsize=12, 
             verticalalignment='top', linespacing=1.5)
    
    # SSIM visualization
    ax4 = plt.subplot2grid((3, 4), (2, 0), colspan=2)
    plot_ssim_scale(ax4, ssim_score)
    ax4.set_title('SSIM Scale Interpretation', fontsize=12, fontweight='bold')
    
    # PSNR visualization  
    ax5 = plt.subplot2grid((3, 4), (2, 2), colspan=2)
    plot_psnr_scale(ax5, psnr_score)
    ax5.set_title('PSNR Scale Interpretation', fontsize=12, fontweight='bold')
    
    plt.subplots_adjust(top=0.90)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison graphic saved to: {output_path}")
    plt.show()

def get_ssim_interpretation(score):
    """Get textual interpretation of SSIM score"""
    if score >= 0.95:
        return "Nearly Identical"
    elif score >= 0.90:
        return "Very Similar" 
    elif score >= 0.80:
        return "Similar"
    elif score >= 0.70:
        return "Moderately Similar"
    elif score >= 0.50:
        return "Somewhat Similar"
    else:
        return "Very Different"

def get_psnr_interpretation(score):
    """Get textual interpretation of PSNR score"""
    if score == float('inf'):
        return "Perfect (Identical)"
    elif score > 40:
        return "Excellent"
    elif score > 30:
        return "Good"
    elif score > 20:
        return "Acceptable"
    else:
        return "Poor"

def plot_ssim_scale(ax, current_score):
    """Plot SSIM scale with current score marked"""
    ranges = [
        (0.0, 0.5, 'Very Different', 'red'),
        (0.5, 0.7, 'Somewhat Similar', 'orange'),
        (0.7, 0.8, 'Moderately Similar', 'yellow'),
        (0.8, 0.9, 'Similar', 'lightgreen'),
        (0.9, 0.95, 'Very Similar', 'green'),
        (0.95, 1.0, 'Nearly Identical', 'darkgreen')
    ]
    
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, extent=[0, 1, 0, 1], aspect='auto', cmap='RdYlGn')
    
    for i, (start, end, label, color) in enumerate(ranges):
        ax.text((start + end) / 2, 0.3, label, ha='center', va='center', 
                fontsize=8, rotation=45, color='black', fontweight='bold')
    
    ax.axvline(x=current_score, color='blue', linewidth=3, linestyle='--')
    ax.text(current_score, 0.7, f'Current: {current_score:.3f}', 
            ha='center', va='center', fontweight='bold', color='blue',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel('SSIM Score')
    ax.set_yticks([])

def plot_psnr_scale(ax, current_score):
    """Plot PSNR scale with current score marked"""
    if current_score == float('inf'):
        current_score = 60
    
    ranges = [
        (0, 20, 'Poor', 'red'),
        (20, 30, 'Acceptable', 'orange'),
        (30, 40, 'Good', 'yellow'),
        (40, 50, 'Excellent', 'lightgreen'),
        (50, 60, 'Perfect', 'darkgreen')
    ]
    
    max_psnr = 60
    
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, extent=[0, max_psnr, 0, 1], aspect='auto', cmap='RdYlGn')
    
    for start, end, label, color in ranges:
        ax.text((start + end) / 2, 0.3, label, ha='center', va='center', 
                fontsize=8, rotation=45, color='black', fontweight='bold')
    
    ax.axvline(x=current_score, color='blue', linewidth=3, linestyle='--')
    ax.text(current_score, 0.7, f'Current: {current_score:.1f} dB', 
            ha='center', va='center', fontweight='bold', color='blue',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_xlim(0, max_psnr)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, max_psnr + 1, 10))
    ax.set_xlabel('PSNR (dB)')
    ax.set_yticks([])

def sharpness(img):
    """Calculates sharpness using Laplacian variance"""
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

def calculate_average_sharpness(video_path, num_samples=DEFAULT_SAMPLES):
    """
    Calculates the average sharpness of a video by sampling frames.
    
    Args:
        video_path (str): Path to the video file.
        num_samples (int): Number of frames to sample for the average.
    
    Returns:
        float: The average sharpness (Laplacian variance)
    """
    print(f"\nCalculating average sharpness for {video_path}...")
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames = len(vr)
        if num_frames == 0:
            raise ValueError("Video is empty or cannot be read.")

        total_sharpness = 0
        
        # Create a list of frame indices to sample
        # We use np.linspace to get evenly distributed frames
        indices = np.linspace(0, num_frames - 1, num_samples, dtype=int)
        
        # Ensure unique indices, just in case (e.g., short video)
        indices = np.unique(indices) 

        for idx in indices:
            frame_np = vr[idx].asnumpy()
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            total_sharpness += sharpness(frame_bgr)
            
        if len(indices) == 0:
            print("Warning: Could not sample any frames.")
            return 0 # Avoid division by zero
            
        avg_sharpness = total_sharpness / len(indices)
        print(f"Average sharpness (from {len(indices)} samples): {avg_sharpness:.2f}")
        return avg_sharpness
        
    except Exception as e:
        print(f"Error calculating sharpness for {video_path}: {e}")
        return 0 # Return 0 on failure

def psnr(img1, img2):
    # Calculate MSE (Mean Squared Error)
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    
    # If MSE is zero, images are identical
    if mse == 0:
        return float('inf')
    
    # Maximum possible pixel value
    max_pixel = 255.0
    
    # Calculate PSNR
    psnr_val = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_val

def ssim(i1, i2):
    try:
        # Handle both file paths and numpy arrays
        if isinstance(i1, str):
            img1 = cv2.imread(i1)
        else:
            img1 = i1
            
        if isinstance(i2, str):
            img2 = cv2.imread(i2)
        else:
            img2 = i2
            
        if img1 is None or img2 is None:
            raise ValueError("Could not load one or both images")
            
        # Ensure images have the same dimensions
        if img1.shape != img2.shape:
            # Choose a consistent target size (use the smaller dimensions)
            target_height = min(img1.shape[0], img2.shape[0])
            target_width = min(img1.shape[1], img2.shape[1])
            
            # Resize both images to the same target size
            img1 = cv2.resize(img1, (target_width, target_height))
            img2 = cv2.resize(img2, (target_width, target_height))
        
        # Convert to float32 for SSIM calculation
        img1_float = img1.astype(np.float32)
        img2_float = img2.astype(np.float32)
        
        # Calculate SSIM
        if len(img1.shape) == 3 and img1.shape[2] == 3:
            ssim_score = ssim_metric(img1_float, img2_float, channel_axis=-1, data_range=255)
        else:
            ssim_score = ssim_metric(img1_float, img2_float, data_range=255)
        
        return ssim_score
        
    except Exception as e:
        print(f"Error calculating SSIM: {e}")
        return None

def extract2Frames(video : str, frame: int):
    if (frame and video) is not None:
        vr = VideoReader(video, ctx=cpu(0)) 
        frameIndex = int(frame)
        if frameIndex >= len(vr):
            raise ValueError(f"Frame index {frameIndex} is out of bounds for video {video} (total frames: {len(vr)})")
        image = vr[frameIndex]
        frameNp = image.asnumpy()
        frame_bgr = cv2.cvtColor(frameNp, cv2.COLOR_RGB2BGR)
        return frame_bgr
    else:
        raise FileNotFoundError(f"File: {video} not found") if video is None else ValueError(f"Frame: {frame} invalid")

### --- MODIFIED FUNCTION --- ###
def compare_full_videos(video_path1, video_path2, start_frame1=0, start_frame2=0, num_samples=DEFAULT_SAMPLES):
    """
    Compares two videos frame-by-frame (using sampling) and calculates
    the average SSIM and PSNR, starting from a specific frame for each.
    
    Args:
        video_path1 (str): Path to the first video file.
        video_path2 (str): Path to the second video file.
        start_frame1 (int): The frame number to start comparison from in video 1.
        start_frame2 (int): The frame number to start comparison from in video 2.
        num_samples (int): Number of frames to sample.
        
    Returns:
        tuple: (average_ssim, average_psnr, number_of_frames_compared)
    """
    print(f"\nStarting full video comparison...")
    print(f"  Video 1: {video_path1} (from frame {start_frame1})")
    print(f"  Video 2: {video_path2} (from frame {start_frame2})")
    print(f"  Sampling {num_samples} frames from each video.")
    
    try:
        vr1 = VideoReader(video_path1, ctx=cpu(0))
        vr2 = VideoReader(video_path2, ctx=cpu(0))
        
        num_frames1 = len(vr1)
        num_frames2 = len(vr2)
        
        if num_frames1 == 0 or num_frames2 == 0:
            raise ValueError("One or both videos are empty.")
        
        # Check if start frames are valid
        if start_frame1 >= num_frames1:
            raise ValueError(f"Start frame {start_frame1} is out of bounds for {video_path1} ({num_frames1} frames)")
        if start_frame2 >= num_frames2:
            raise ValueError(f"Start frame {start_frame2} is out of bounds for {video_path2} ({num_frames2} frames)")

        # Calculate remaining frames in each video
        remaining_frames1 = num_frames1 - start_frame1
        remaining_frames2 = num_frames2 - start_frame2
        
        # The number of frames we can compare is the minimum of the two
        comparable_frames = min(remaining_frames1, remaining_frames2)
        
        if comparable_frames < num_samples:
            print(f"Warning: Remaining comparable frames ({comparable_frames}) is less than sample size. Using {comparable_frames} frames.")
            num_samples = comparable_frames

        if num_samples == 0:
            print("No comparable frames to sample.")
            return 0.0, 0.0, 0
            
        total_ssim = 0.0
        psnr_scores = []
        
        # Create a list of frame indices for *both* videos
        # We sample 'num_samples' from the 'comparable_frames'
        # and then add the respective start frame offsets.
        
        # 1. Get sample indices from 0 to comparable_frames
        sample_indices_relative = np.linspace(0, comparable_frames - 1, num_samples, dtype=int)
        
        # 2. Add the start frame offsets
        indices1 = sample_indices_relative + start_frame1
        indices2 = sample_indices_relative + start_frame2
        
        # Ensure unique (though linspace should already be sorted)
        indices1 = np.unique(indices1)
        indices2 = np.unique(indices2)
        
        # Recalculate number of samples in case 'unique' changed anything (e.g., short videos)
        num_compared = len(indices1)
        print(f"Comparing {num_compared} unique sampled frames...")

        for i in range(num_compared):
            idx1 = indices1[i]
            idx2 = indices2[i]
            
            if (i+1) % 50 == 0:
                print(f"  ... processing sample {i+1}/{num_compared} (v1 frame {idx1} vs v2 frame {idx2})")
                
            # Get frames from both videos
            frame1_np = vr1[idx1].asnumpy()
            frame2_np = vr2[idx2].asnumpy()
            
            # Convert to BGR for OpenCV functions (ssim, psnr)
            frame1_bgr = cv2.cvtColor(frame1_np, cv2.COLOR_RGB2BGR)
            frame2_bgr = cv2.cvtColor(frame2_np, cv2.COLOR_RGB2BGR)
            
            # Calculate metrics for the current pair of frames
            current_ssim = ssim(frame1_bgr, frame2_bgr)
            current_psnr = psnr(frame1_bgr, frame2_bgr)
            
            total_ssim += current_ssim
            
            # Handle 'inf' PSNR (identical frames) by not including in mean
            if current_psnr != float('inf'):
                psnr_scores.append(current_psnr)
        
        if num_compared == 0:
            return 0.0, 0.0, 0 # No frames processed

        # Calculate averages
        avg_ssim = total_ssim / num_compared
        
        # If psnr_scores is empty (all frames were identical), avg is 'inf'
        avg_psnr = np.mean(psnr_scores) if psnr_scores else float('inf')
        
        print("Full video comparison complete.")
        return avg_ssim, avg_psnr, num_compared # Return number of frames

    except Exception as e:
        print(f"Error during full video comparison: {e}")
        return 0.0, 0.0, 0 # Return 0 on failure


### --- MODIFIED FUNCTION --- ###
def main():
    args = argparse.ArgumentParser()
    args.add_argument("-v1","--video1")
    args.add_argument("-v2","--video2") 
    args.add_argument("-f1","--frame1")
    args.add_argument("-f2","--frame2")
    args.add_argument("-i1","--image1")
    args.add_argument("-i2","--image2")
    args.add_argument("-o","--output", default="comparison_results.png", help="Output graphic filename")
    ### --- NEW ARGUMENT --- ###
    args.add_argument("-s","--specific", action="store_true", help="Compare specific frames only (f1 vs f2), not the full video average.")
    
    args = args.parse_args()
    v1 = args.video1
    v2 = args.video2
    f1 = args.frame1
    f2 = args.frame2
    i1 = args.image1
    i2 = args.image2
    output = args.output
    specific_frame_mode = args.specific

    try:
        if (v1 or v2) and (i1 or i2):
            raise ValueError("You have to specify either Image or Video, not both!")
        
        img1_path = i1
        img2_path = i2
        
        # Initialize sharpness scores
        sharpnessScore1 = 0
        sharpnessScore2 = 0
        
        # Initialize dynamic labels for the graphic
        title1 = "Image 1"
        title2 = "Image 2"
        metric_label = "" # Label for SSIM/PSNR (e.g., "Avg.")
        
        ### --- MODIFIED LOGIC --- ###
        if v1 and v2:
            # --- Average sharpness calculation (common for all video modes) ---
            sharpnessScore1 = calculate_average_sharpness(v1, num_samples=DEFAULT_SAMPLES)
            sharpnessScore2 = calculate_average_sharpness(v2, num_samples=DEFAULT_SAMPLES)
            
            if specific_frame_mode:
                # --- MODE 1: Compare specific frames ONLY ---
                if not f1 or not f2:
                    raise ValueError("You must specify --frame1 (-f1) and --frame2 (-f2) when using --specific (-s) mode.")
                
                print(f"Comparing SPECIFIC frames: {f1} from {v1} vs. {f2} from {v2}...")
                img1 = extract2Frames(v1, f1)
                img2 = extract2Frames(v2, f2)
                
                # Save temporary images for the graphic
                cv2.imwrite("temp_img1.png", img1)
                cv2.imwrite("temp_img2.png", img2)
                img1_path = "temp_img1.png"
                img2_path = "temp_img2.png"
                
                ssimScore = ssim(img1, img2)
                psnrScore = psnr(img1, img2)
                
                # Set labels for graphic
                title1 = f"{v1} (Frame {f1})"
                title2 = f"{v2} (Frame {f2})"
                metric_label = "(for displayed frames)"

            else:
                # --- MODE 2: Full video average comparison ---
                # (This is now the default behavior for -v1/-v2)
                
                # Use f1/f2 as start frames if provided, otherwise default to 0
                start_frame1 = int(f1) if f1 else 0
                start_frame2 = int(f2) if f2 else 0

                if f1 or f2:
                    print(f"Full video comparison requested with custom start frames.")
                else:
                    print("Full video comparison requested, starting from frame 0.")

                ssimScore, psnrScore, num_compared = compare_full_videos(
                    v1, v2, 
                    start_frame1=start_frame1, 
                    start_frame2=start_frame2, 
                    num_samples=DEFAULT_SAMPLES
                )
                
                # Extract the *start_frame* from each video for the graphic display
                print(f"Extracting start frames ({start_frame1} and {start_frame2}) for the comparison graphic...")
                img1 = extract2Frames(v1, start_frame1)
                img2 = extract2Frames(v2, start_frame2)
                
                cv2.imwrite("temp_img1.png", img1)
                cv2.imwrite("temp_img2.png", img2)
                img1_path = "temp_img1.png"
                img2_path = "temp_img2.png"

                # Set labels for graphic
                title1 = f"{v1} (Sample from Frame {start_frame1})"
                title2 = f"{v2} (Sample from Frame {start_frame2})"
                metric_label = f"(Avg. of {num_compared} frames, from v1[{start_frame1}] / v2[{start_frame2}])"

        elif i1 and i2:
            # --- MODE 3: Compare two images ---
            if f1 or f2 or specific_frame_mode:
                print("Warning: --frame1, --frame2, and --specific are ignored when comparing images.")
                
            img1 = cv2.imread(i1)
            img2 = cv2.imread(i2)
            if img1 is None or img2 is None:
                raise FileNotFoundError("Could not read one or both input images.")

            # Calculate SSIM (handles resizing internally)
            ssimScore = ssim(i1, i2)
            
            # Handle resizing for PSNR and Sharpness
            if img1.shape != img2.shape:
                print("Resizing images for PSNR/Sharpness calculation...")
                target_height = min(img1.shape[0], img2.shape[0])
                target_width = min(img1.shape[1], img2.shape[1])
                img1 = cv2.resize(img1, (target_width, target_height))
                img2 = cv2.resize(img2, (target_width, target_height))
                
            psnrScore = psnr(img1, img2)
            
            # --- Sharpness calculation for *single* images ---
            print("\nCalculating sharpness for images...")
            sharpnessScore1 = sharpness(img1)
            sharpnessScore2 = sharpness(img2)
            print(f"Image 1 Sharpness: {sharpnessScore1:.2f}")
            print(f"Image 2 Sharpness: {sharpnessScore2:.2f}")

            # Set labels for graphic
            title1 = f"Image 1 ({i1})"
            title2 = f"Image 2 ({i2})"
            metric_label = "(for displayed images)"

        else:
            raise ValueError("You have to specify 2 videos (with or without frames) OR 2 images!")
        
        # --- Print Results ---
        print(f"\n{'='*50}")
        print("QUALITY METRICS RESULTS")
        print(f"{'='*50}")
        print(f"SSIM Score {metric_label}: {ssimScore:.4f}")
        print(f"PSNR Score {metric_label}: {psnrScore:.2f} dB")
        print(f"SSIM Interpretation: {get_ssim_interpretation(ssimScore)}")
        print(f"PSNR Interpretation: {get_psnr_interpretation(psnrScore)}")
        
        # Calculate relative sharpness
        if sharpnessScore1 == 0:
            sharpInPerc = 0.0
            print("\nWarning: Base Image/Video 1 has zero sharpness. Cannot calculate relative percentage.")
        else:
            sharpInPerc = (sharpnessScore2 / sharpnessScore1) * 100
            print(f"\nRelative Sharpness: {sharpInPerc:.2f}% (Target is {sharpInPerc:.2f}% as sharp as base)")

        # Create comparison graphic
        create_comparison_graphic(img1_path, img2_path, ssimScore, psnrScore, sharpInPerc, 
                                  output, title1=title1, title2=title2, metric_label=metric_label)
        
    except Exception as e:
        print(f"\n[ERROR] {e}")

if __name__ == "__main__":
    main()