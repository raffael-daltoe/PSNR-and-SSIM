import cv2, argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_metric
from decord import VideoReader, cpu

def create_comparison_graphic(img1_path, img2_path, ssim_score, psnr_score, sharpness, output_path="comparison_results.png"):
    """
    Create a comprehensive graphic comparing two images with SSIM and PSNR metrics
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image  
        ssim_score: SSIM score
        psnr_score: PSNR score
        output_path: Path to save the output graphic
    """
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
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
    ax1.set_title('Image 1', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot2grid((3, 3), (0, 2), colspan=2)
    ax2.imshow(img2_rgb)
    ax2.set_title('Image 2', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Metrics display
    ax3 = plt.subplot2grid((10, 3), (0, 0), colspan=3)
    ax3.axis('off')
    
    # Create metrics text
    metrics_text = f"""
    QUALITY METRICS COMPARISON
    {'='*40}
    
    STRUCTURAL SIMILARITY INDEX (SSIM)
    Score: {ssim_score:.4f}
    Interpretation: {get_ssim_interpretation(ssim_score)}
    
    PEAK SIGNAL-TO-NOISE RATIO (PSNR)  
    Score: {psnr_score:.2f} dB
    Interpretation: {get_psnr_interpretation(psnr_score)}
    
    SHARPNESS:
    Score: {sharpness}%
    
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
    
    #plt.tight_layout()
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
    
    # Create gradient bar
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, extent=[0, 1, 0, 1], aspect='auto', cmap='RdYlGn')
    
    # Add range labels
    for i, (start, end, label, color) in enumerate(ranges):
        ax.text((start + end) / 2, 0.3, label, ha='center', va='center', 
                fontsize=8, rotation=45, color='black', fontweight='bold')
    
    # Mark current score
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
    # Handle infinite PSNR (identical images)
    if current_score == float('inf'):
        current_score = 60  # Place at top of scale for display
    
    ranges = [
        (0, 20, 'Poor', 'red'),
        (20, 30, 'Acceptable', 'orange'),
        (30, 40, 'Good', 'yellow'),
        (40, 50, 'Excellent', 'lightgreen'),
        (50, 60, 'Perfect', 'darkgreen')
    ]
    
    max_psnr = 60
    
    # Create gradient bar
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, extent=[0, max_psnr, 0, 1], aspect='auto', cmap='RdYlGn')
    
    # Add range labels
    for start, end, label, color in ranges:
        ax.text((start + end) / 2, 0.3, label, ha='center', va='center', 
                fontsize=8, rotation=45, color='black', fontweight='bold')
    
    # Mark current score
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
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

def psnr(img1, img2):
    # Calculate MSE (Mean Squared Error)
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    
    # If MSE is zero, images are identical
    if mse == 0:
        return float('inf')
    
    # Maximum possible pixel value
    max_pixel = 255.0
    
    # Calculate PSNR
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

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
            print(f"Images have different dimensions: {img1.shape} vs {img2.shape}")
            
            # Choose a consistent target size (use the smaller dimensions)
            target_height = min(img1.shape[0], img2.shape[0])
            target_width = min(img1.shape[1], img2.shape[1])
            
            print(f"Resizing both images to: ({target_height}, {target_width})")
            
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
        image = vr[frameIndex]
        frameNp = image.asnumpy()
        frame_bgr = cv2.cvtColor(frameNp, cv2.COLOR_RGB2BGR)
        cv2.imwrite("imageFrame.png",frame_bgr)
        return frame_bgr
    else:
        raise FileNotFoundError(f"File: {video} not found") if video is None else ValueError(f"Frame: {frame} invalid")

def main():
    args = argparse.ArgumentParser()
    args.add_argument("-v1","--video1")
    args.add_argument("-v2","--video2") 
    args.add_argument("-f1","--frame1")
    args.add_argument("-f2","--frame2")
    args.add_argument("-i1","--image1")
    args.add_argument("-i2","--image2")
    args.add_argument("-o","--output", default="comparison_results.png", help="Output graphic filename")
    args = args.parse_args()
    v1 = args.video1
    v2 = args.video2
    f1 = args.frame1
    f2 = args.frame2
    i1 = args.image1
    i2 = args.image2
    output = args.output

    try:
        if (v1 or v2) and (i1 or i2):
            raise ValueError("You have to specify either Image or Video, not both!")
        
        img1_path = i1
        img2_path = i2
        
        if v1 and v2:
            img1 = extract2Frames(v1, f1)
            img2 = extract2Frames(v2, f2)
            # Save temporary images for the graphic
            cv2.imwrite("temp_img1.png", img1)
            cv2.imwrite("temp_img2.png", img2)
            img1_path = "temp_img1.png"
            img2_path = "temp_img2.png"
            ssimScore = ssim(img1, img2)
            psnrScore = psnr(img1, img2)
        elif i1 and i2:
            ssimScore = ssim(i1, i2)
            # Calculate PSNR from the loaded images in ssim function
            img1 = cv2.imread(i1)
            img2 = cv2.imread(i2)
            if img1.shape != img2.shape:
                target_height = min(img1.shape[0], img2.shape[0])
                target_width = min(img1.shape[1], img2.shape[1])
                img1 = cv2.resize(img1, (target_width, target_height))
                img2 = cv2.resize(img2, (target_width, target_height))
            psnrScore = psnr(img1, img2)
            sharpnessScore1 = sharpness(img1)
            sharpnessScore2 = sharpness(img2)
        else:
            raise ValueError("You have to specify 2 videos OR 2 images!")
        
        # Print results to console
        print(f"\n{'='*50}")
        print("QUALITY METRICS RESULTS")
        print(f"{'='*50}")
        print(f"SSIM Score: {ssimScore:.4f}")
        print(f"PSNR Score: {psnrScore:.2f} dB")
        print(f"SSIM Interpretation: {get_ssim_interpretation(ssimScore)}")
        print(f"PSNR Interpretation: {get_psnr_interpretation(psnrScore)}")
        
        sharpInPerc = (sharpnessScore2 / sharpnessScore1) * 100

        # Create comparison graphic
        create_comparison_graphic(img1_path, img2_path, ssimScore, psnrScore, sharpInPerc ,output)
        
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()

#https://brpucrs-my.sharepoint.com/:x:/g/personal/80403070_pucrs_br/EWzYI53KqcdFgewxUDzr7IUB-DBvp0p6yaveHk05oLisXg?e=okqIpl