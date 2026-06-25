#!/usr/bin/env python3
"""
Gaussian Splatting Automation Pipeline for macOS (Metal-Optimized)
mimicking intelligent tools like Luma3D.

This script executes three phases:
Phase 1: Smart Frame Extraction using Laplacian-based sharpness detection.
Phase 2: Camera Tracking and Sparse Reconstruction using COLMAP.
Phase 3: Metal-Optimized Gaussian Splat training using OpenSplat.

Author: Antigravity AI
Date: June 2026
"""

import os
import sys
import shutil
import time
import subprocess
import argparse
from pathlib import Path

# Verify dependencies on run
try:
    import cv2
    import numpy as np
except ImportError:
    print("\n[ERROR] Required Python dependencies are missing.")
    print("Please install them using pip:")
    print("    pip install opencv-python numpy")
    print("\nAlso ensure you have installed system dependencies:")
    print("    brew install colmap")
    sys.exit(1)

def print_banner(title):
    """Prints a beautiful colored phase banner to the terminal."""
    border = "=" * 80
    print(f"\n\033[94m{border}\033[0m")
    print(f"\033[92m[PHASE] {title}\033[0m")
    print(f"\033[94m{border}\033[0m\n")

def compute_sharpness(frame):
    """
    Computes the variance of the Laplacian of the frame.
    Higher values represent sharper images with more high-frequency detail.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def extract_smart_frames(video_path, output_dir, target_fps, sharpness_threshold):
    """
    Phase 1: Smart Frame Extraction (The "Luma" Method)
    
    Splits the video into segments corresponding to the target extraction rate.
    Within each segment, it calculates the sharpness of every frame and selects 
    only the single sharpest frame. If that frame's sharpness is below the 
    threshold, it is discarded to prevent motion blur and compression artifacts.
    """
    print(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video Specifications:")
    print(f"  - Frame Rate: {fps:.2f} FPS")
    print(f"  - Dimensions: {width}x{height}")
    print(f"  - Total Frames: {total_frames}")
    
    # Calculate segment size (number of frames per window to sample one frame)
    segment_size = max(1, int(fps / target_fps))
    expected_frames = total_frames / segment_size
    print(f"Target extraction rate: ~{target_fps} FPS (Segment size: {segment_size} frames)")
    print(f"Processing segments... (Expecting to evaluate ~{expected_frames:.1f} segments)")
    
    os.makedirs(output_dir, exist_ok=True)
    
    frame_idx = 0
    saved_count = 0
    discarded_count = 0
    
    best_frame = None
    best_sharpness = -1.0
    best_frame_idx = -1
    
    sharpness_scores = []
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video: Save the best frame of the last (incomplete) segment if valid
            if best_frame is not None:
                if best_sharpness >= sharpness_threshold:
                    save_path = os.path.join(output_dir, f"frame_{best_frame_idx:05d}.jpg")
                    cv2.imwrite(save_path, best_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    saved_count += 1
                else:
                    discarded_count += 1
            break
            
        sharpness = compute_sharpness(frame)
        sharpness_scores.append(sharpness)
        
        # Keep track of the sharpest frame in the current segment
        if sharpness > best_sharpness:
            best_sharpness = sharpness
            best_frame = frame.copy()
            best_frame_idx = frame_idx
            
        frame_idx += 1
        
        # End of segment reached
        if frame_idx % segment_size == 0:
            if best_frame is not None:
                if best_sharpness >= sharpness_threshold:
                    save_path = os.path.join(output_dir, f"frame_{best_frame_idx:05d}.jpg")
                    cv2.imwrite(save_path, best_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    saved_count += 1
                else:
                    discarded_count += 1
            
            # Reset segment states
            best_frame = None
            best_sharpness = -1.0
            best_frame_idx = -1
            
            # Print brief progress
            if frame_idx % (segment_size * 10) == 0 or frame_idx >= total_frames - segment_size:
                progress = (frame_idx / total_frames) * 100
                print(f"  Progress: {progress:.1f}% | Evaluated {frame_idx}/{total_frames} frames | Saved {saved_count} frames | Discarded {discarded_count} blurred")
                
    cap.release()
    elapsed = time.time() - start_time
    
    avg_sharpness = np.mean(sharpness_scores) if sharpness_scores else 0.0
    max_sharpness = np.max(sharpness_scores) if sharpness_scores else 0.0
    min_sharpness = np.min(sharpness_scores) if sharpness_scores else 0.0
    
    print(f"\nSmart Frame Extraction Finished in {elapsed:.2f}s:")
    print(f"  - Extracted & Saved: {saved_count} sharp frames")
    print(f"  - Discarded (Below threshold {sharpness_threshold}): {discarded_count} blurry frames")
    print(f"  - Sharpness stats: Min={min_sharpness:.1f}, Max={max_sharpness:.1f}, Avg={avg_sharpness:.1f}")
    
    if saved_count == 0:
        raise ValueError(
            f"No frames were saved! All evaluated frames fell below the sharpness threshold ({sharpness_threshold}). "
            f"Max evaluated sharpness was {max_sharpness:.1f}. Try lowering the --sharpness-threshold."
        )
    elif saved_count < 10:
        print(f"\n[WARNING] Only {saved_count} frames were extracted. COLMAP requires a substantial number of frames (usually 30+) for reliable tracking. Consider lowering --sharpness-threshold or increasing --target-fps.")
        
    return saved_count

def run_command(cmd, log_file=None):
    """
    Runs a CLI command, streaming output to stdout and writing to log_file.
    Raises CalledProcessError if return code is non-zero.
    """
    print(f"Executing: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    log_fd = None
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        log_fd = open(log_file, "w")
        
    try:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if log_fd:
                log_fd.write(line)
    except Exception as e:
        process.kill()
        raise e
    finally:
        if log_fd:
            log_fd.close()
            
    ret_code = process.wait()
    if ret_code != 0:
        raise subprocess.CalledProcessError(ret_code, cmd)

def run_colmap_pipeline(colmap_path, workspace_dir, single_camera=True, camera_model="RADIAL"):
    """
    Phase 2: Camera Tracking using COLMAP
    
    Executes Feature Extraction, Feature Matching, and Sparse Mapping.
    """
    db_path = os.path.join(workspace_dir, "database.db")
    image_dir = os.path.join(workspace_dir, "images")
    sparse_dir = os.path.join(workspace_dir, "sparse")
    log_dir = os.path.join(workspace_dir, "logs")
    
    # Ensure databases/dirs clean
    if os.path.exists(db_path):
        os.remove(db_path)
    if os.path.exists(sparse_dir):
        shutil.rmtree(sparse_dir)
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 2.1 Feature Extraction
    print("\n--- 2.1 Extracting Features ---")
    feat_cmd = [
        colmap_path, "feature_extractor",
        "--database_path", db_path,
        "--image_path", image_dir,
        "--ImageReader.single_camera", "1" if single_camera else "0",
        "--ImageReader.camera_model", camera_model
    ]
    run_command(feat_cmd, log_file=os.path.join(log_dir, "colmap_feature_extraction.log"))
    
    # 2.2 Exhaustive Matching
    print("\n--- 2.2 Matching Features (Exhaustive Matcher) ---")
    match_cmd = [
        colmap_path, "exhaustive_matcher",
        "--database_path", db_path
    ]
    run_command(match_cmd, log_file=os.path.join(log_dir, "colmap_matching.log"))
    
    # 2.3 Sparse Mapping
    print("\n--- 2.3 Running Sparse Reconstruction (Mapper) ---")
    map_cmd = [
        colmap_path, "mapper",
        "--database_path", db_path,
        "--image_path", image_dir,
        "--output_path", sparse_dir
    ]
    run_command(map_cmd, log_file=os.path.join(log_dir, "colmap_mapping.log"))
    
    # Verification of COLMAP Output
    reconstruct_dir = os.path.join(sparse_dir, "0")
    if not os.path.exists(reconstruct_dir):
        # Look for any reconstruction directory, e.g. 1, 2 or search files inside
        subdirs = [os.path.join(sparse_dir, d) for d in os.listdir(sparse_dir) if os.path.isdir(os.path.join(sparse_dir, d))]
        valid_reconstruct = None
        for sd in subdirs:
            if (os.path.exists(os.path.join(sd, "cameras.bin")) or os.path.exists(os.path.join(sd, "cameras.txt"))) and \
               (os.path.exists(os.path.join(sd, "images.bin")) or os.path.exists(os.path.join(sd, "images.txt"))) and \
               (os.path.exists(os.path.join(sd, "points3D.bin")) or os.path.exists(os.path.join(sd, "points3D.txt"))):
                valid_reconstruct = sd
                break
        
        if valid_reconstruct:
            reconstruct_dir = valid_reconstruct
            print(f"\n[INFO] COLMAP created reconstruction in non-standard folder: {reconstruct_dir}")
        else:
            raise RuntimeError(
                "COLMAP mapping finished, but failed to create a valid sparse model reconstruction inside "
                f"'{sparse_dir}'. No valid model directories (e.g. 'sparse/0') with camera/image/point data were found. "
                "This usually means the images did not have enough overlapping visual features to build a 3D model."
            )
            
    print(f"\nCOLMAP Camera Tracking Successful! Output stored in: {reconstruct_dir}")
    return reconstruct_dir

def run_opensplat_training(opensplat_path, workspace_dir, output_ply, num_iters, custom_cmd=None):
    """
    Phase 3: Metal-Optimized Splat Training
    
    Runs OpenSplat training on the workspace folder, outputs the trained Gaussian Splat .ply file.
    """
    log_dir = os.path.join(workspace_dir, "logs")
    os.makedirs(os.path.dirname(output_ply), exist_ok=True)
    
    if custom_cmd:
        # User defined a custom template command. Format key variables.
        # Format can use variables: input_dir, output_ply, num_iters
        formatted_cmd = custom_cmd.format(
            input_dir=workspace_dir,
            output_ply=output_ply,
            num_iters=num_iters
        )
        cmd = formatted_cmd.split()
    else:
        # Default OpenSplat syntax
        cmd = [
            opensplat_path,
            "-i", workspace_dir,
            "-o", output_ply,
            "-n", str(num_iters)
        ]
        
    print(f"Starting Training Process...")
    run_command(cmd, log_file=os.path.join(log_dir, "training.log"))
    
    if not os.path.exists(output_ply):
        raise FileNotFoundError(f"Training completed, but final output PLY file '{output_ply}' was not found.")
        
    print(f"\nTraining Successful! Gaussian Splat (.ply) saved to: {output_ply}")

def main():
    parser = argparse.ArgumentParser(
        description="Automated local Python pipeline for macOS/Metal generating Gaussian Splats (.ply) from video (.mp4)."
    )
    
    # Inputs & Workspace
    parser.add_argument("--video", type=str, required=True, help="Path to the input MP4 video file.")
    parser.add_argument("--workspace", type=str, default="", help="Path to workspace directory (created near video if not specified).")
    parser.add_argument("--output", type=str, default="", help="Path for the final output PLY file. Defaults to <workspace>/output/splat.ply")
    
    # Extraction parameters (Phase 1)
    parser.add_argument("--fps", type=float, default=2.5, help="Target frames per second to extract from video (default: 2.5).")
    parser.add_argument("--sharpness-threshold", type=float, default=100.0, help="Laplacian variance threshold. Lower threshold to keep more frames (default: 100.0).")
    
    # COLMAP parameters (Phase 2)
    parser.add_argument("--colmap-path", type=str, default="colmap", help="Path to the colmap executable (default: 'colmap').")
    parser.add_argument("--camera-model", type=str, default="RADIAL", help="COLMAP camera model (default: 'RADIAL').")
    parser.add_argument("--no-single-camera", action="store_true", help="Disable the assumption that all frames share a single camera parameter set.")
    
    # Training parameters (Phase 3)
    parser.add_argument("--opensplat-path", type=str, default="opensplat", help="Path to the opensplat executable (default: 'opensplat').")
    parser.add_argument("--num-iters", type=int, default=30000, help="Number of training iterations (default: 30000).")
    parser.add_argument("--custom-cmd", type=str, default="", help="Custom trainer command template (e.g. 'custom_trainer --dataset {input_dir} --out {output_ply} --iters {num_iters}').")
    
    # Dependency Check Override
    parser.add_argument("--skip-dep-check", action="store_true", help="Skip checking system executables in PATH prior to running.")
    
    args = parser.parse_args()
    
    # Setup Paths
    video_path = os.path.abspath(args.video)
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        sys.exit(1)
        
    video_dir = os.path.dirname(video_path)
    video_name = Path(video_path).stem
    
    # If no workspace specified, default to video_name_workspace/ inside video folder
    if not args.workspace:
        workspace_dir = os.path.join(video_dir, f"{video_name}_workspace")
    else:
        workspace_dir = os.path.abspath(args.workspace)
        
    # If no output specified, default to workspace/output/splat.ply
    if not args.output:
        output_ply = os.path.join(workspace_dir, "output", f"{video_name}_splat.ply")
    else:
        output_ply = os.path.abspath(args.output)
        
    image_dir = os.path.join(workspace_dir, "images")
    
    print(f"=== Starting Gaussian Splat Pipeline ===")
    print(f"Input Video: {video_path}")
    print(f"Workspace:   {workspace_dir}")
    print(f"Output PLY:  {output_ply}")
    
    # Check dependencies before starting
    if not args.skip_dep_check:
        try:
            # Validate COLMAP
            if shutil.which(args.colmap_path) is None:
                raise FileNotFoundError(
                    f"COLMAP executable '{args.colmap_path}' not found in PATH.\n"
                    "Make sure it is installed (e.g., 'brew install colmap') or provide the exact path with --colmap-path."
                )
            
            # Validate trainer bin if we're not overriding with custom trainer command
            if not args.custom_cmd:
                if shutil.which(args.opensplat_path) is None:
                    raise FileNotFoundError(
                        f"OpenSplat executable '{args.opensplat_path}' not found in PATH.\n"
                        "Please download/build it or specify the exact path using --opensplat-path.\n"
                        "If you are using a different training tool, supply a custom command via --custom-cmd."
                    )
            print("System dependency check: PASSED\n")
        except FileNotFoundError as err:
            print(f"[ERROR] Dependency Check Failed:\n{err}")
            sys.exit(1)
            
    pipeline_start = time.time()
    
    try:
        # Phase 1: Smart Frame Extraction
        print_banner("Phase 1: Smart Frame Extraction (The 'Luma' Method)")
        num_extracted = extract_smart_frames(
            video_path=video_path,
            output_dir=image_dir,
            target_fps=args.fps,
            sharpness_threshold=args.sharpness_threshold
        )
        
        # Phase 2: Camera Tracking (COLMAP)
        print_banner("Phase 2: Camera Tracking & Reconstruction (COLMAP)")
        reconstruction_dir = run_colmap_pipeline(
            colmap_path=args.colmap_path,
            workspace_dir=workspace_dir,
            single_camera=not args.no_single_camera,
            camera_model=args.camera_model
        )
        
        # Phase 3: Metal-Optimized Splat Training
        print_banner("Phase 3: Metal-Optimized Training")
        run_opensplat_training(
            opensplat_path=args.opensplat_path,
            workspace_dir=workspace_dir,
            output_ply=output_ply,
            num_iters=args.num_iters,
            custom_cmd=args.custom_cmd
        )
        
        total_time = time.time() - pipeline_start
        print(f"\n\033[92m[SUCCESS] Pipeline completed successfully in {total_time/60:.2f} minutes!\033[0m")
        print(f"Final trained Gaussian Splat point cloud: \033[94m{output_ply}\033[0m")
        print("You can view the resulting .ply file using viewers like Splat (https://playcanvas.com/super-splat) or native viewers.")
        
    except Exception as e:
        print(f"\n\033[91m[FAILURE] Pipeline failed during execution: {e}\033[0m")
        sys.exit(1)

if __name__ == "__main__":
    main()
