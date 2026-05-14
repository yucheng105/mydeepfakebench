#!/usr/bin/env python
"""
Generate evaluation metadata CSV for Celeb-DF-v2 dataset (TEST DATA ONLY).
This script enumerates frames from Celeb-DF-v2 test set defined in List_of_testing_videos.txt
and creates a CSV compatible with run_ucf_inference.py and run_xception_inference.py
"""

import os
import csv
from pathlib import Path

# Repository root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EVAL_ROOT = os.path.join(REPO_ROOT, 'evaluation')
METADATA_DIR = os.path.join(EVAL_ROOT, 'metadata')
DATASET_ROOT = os.path.join(REPO_ROOT, 'datasets', 'rgb', 'Celeb-DF-v2')
OUTPUT_CSV = os.path.join(METADATA_DIR, 'celebdfv2_metadata.csv')
TEST_LIST_FILE = os.path.join(DATASET_ROOT, 'List_of_testing_videos.txt')

def load_test_videos(test_list_file):
    """
    Load test video list from List_of_testing_videos.txt
    Format: label video_path
    where label 1 = real, label 0 = fake
    Returns dict: {video_name: label}
    """
    test_videos = {}
    if not os.path.exists(test_list_file):
        print(f"Warning: Test list file not found: {test_list_file}")
        return test_videos
    
    with open(test_list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) == 2:
                label, video_path = parts
                # Extract video name from path (e.g., "id1_0007" from "Celeb-real/id1_0007.mp4")
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                video_dir = os.path.dirname(video_path)
                test_videos[f"{video_dir}/{video_name}"] = int(label)
    
    return test_videos

def enumerate_test_samples(test_videos):
    """Enumerate test samples from the test video list and yield metadata rows."""
    idx_real = 0
    idx_fake = 0
    
    for test_video_key, label in sorted(test_videos.items()):
        # label 0 = fake (Celeb-synthesis), label 1 = real (Celeb-real, YouTube-real)
        # Convert label: label 0 -> our label 1 (fake), label 1 -> our label 0 (real)
        our_label = 1 if label == 0 else 0
        
        # Find the frames directory
        video_dir = os.path.dirname(test_video_key)
        video_name = os.path.basename(test_video_key)
        
        frames_path = os.path.join(DATASET_ROOT, video_dir, 'frames', video_name)
        
        # Check if the sample directory exists and contains frames
        if not os.path.isdir(frames_path):
            print(f"Skipping {test_video_key}: frames directory not found at {frames_path}")
            continue
        
        # Count frames
        frames = sorted([f for f in os.listdir(frames_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if len(frames) == 0:
            print(f"Skipping {test_video_key}: no frames found")
            continue
        
        # Create sample_id based on label
        if our_label == 0:  # real
            sample_id = f"celeb_test_real_{idx_real:03d}"
            idx_real += 1
            dataset_name = 'Celeb-DF-v2-Test-Real'
        else:  # fake
            sample_id = f"celeb_test_fake_{idx_fake:03d}"
            idx_fake += 1
            dataset_name = 'Celeb-DF-v2-Test-Synthesis'
        
        # Frame directory path (relative to repo for portability)
        frame_dir_rel = os.path.relpath(frames_path, REPO_ROOT).replace('\\', '/')
        
        yield {
            'sample_id': sample_id,
            'path': frame_dir_rel,
            'label': our_label,
            'dataset': dataset_name,
            'has_audio': 0,
            'fps': 30,
            'duration_sec': len(frames) / 30,
            'face_detected': 1,
            'notes': f'{len(frames)} frames available'
        }

def main():
    print("Generating Celeb-DF-v2 TEST DATA evaluation metadata...")
    print(f"Loading test video list from {TEST_LIST_FILE}...")
    
    # Load test videos from the list file
    test_videos = load_test_videos(TEST_LIST_FILE)
    
    if not test_videos:
        print("Error: No test videos found!")
        return
    
    print(f"Found {len(test_videos)} test videos in the list")
    
    all_samples = []
    
    # Process test samples
    for sample in enumerate_test_samples(test_videos):
        all_samples.append(sample)
    
    # Write metadata CSV
    print(f"\nWriting metadata to {OUTPUT_CSV}...")
    print(f"Total test samples: {len(all_samples)}")
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'sample_id', 'path', 'label', 'dataset', 'has_audio', 
            'fps', 'duration_sec', 'face_detected', 'notes'
        ])
        writer.writeheader()
        writer.writerows(all_samples)
    
    print(f"Metadata saved to {OUTPUT_CSV}")
    print(f"\nTest sample count breakdown:")
    real_count = sum(1 for s in all_samples if s['label'] == 0)
    fake_count = sum(1 for s in all_samples if s['label'] == 1)
    print(f"  Real (label=0): {real_count}")
    print(f"  Fake (label=1): {fake_count}")
    print(f"  Total: {len(all_samples)}")

if __name__ == '__main__':
    main()
