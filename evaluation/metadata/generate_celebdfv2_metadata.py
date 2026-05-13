#!/usr/bin/env python
"""
Generate evaluation metadata CSV for Celeb-DF-v2 dataset.
This script enumerates frames from Celeb-DF-v2 and creates a CSV
compatible with run_ucf_inference.py and run_xception_inference.py
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

def enumerate_samples(parent_dir, label, dataset_name):
    """Enumerate all samples in a directory and yield metadata rows."""
    if not os.path.isdir(parent_dir):
        print(f"Warning: Directory not found: {parent_dir}")
        return
    
    sample_dirs = sorted([d for d in os.listdir(parent_dir) 
                         if os.path.isdir(os.path.join(parent_dir, d))])
    
    for idx, sample_dir in enumerate(sample_dirs):
        sample_path = os.path.join(parent_dir, sample_dir)
        
        # Check if the sample directory exists and contains frames
        if not os.path.isdir(sample_path):
            print(f"Skipping {sample_dir}: sample directory not found")
            continue
        
        # Count frames
        frames = [f for f in os.listdir(sample_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(frames) == 0:
            print(f"Skipping {sample_dir}: no frames found")
            continue
        
        # Create sample_id
        prefix = "celeb_real" if label == 0 else "celeb_fake"
        sample_id = f"{prefix}_{idx:03d}"
        
        # Frame directory path (relative to repo for portability)
        frame_dir_rel = os.path.relpath(sample_path, REPO_ROOT).replace('\\', '/')
        
        yield {
            'sample_id': sample_id,
            'path': frame_dir_rel,
            'label': label,
            'dataset': dataset_name,
            'has_audio': 0,
            'fps': 30,
            'duration_sec': len(frames) / 30,
            'face_detected': 1,
            'notes': f'{len(frames)} frames available'
        }

def main():
    print("Generating Celeb-DF-v2 evaluation metadata...")
    
    # Define data sources: (directory, label, dataset_name_suffix)
    sources = [
        (os.path.join(DATASET_ROOT, 'Celeb-real', 'frames'), 0, 'Celeb-DF-v2-Real'),
        (os.path.join(DATASET_ROOT, 'Celeb-synthesis', 'frames'), 1, 'Celeb-DF-v2-Synthesis'),
    ]
    
    all_samples = []
    
    for source_dir, label, dataset_name in sources:
        if label == 0:
            print(f"\nProcessing Celeb-real samples from {source_dir}...")
        else:
            print(f"\nProcessing Celeb-synthesis samples from {source_dir}...")

        for sample in enumerate_samples(source_dir, label, dataset_name):
            all_samples.append(sample)
    
    # Write metadata CSV
    print(f"\nWriting metadata to {OUTPUT_CSV}...")
    print(f"Total samples: {len(all_samples)}")
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'sample_id', 'path', 'label', 'dataset', 'has_audio', 
            'fps', 'duration_sec', 'face_detected', 'notes'
        ])
        writer.writeheader()
        writer.writerows(all_samples)
    
    print(f"Metadata saved to {OUTPUT_CSV}")
    print(f"\nSample count breakdown:")
    real_count = sum(1 for s in all_samples if s['label'] == 0)
    fake_count = sum(1 for s in all_samples if s['label'] == 1)
    print(f"  Real (label=0): {real_count}")
    print(f"  Fake (label=1): {fake_count}")
    print(f"  Total: {len(all_samples)}")

if __name__ == '__main__':
    main()
