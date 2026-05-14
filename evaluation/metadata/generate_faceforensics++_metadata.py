#!/usr/bin/env python
"""
Generate evaluation metadata CSV for FaceForensics++ dataset (TEST DATA ONLY).
This script reads test data from preprocessing/dataset_json/FaceForensics++.json
and creates a CSV compatible with run_ucf_inference.py and run_xception_inference.py
"""

import os
import json
import csv

# Repository root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EVAL_ROOT = os.path.join(REPO_ROOT, 'evaluation')
METADATA_DIR = os.path.join(EVAL_ROOT, 'metadata')
DATASET_JSON = os.path.join(REPO_ROOT, 'preprocessing', 'dataset_json', 'FaceForensics++.json')
OUTPUT_CSV = os.path.join(METADATA_DIR, 'faceforensics++_metadata.csv')

# Label mapping
LABEL_MAP = {
    'FF-real': 0,  # Real
    'FF-DF': 1,    # Deepfakes
    'FF-F2F': 1,   # Face2Face
    'FF-FS': 1,    # FaceSwap
    'FF-NT': 1,    # NeuralTextures
}

# Method name mapping for dataset field
METHOD_DATASET_MAP = {
    'FF-real': 'FF-Real',
    'FF-DF': 'FF-Deepfakes',
    'FF-F2F': 'FF-Face2Face',
    'FF-FS': 'FF-FaceSwap',
    'FF-NT': 'FF-NeuralTextures',
}

def load_faceforensics_json(json_file):
    """
    Load FaceForensics++ dataset JSON.
    Returns dict with structure: {method: {compression: {video_name: {label, frames}}}}
    """
    if not os.path.exists(json_file):
        print(f"Error: Dataset JSON not found: {json_file}")
        return {}
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract FaceForensics++ data
    ff_data = data.get('FaceForensics++', {})
    
    # Filter to only test split
    test_data = {}
    for method, method_data in ff_data.items():
        if 'test' in method_data:
            test_data[method] = method_data['test']
    
    return test_data

def enumerate_test_samples(test_data):
    """Enumerate test samples and yield metadata rows."""
    sample_id_counter = 0
    
    # Sort methods and compression levels for consistent output
    for method in sorted(test_data.keys()):
        compression_data = test_data[method]
        
        for compression_level in sorted(compression_data.keys()):
            videos_data = compression_data[compression_level]
            
            for video_name in sorted(videos_data.keys()):
                video_info = videos_data[video_name]
                
                # Get label
                label_str = video_info.get('label', method)
                label = LABEL_MAP.get(label_str, 1)  # Default to fake (1) if unknown
                
                # Get frames
                frames = video_info.get('frames', [])
                
                if len(frames) == 0:
                    print(f"Skipping {method}/{compression_level}/{video_name}: no frames found")
                    continue
                
                # Create sample_id
                sample_id = f"ff_test_{sample_id_counter:06d}"
                sample_id_counter += 1
                
                # Get dataset name
                dataset_name = METHOD_DATASET_MAP.get(method, method)
                dataset_name_with_comp = f"{dataset_name}-{compression_level}"
                
                # Frame directory path (relative to repo for portability).
                # JSON stores Windows-style separators; normalize first so this
                # works on both Windows and Linux.
                first_frame = str(frames[0]).replace('\\', '/')
                frame_dir_rel = first_frame.rsplit('/', 1)[0] if '/' in first_frame else ""
                if frame_dir_rel and not frame_dir_rel.startswith('datasets/rgb/'):
                    frame_dir_rel = f"datasets/rgb/{frame_dir_rel}"
                
                yield {
                    'sample_id': sample_id,
                    'path': frame_dir_rel,
                    'label': label,
                    'dataset': dataset_name_with_comp,
                    'has_audio': 0,  # FaceForensics++ typically doesn't use audio
                    'fps': 30,
                    'duration_sec': len(frames) / 30,
                    'face_detected': 1,
                    'notes': f'{len(frames)} frames available, compression={compression_level}'
                }

def main():
    print("Generating FaceForensics++ TEST DATA evaluation metadata...")
    print(f"Loading dataset JSON from {DATASET_JSON}...")
    
    # Load test data from JSON
    test_data = load_faceforensics_json(DATASET_JSON)
    
    if not test_data:
        print("Error: No test data found in JSON!")
        return
    
    # Count total test videos
    total_videos = sum(
        len(videos) 
        for compression_data in test_data.values() 
        for videos in compression_data.values()
    )
    print(f"Found {len(test_data)} methods with {total_videos} total test videos")
    
    all_samples = []
    
    # Process test samples
    for sample in enumerate_test_samples(test_data):
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
    
    # Breakdown by method
    print(f"\nBreakdown by method:")
    dataset_counts = {}
    for s in all_samples:
        dataset = s['dataset']
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
    
    for dataset in sorted(dataset_counts.keys()):
        print(f"  {dataset}: {dataset_counts[dataset]}")

if __name__ == '__main__':
    main()
