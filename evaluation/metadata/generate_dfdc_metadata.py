"""
Generate DFDC evaluation metadata CSV (TEST DATA ONLY).
Writes: evaluation/metadata/dfdc_metadata.csv
Header matches `celebdfv2_metadata.csv`:
  sample_id,path,label,dataset,has_audio,fps,duration_sec,face_detected,notes

The script reads test data from preprocessing/dataset_json/DFDC.json
"""

import os
import json
import csv

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASET_JSON = os.path.join(REPO_ROOT, 'preprocessing', 'dataset_json', 'DFDC.json')
OUT_CSV = os.path.join(REPO_ROOT, 'evaluation', 'metadata', 'dfdc_metadata.csv')

# Label mapping: DFDC_Real -> 0, DFDC_Fake -> 1
LABEL_MAP = {
    'DFDC_Real': 0,
    'DFDC_Fake': 1,
}

def load_dfdc_json(json_file):
    """
    Load DFDC dataset JSON.
    Returns dict with structure: {category: {split: {video_name: {label, frames}}}}
    """
    if not os.path.exists(json_file):
        print(f"Error: Dataset JSON not found: {json_file}")
        return {}
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract DFDC test data
    dfdc_data = data.get('DFDC', {})
    
    test_data = {}
    for category, category_data in dfdc_data.items():
        if 'test' in category_data:
            test_data[category] = category_data['test']
    
    return test_data

def enumerate_test_samples(test_data):
    """Enumerate test samples and yield metadata rows."""
    sample_id_counter = 0
    
    # Sort categories for consistent output
    for category in sorted(test_data.keys()):
        videos_data = test_data[category]
        
        for video_name in sorted(videos_data.keys()):
            video_info = videos_data[video_name]
            
            # Get label
            label_str = video_info.get('label', category)
            label = LABEL_MAP.get(label_str, 1)  # Default to fake (1) if unknown
            
            # Get frames
            frames = video_info.get('frames', [])
            
            if len(frames) == 0:
                print(f"Skipping {category}/{video_name}: no frames found")
                continue
            
            # Create sample_id
            sample_id = f"dfdc_test_{sample_id_counter:06d}"
            sample_id_counter += 1
            
            # Get dataset name
            dataset_name = f"DFDC-{category}"
            
            # Frame directory path (relative to repo for portability).
            # JSON stores Windows-style separators, so normalize first.
            first_frame = str(frames[0]).replace('\\', '/')
            frame_dir_rel = first_frame.rsplit('/', 1)[0] if '/' in first_frame else ""
            if frame_dir_rel and not frame_dir_rel.startswith('datasets/rgb/'):
                frame_dir_rel = f"datasets/rgb/{frame_dir_rel}"
            
            yield {
                'sample_id': sample_id,
                'path': frame_dir_rel,
                'label': label,
                'dataset': dataset_name,
                'has_audio': 0,  # DFDC typically doesn't use audio in evaluation
                'fps': 30,
                'duration_sec': len(frames) / 30,
                'face_detected': 1,
                'notes': f'{len(frames)} frames available'
            }

def main():
    print("Generating DFDC TEST DATA evaluation metadata...")
    print(f"Loading dataset JSON from {DATASET_JSON}...")
    
    # Load test data from JSON
    test_data = load_dfdc_json(DATASET_JSON)
    
    if not test_data:
        print("Error: No test data found in JSON!")
        return
    
    # Count total test videos
    total_videos = sum(len(videos) for videos in test_data.values())
    print(f"Found {len(test_data)} categories with {total_videos} total test videos")
    
    rows = []
    
    # Process test samples
    for sample in enumerate_test_samples(test_data):
        rows.append(sample)
    
    # Write metadata CSV
    print(f"\nWriting metadata to {OUT_CSV}...")
    print(f"Total test samples: {len(rows)}")
    
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fieldnames = ['sample_id', 'path', 'label', 'dataset', 'has_audio', 'fps', 'duration_sec', 'face_detected', 'notes']
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Metadata saved to {OUT_CSV}")
    print(f"\nTest sample count breakdown:")
    real_count = sum(1 for s in rows if s['label'] == 0)
    fake_count = sum(1 for s in rows if s['label'] == 1)
    print(f"  Real (label=0): {real_count}")
    print(f"  Fake (label=1): {fake_count}")
    print(f"  Total: {len(rows)}")
    
    # Breakdown by category
    print(f"\nBreakdown by category:")
    category_counts = {}
    for s in rows:
        dataset = s['dataset']
        category_counts[dataset] = category_counts.get(dataset, 0) + 1
    
    for category in sorted(category_counts.keys()):
        print(f"  {category}: {category_counts[category]}")

if __name__ == '__main__':
    main()
