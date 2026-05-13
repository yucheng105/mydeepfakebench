"""
Generate DFDC evaluation metadata CSV compatible with Celeb-DF-v2 layout.
Writes: evaluation/metadata/dfdc_metadata.csv
Header matches `celebdfv2_metadata.csv`:
  sample_id,path,label,dataset,has_audio,fps,duration_sec,face_detected,notes

The script reads `datasets/rgb/DFDC/test/metadata.json` and per-sample frame
directories under `datasets/rgb/DFDC/test/frames/` to compute frame counts
and infer fps when an augmentation provides `framerate_change`.
"""

import os
import json
import csv

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DFDC_ROOT = os.path.join(REPO_ROOT, 'datasets', 'rgb', 'DFDC')
TEST_ROOT = os.path.join(DFDC_ROOT, 'test')
FRAMES_ROOT = os.path.join(TEST_ROOT, 'frames')
METADATA_JSON = os.path.join(TEST_ROOT, 'metadata.json')
OUT_CSV = os.path.join(REPO_ROOT, 'evaluation', 'metadata', 'dfdc_metadata.csv')


def infer_fps(info):
    # look for augmenter->framerate_change->fps
    aug = info.get('augmentations', {}) or {}
    augmenter = aug.get('augmenter', {}) or {}
    fr = augmenter.get('framerate_change') or {}
    fps = fr.get('fps')
    return int(fps) if fps else 30


def has_audio(info):
    aug = info.get('augmentations', {}) or {}
    augmenter = aug.get('augmenter', {}) or {}
    # if 'no_audio' present then audio removed
    return 0 if 'no_audio' in augmenter else 1


if __name__ == '__main__':
    print("Generating DFDC evaluation metadata with extended header...")
    if not os.path.exists(METADATA_JSON):
        print(f"Metadata JSON not found: {METADATA_JSON}")
        raise SystemExit(1)
    with open(METADATA_JSON, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    rows = []
    missing = 0
    real = 0
    fake = 0

    for video_name, info in meta.items():
        basename = os.path.splitext(video_name)[0]
        frame_dir = os.path.join(FRAMES_ROOT, basename)
        if not os.path.exists(frame_dir):
            missing += 1
            continue

        frame_files = [n for n in os.listdir(frame_dir) if os.path.isfile(os.path.join(frame_dir, n))]
        frame_count = len(frame_files)

        label = int(info.get('is_fake', 0))
        if label == 1:
            fake += 1
        else:
            real += 1

        fps = infer_fps(info)
        ha = has_audio(info)
        duration_sec = round(frame_count / fps, 6) if fps > 0 else 'N/A'
        notes = f"{frame_count} frames available"

        # Use repo-relative, portable forward-slash path
        rel_path = os.path.relpath(frame_dir, REPO_ROOT).replace('\\', '/')

        rows.append({
            'sample_id': basename,
            'path': rel_path,
            'label': label,
            'dataset': 'DFDC',
            'has_audio': ha,
            'fps': fps,
            'duration_sec': duration_sec,
            'face_detected': 1,
            'notes': notes,
        })

    # write csv
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fieldnames = ['sample_id', 'path', 'label', 'dataset', 'has_audio', 'fps', 'duration_sec', 'face_detected', 'notes']
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} samples to {OUT_CSV} (real={real}, fake={fake}, missing_dirs={missing})")
