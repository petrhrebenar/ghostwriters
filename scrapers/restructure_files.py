#!/usr/bin/env python3
"""
Restructure existing decision files from year/month/uuid.json to court_code/year/uuid.json
"""

import json
import shutil
from pathlib import Path
from collections import Counter

decisions_dir = Path("data/rozhodnuti/decisions")
stats = Counter()
errors = []

# Process all existing files
for old_path in decisions_dir.rglob("*.json"):
    try:
        # Read the file to get court_code and year
        with open(old_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        court_code = data.get('metadata', {}).get('courtCode', 'UNKNOWN')
        year = data.get('metadata', {}).get('decisionAt', '2020')[:4]
        uuid = old_path.stem
        
        # New path: court_code/year/uuid.json
        new_dir = decisions_dir / court_code / year
        new_dir.mkdir(parents=True, exist_ok=True)
        new_path = new_dir / f"{uuid}.json"
        
        # Move file
        shutil.move(str(old_path), str(new_path))
        stats['moved'] += 1
        stats[court_code] += 1
        
        if stats['moved'] % 100 == 0:
            print(f"Moved {stats['moved']} files...")
            
    except Exception as e:
        errors.append((str(old_path), str(e)))
        stats['errors'] += 1
        print(f"Error processing {old_path}: {e}")

# Clean up empty directories
for old_dir in sorted(decisions_dir.rglob("*"), reverse=True):
    if old_dir.is_dir() and not any(old_dir.iterdir()):
        old_dir.rmdir()
        stats['removed_dirs'] += 1

print(f"\n{'='*60}")
print(f"Restructure complete!")
print(f"{'='*60}")
print(f"Files moved: {stats['moved']}")
print(f"Errors: {stats.get('errors', 0)}")
print(f"Directories removed: {stats.get('removed_dirs', 0)}")
print(f"\nFiles by court:")
for court, count in stats.most_common():
    if court not in ['moved', 'errors', 'removed_dirs']:
        print(f"  {court}: {count}")

if errors:
    print(f"\nErrors:")
    for path, error in errors[:10]:
        print(f"  {path}: {error}")
