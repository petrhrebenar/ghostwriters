#!/usr/bin/env python3
"""
Extract spisová značka (file numbers) from ECLI identifiers in the dataset.
"""

import csv
import re
from pathlib import Path
from typing import List


def ecli_to_spis_zn(ecli: str) -> str:
    """
    Convert ECLI identifier to spisová značka format.
    
    ECLI format: ECLI:CZ:US:YEAR:SENAT.NUMBER.VERSION.SUBVERSION
    Example: ECLI:CZ:US:2023:4.US.1971.23.1
    NALUS format: SENAT-NUMBER-VERSION or SENAT-NUMBER-VERSION_SUBVERSION
    Example: 4-1971-23 or 4-1971-23_1
    
    Args:
        ecli: ECLI identifier string
    
    Returns:
        Spisová značka string
    """
    # Parse ECLI: ECLI:CZ:US:2023:4.US.1971.23.1
    match = re.match(r'ECLI:CZ:US:(\d+):(\d+)\.US\.(\d+)\.(\d+)\.(\d+)', ecli)
    if match:
        year = match.group(1)
        senat = match.group(2)
        number = match.group(3)
        version = match.group(4)
        subversion = match.group(5)
        
        # Convert to NALUS format: SENAT-NUMBER-VERSION_SUBVERSION
        return f"{senat}-{number}-{version}_{subversion}"
    
    # Alternative format without subversion
    match = re.match(r'ECLI:CZ:US:(\d+):(\d+)\.US\.(\d+)\.(\d+)', ecli)
    if match:
        year = match.group(1)
        senat = match.group(2)
        number = match.group(3)
        version = match.group(4)
        
        return f"{senat}-{number}-{version}"
    
    return None


def extract_spis_zn_from_csv(csv_path: str) -> List[str]:
    """
    Extract unique spisová značka from CSV file.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        List of unique spisová značka
    """
    spis_zn_list = set()
    
    # Increase field size limit for large text fields
    csv.field_size_limit(1000000)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            ecli = row.get('doc_id', '')
            if ecli:
                spis_zn = ecli_to_spis_zn(ecli)
                if spis_zn:
                    spis_zn_list.add(spis_zn)
    
    return sorted(list(spis_zn_list))


def main():
    """Extract spis_zn from the dataset."""
    csv_path = "../subset_disent2.csv"
    output_path = "spis_zn_list.txt"
    
    spis_zn_list = extract_spis_zn_from_csv(csv_path)
    
    print(f"Extracted {len(spis_zn_list)} unique spisová značka")
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        for spis_zn in spis_zn_list:
            f.write(f"{spis_zn}\n")
    
    print(f"Saved to {output_path}")
    
    # Show first few
    print("\nFirst 10 spis_zn:")
    for spis_zn in spis_zn_list[:10]:
        print(f"  {spis_zn}")


if __name__ == "__main__":
    main()
