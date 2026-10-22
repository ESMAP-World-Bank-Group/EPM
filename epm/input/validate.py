#!/usr/bin/env python3
"""
Minimal validation script for EPM input data.

Validates CSV files against the Frictionless Data Package schema.
"""

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Any


def validate_type(value: str, field_type: str) -> bool:
    """Validate a value against a field type."""
    if not value or value.strip() == '':
        return True  # Empty values are allowed
    
    try:
        if field_type == 'integer':
            int(value)
        elif field_type == 'number':
            float(value)
        elif field_type == 'string':
            pass  # Always valid
        return True
    except (ValueError, TypeError):
        return False


def validate_primary_key(row: Dict[str, str], primary_key: List[str], seen_keys: set) -> tuple[bool, str]:
    """Validate primary key uniqueness."""
    key_values = tuple(row.get(k, '') for k in primary_key)
    key_str = str(key_values)
    
    if key_str in seen_keys:
        return False, f"Duplicate primary key: {key_values}"
    seen_keys.add(key_str)
    return True, ""


def validate_resource(resource: Dict[str, Any], data_folder: Path) -> tuple[bool, List[str]]:
    """Validate a single resource against its schema."""
    errors = []
    resource_name = resource['name']
    csv_path = data_folder / resource['path']
    
    if not csv_path.exists():
        errors.append(f"{resource_name}: File not found: {csv_path}")
        return False, errors
    
    schema = resource.get('schema', {})
    fields = {f['name']: f for f in schema.get('fields', [])}
    primary_key = schema.get('primaryKey', [])
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            original_headers = reader.fieldnames or []
            
            # Handle empty header (first column might be unnamed)
            # Map empty headers to expected field names if they match position
            header_map = {}
            field_list = list(fields.keys())
            for i, h in enumerate(original_headers):
                if h and h.strip():
                    header_map[h] = h
                elif i < len(field_list):
                    # Map empty header to first expected field if position matches
                    header_map[field_list[0]] = h or f'_unnamed_{i}'
            
            # Check required fields exist
            for field_name in fields.keys():
                if field_name not in original_headers:
                    # Check if it's mapped from an empty header
                    if field_name in header_map:
                        continue
                    errors.append(f"{resource_name}: Missing field '{field_name}' in CSV")
            
            # Validate rows
            seen_keys = set()
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                # Handle empty header mapping
                normalized_row = {}
                for field_name in fields.keys():
                    if field_name in row:
                        normalized_row[field_name] = row[field_name]
                    elif field_name in header_map:
                        # Get value from mapped header
                        mapped_header = header_map[field_name]
                        if mapped_header in row:
                            normalized_row[field_name] = row[mapped_header]
                        elif mapped_header.startswith('_unnamed_'):
                            # Get first column value
                            first_col = list(row.keys())[0] if row else None
                            if first_col:
                                normalized_row[field_name] = row[first_col]
                row.update(normalized_row)
                
                # Validate field types
                for field_name, field_def in fields.items():
                    if field_name in row:
                        value = row[field_name]
                        if not validate_type(value, field_def.get('type', 'string')):
                            errors.append(
                                f"{resource_name}: Row {row_num}, field '{field_name}': "
                                f"Invalid type '{field_def['type']}' for value '{value}'"
                            )
                
                # Validate primary key (skip if key is empty)
                if primary_key:
                    key_values = tuple(row.get(k, '') for k in primary_key)
                    # Skip validation if all key values are empty
                    if all(not v or v.strip() == '' for v in key_values):
                        continue
                    valid, msg = validate_primary_key(row, primary_key, seen_keys)
                    if not valid:
                        errors.append(f"{resource_name}: Row {row_num}: {msg}")
    
    except Exception as e:
        errors.append(f"{resource_name}: Error reading file: {str(e)}")
    
    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(
        description='Validate EPM input data against datapackage.json schema'
    )
    parser.add_argument(
        '--data-folder',
        type=str,
        default='data_test',
        help='Name of the data folder to validate (default: data_test)'
    )
    parser.add_argument(
        '--datapackage',
        type=str,
        default='datapackage.json',
        help='Path to datapackage.json (default: datapackage.json)'
    )
    
    args = parser.parse_args()
    
    # Get script directory and construct paths
    script_dir = Path(__file__).parent
    datapackage_path = script_dir / args.datapackage
    data_folder = script_dir / args.data_folder
    
    if not datapackage_path.exists():
        print(f"Error: datapackage.json not found at {datapackage_path}")
        return 1
    
    if not data_folder.exists():
        print(f"Error: data folder not found at {data_folder}")
        return 1
    
    # Load datapackage
    with open(datapackage_path, 'r') as f:
        datapackage = json.load(f)
    
    # Validate all resources (skip H2-related resources)
    all_errors = []
    validated = 0
    skipped = 0
    
    for resource in datapackage.get('resources', []):
        resource_name = resource.get('name', '')
        resource_path = resource.get('path', '')
        
        # Skip H2-related resources
        if 'H2' in resource_name or 'h2' in resource_name or 'h2/' in resource_path:
            skipped += 1
            continue
        
        is_valid, errors = validate_resource(resource, data_folder)
        if is_valid:
            validated += 1
        else:
            all_errors.extend(errors)
    
    # Report results
    total = len(datapackage.get('resources', []))
    print(f"Validated {validated}/{total} resources (skipped {skipped} H2-related resources)")
    
    if all_errors:
        print(f"\n{len(all_errors)} error(s) found:")
        for error in all_errors:
            print(f"  - {error}")
        return 1
    
    print("✓ All resources validated successfully")
    return 0


if __name__ == '__main__':
    exit(main())

