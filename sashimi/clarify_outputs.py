#!/usr/bin/env python3
"""
Script to organize scattered model training outputs into a structured directory hierarchy.
Creates symlinks in outputs_clarified/ organized by wandb group names and run metadata.
"""

import argparse
import os
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Organize training outputs into clarified directory structure')
    parser.add_argument('--reset', action='store_true', 
                       help='Remove outputs_clarified directory before processing')
    args = parser.parse_args()
    
    # Get script directory and validate structure
    script_dir = Path(__file__).parent.absolute()
    outputs_dir = script_dir / "s4" / "outputs"
    clarified_dir = script_dir / "outputs_clarified"
    
    # Validate outputs directory exists
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")
    
    # Handle reset functionality
    if args.reset:
        if clarified_dir.exists():
            print(f"Removing existing outputs_clarified directory...")
            shutil.rmtree(clarified_dir)
        clarified_dir.mkdir(parents=True, exist_ok=True)
        print("Reset complete. Starting fresh clarification.")
    else:
        # Create clarified directory if it doesn't exist
        clarified_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze existing symlinks (if not in reset mode)
    already_clarified = set()
    if not args.reset and clarified_dir.exists():
        print("Analyzing existing symlinks...")
        for group_dir in clarified_dir.iterdir():
            if group_dir.is_dir():
                for symlink in group_dir.iterdir():
                    if symlink.is_symlink():
                        try:
                            target_path = symlink.resolve()
                            already_clarified.add(str(target_path))
                        except OSError:
                            # Handle broken symlinks
                            continue
    
    # Find all datetime subdirectories
    print("Discovering datetime subdirectories...")
    datetime_dirs = []
    
    for date_dir in outputs_dir.iterdir():
        if not date_dir.is_dir():
            continue

        # Find time subdirectories
        for time_dir in date_dir.iterdir():
            if not time_dir.is_dir():
                continue
            
            # Check if already clarified
            abs_path = str(time_dir.absolute())
            if abs_path not in already_clarified:
                datetime_dirs.append(time_dir)
    
    print(f"Found {len(datetime_dirs)} new datetime subdirectories to process")
    
    if not datetime_dirs:
        print("No new directories to process.")
        return
    
    # Process each datetime subdirectory
    processed_count = 0
    error_count = 0
    
    for datetime_dir in tqdm(datetime_dirs, desc="Processing directories"):
        try:
            process_datetime_directory(datetime_dir, clarified_dir)
            processed_count += 1
        except Exception as e:
            print(f"\nError processing {datetime_dir}: {e}")
            error_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count}")
    if error_count > 0:
        print(f"Errors encountered: {error_count}")


def process_datetime_directory(datetime_dir, clarified_dir):
    """Process a single datetime subdirectory."""
    
    # Validate config file exists
    config_file = datetime_dir / ".hydra" / "config.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    # Read and parse config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'wandb' not in config:
        raise KeyError(f"No wandb section found in config: {config_file}")
    
    wandb_config = config['wandb']
    
    # Determine group directory
    group = wandb_config.get('group')
    if group is None or group == '':
        group = "null"
    
    group_dir = clarified_dir / group
    group_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique run name
    date_str = datetime_dir.parent.name  # YYYY-MM-DD
    time_str = datetime_dir.name        # HH-MM-SS-XXXXXX
    
    # Remove dashes to create datetime identifier
    datetime_id = date_str.replace('-', '') + time_str.replace('-', '')
    
    # Get name from wandb config
    name = wandb_config.get('name')
    if name is not None:
        run_name = f"{name}-{datetime_id}"
    else:
        run_name = datetime_id
    
    # Check for duplicate run names
    symlink_path = group_dir / run_name
    if symlink_path.exists():
        raise FileExistsError(f"Run name already exists: {symlink_path}")
    
    # Create symlink
    target_path = datetime_dir.absolute()
    symlink_path.symlink_to(target_path)


if __name__ == "__main__":
    main()
