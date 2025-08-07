"""
CLI tool to automatically generate a configuration file from a dataset.

This script inspects a CSV file, detects potential constraints and feature types,
and interactively prompts the user to confirm them before generating a YAML
configuration file.
"""
import argparse
import pandas as pd
import yaml
import sys
import os
from typing import List, Dict, Any

# Adjust the path to import from the parent package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from engisynth.constraints.detector import ConstraintDetector, auto_detect_feature_types

def interactive_confirm(constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Interactively prompts the user to confirm or discard detected constraints.
    """
    confirmed_constraints = []
    if not constraints:
        print("No significant constraints detected.")
        return confirmed_constraints

    print("\n---  Interactive Constraint Confirmation ---")
    print("Please confirm the following detected constraints. [y/N]")

    for i, const in enumerate(constraints):
        const_type = const['type']
        cols = ', '.join(const['cols'])
        
        details = ""
        if const_type == 'range':
            details = f"min: {const.get('min')}, max: {const.get('max')}"
        elif const_type == 'sum_equal':
            details = f"sum ≈ {const.get('value')}"
        elif const_type == 'category_allowed':
            values = const.get('values', [])
            details = f"values: {{{', '.join(map(str, values[:5]))}" + (", ..." if len(values) > 5 else "") + "}"

        prompt = f"  [{i+1}/{len(constraints)}] {const_type.upper():<18} | On: {cols:<25} | Details: {details:<30} | Keep? [y/N]: "
        
        try:
            user_input = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted by user. Exiting.")
            sys.exit(0)

        if user_input == 'y':
            confirmed_constraints.append(const)
            print(f"    ->  Kept.")
        else:
            print(f"    ->  Discarded.")
            
    return confirmed_constraints

def main():
    """
    Main function to run the configuration creation process.
    """
    parser = argparse.ArgumentParser(
        description="Automatically detect constraints from a CSV file and generate a configuration YAML.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--csv", required=True, help="Path to the input CSV file.")
    parser.add_argument("--out", required=True, help="Path to save the generated YAML configuration file.")
    parser.add_argument("--target", required=False, default=None, help="(Optional) Name of the target column for modeling.")
    
    args = parser.parse_args()

    print(f"📄 Loading data from '{args.csv}'...")
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f" Error: The file '{args.csv}' was not found.", file=sys.stderr)
        sys.exit(1)
    
    # 1. Auto-detect constraints
    detector = ConstraintDetector(df)
    detected_constraints = detector.detect_all()
    
    # 2. Interactive confirmation
    confirmed_constraints = interactive_confirm(detected_constraints)
    
    # 3. Auto-detect feature types
    print("\n---   Auto-Detecting Feature Types ---")
    feature_types = auto_detect_feature_types(df)
    if args.target and args.target not in feature_types:
        print(f" Warning: Target column '{args.target}' not found in the dataframe columns. It will be ignored.")
        target_col = None
    else:
        target_col = args.target
    print(f" Done. Found types for {len(feature_types)} columns.")

    # 4. Build the final YAML configuration dictionary
    final_config = {
        'dataset': os.path.abspath(args.csv),
        'target': target_col,
        'feature_types': feature_types,
        'constraints': confirmed_constraints,
        'generator': {
            'type': 'ctgan',
            'epochs': 300,
            'batch_size': 64
        },
        'noise': {},
        'normalization': {}
    }
    
    # 5. Write to file
    print(f"\n---  Saving Configuration ---")
    try:
        with open(args.out, 'w', encoding='utf-8') as f:
            yaml.dump(final_config, f, sort_keys=False, indent=2, default_flow_style=False)
        print(f" Successfully generated configuration file at '{args.out}'!")
    except Exception as e:
        print(f" Error: Could not write to file '{args.out}'. Reason: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
