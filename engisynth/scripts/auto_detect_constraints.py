import argparse
import pandas as pd
import yaml
from typing import List, Dict, Any

# 假设 detector.py 在 engisynth.constraints 模块下
# 为了让这个脚本能独立运行，需要处理一下 Python 的模块搜索路径
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from engisynth.src.engisynth.constraints.detector import ConstraintDetector, auto_detect_feature_types

def interactive_confirm(constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    通过命令行交互，让用户确认是否保留每个探测到的约束。
    """
    confirmed_constraints = []
    if not constraints:
        print("No constraints detected.")
        return confirmed_constraints

    print("\n--- Interactive Constraint Confirmation ---")
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

        prompt = f"[{i+1}/{len(constraints)}] {const_type.upper():<18} | On columns: {cols:<25} | Details: {details:<30} | Keep? [y/N]: "
        
        user_input = input(prompt).strip().lower()
        if user_input == 'y':
            confirmed_constraints.append(const)
            print(f"  -> Kept.")
        else:
            print(f"  -> Discarded.")
            
    return confirmed_constraints

def main():
    parser = argparse.ArgumentParser(
        description="Automatically detect constraints from a CSV file and generate a configuration YAML.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--csv", required=True, help="Path to the input CSV file.")
    parser.add_argument("--out", required=True, help="Path to save the generated YAML configuration file.")
    parser.add_argument("--target", required=False, default=None, help="(Optional) Name of the target column for modeling.")
    
    args = parser.parse_args()

    print(f"Loading data from {args.csv}...")
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"Error: The file '{args.csv}' was not found.")
        sys.exit(1)
    
    # 1. 自动探测
    detector = ConstraintDetector(df)
    detected_constraints = detector.detect_all()
    
    # 2. 交互式确认
    confirmed_constraints = interactive_confirm(detected_constraints)
    
    # 3. 自动生成特征类型
    print("\n  Detecting feature types...")
    feature_types = auto_detect_feature_types(df)
    if args.target and args.target not in feature_types:
        print(f"Warning: Target column '{args.target}' not found in the dataframe columns.")
        target_col = None
    else:
        target_col = args.target
    print(f"Done. Found {len(feature_types)} columns.")

    # 4. 构建最终的 YAML 配置字典
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
    
    # 5. 写入文件
    print(f"\n Saving confirmed configuration to {args.out}...")
    try:
        with open(args.out, 'w', encoding='utf-8') as f:
            yaml.dump(final_config, f, sort_keys=False, indent=2, default_flow_style=False)
        print("Successfully generated configuration file!")
    except Exception as e:
        print(f"Error: Could not write to file '{args.out}'. Reason: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
