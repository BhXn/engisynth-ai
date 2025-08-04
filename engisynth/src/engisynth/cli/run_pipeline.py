import argparse
import pandas as pd
import yaml
import os
import json
from datetime import datetime

from engisynth.config.load import load_config
from engisynth.data.io import read_table
from engisynth.generators.ctgan import CTGANAdapter
from engisynth.generators.ctgan_constrained import ConstrainedCTGAN
from engisynth.evaluation.metrics import delta_r
from engisynth.constraints.manager import ConstraintManager


def create_generator(cfg):
    """根据配置创建生成器"""
    if cfg.generator.type == "ctgan":
        return CTGANAdapter(
            epochs=cfg.generator.epochs,
            batch_size=cfg.generator.batch_size
        )
    elif cfg.generator.type == "constrained_ctgan":
        # 准备噪声配置
        noise_config = {}
        for col, noise_cfg in cfg.noise.items():
            noise_config[col] = noise_cfg.dict()

        # 准备归一化配置
        norm_config = {}
        for col, norm_cfg in cfg.normalization.items():
            norm_config[col] = norm_cfg.dict()

        # 准备约束配置
        constraints_cfg = [c.dict() for c in cfg.constraints]

        return ConstrainedCTGAN(
            epochs=cfg.generator.epochs,
            batch_size=cfg.generator.batch_size,
            constraints_cfg=constraints_cfg,
            noise_config=noise_config,
            normalization_config=norm_config
        )
    else:
        raise ValueError(f"Unknown generator type: {cfg.generator.type}")


def evaluate_constraints(df, constraint_manager):
    """评估数据的约束满足情况"""
    if constraint_manager is None:
        return {}

    return constraint_manager.evaluate(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="输入数据文件路径")
    ap.add_argument("--cfg", required=True, help="配置文件路径")
    ap.add_argument("--output", required=True, help="输出目录")
    ap.add_argument("--verbose", action="store_true", help="显示详细信息")
    args = ap.parse_args()

    # 加载配置和数据
    print("Loading configuration...")
    cfg = load_config(args.cfg)

    print(f"Reading data from {args.csv}...")
    df = read_table(args.csv)
    print(f"Data shape: {df.shape}")

    # 创建生成器
    print(f"Creating {cfg.generator.type} generator...")
    gen = create_generator(cfg)

    # 训练生成器
    print("Training generator...")
    gen.fit(df)

    # 生成合成数据
    print(f"Generating {len(df)} synthetic samples...")
    synth = gen.sample(len(df))

    if len(synth) == 0:
        print("Error: Failed to generate valid samples")
        return

    print(f"Generated {len(synth)} valid samples")

    # 评估质量
    print("Evaluating synthetic data quality...")

    # 1. 下游任务性能评估
    task_type = "classification" if df[cfg.target].dtype == 'O' or df[cfg.target].nunique() < 10 else "regression"
    dr = delta_r(df, synth, cfg.target, task=task_type)

    # 2. 约束满足度评估
    constraint_manager = ConstraintManager(cfg.constraints) if cfg.constraints else None
    orig_constraints = evaluate_constraints(df, constraint_manager)
    synth_constraints = evaluate_constraints(synth, constraint_manager)

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 保存数据
    print(f"Saving results to {args.output}...")
    df.to_csv(f"{args.output}/original.csv", index=False)
    synth.to_csv(f"{args.output}/synthetic.csv", index=False)

    # 生成详细报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "config_file": args.cfg,
        "input_file": args.csv,
        "generator_type": cfg.generator.type,
        "data_shape": list(df.shape),
        "synthetic_shape": list(synth.shape),
        "task_type": task_type,
        "delta_r": float(dr),
        "constraints": {
            "original": orig_constraints,
            "synthetic": synth_constraints
        }
    }

    # 计算基本统计信息
    if args.verbose:
        report["statistics"] = {
            "original": {
                col: {
                    "mean": float(df[col].mean()) if df[col].dtype in ['int64', 'float64'] else None,
                    "std": float(df[col].std()) if df[col].dtype in ['int64', 'float64'] else None,
                    "min": float(df[col].min()) if df[col].dtype in ['int64', 'float64'] else None,
                    "max": float(df[col].max()) if df[col].dtype in ['int64', 'float64'] else None,
                }
                for col in df.columns if col != cfg.target
            },
            "synthetic": {
                col: {
                    "mean": float(synth[col].mean()) if synth[col].dtype in ['int64', 'float64'] else None,
                    "std": float(synth[col].std()) if synth[col].dtype in ['int64', 'float64'] else None,
                    "min": float(synth[col].min()) if synth[col].dtype in ['int64', 'float64'] else None,
                    "max": float(synth[col].max()) if synth[col].dtype in ['int64', 'float64'] else None,
                }
                for col in synth.columns if col != cfg.target
            }
        }

    # 保存报告
    with open(f"{args.output}/report.json", "w") as f:
        json.dump(report, f, indent=2)

    # 简化的文本报告
    with open(f"{args.output}/report.txt", "w") as f:
        f.write("=== Synthetic Data Generation Report ===\n\n")
        f.write(f"Timestamp: {report['timestamp']}\n")
        f.write(f"Generator: {cfg.generator.type}\n")
        f.write(f"Input shape: {df.shape}\n")
        f.write(f"Output shape: {synth.shape}\n")
        f.write(f"Task type: {task_type}\n")
        f.write(f"\nΔR (relative performance difference): {dr:.4f}\n")
        f.write("(Lower is better, 0 means perfect reproduction)\n")

        if constraint_manager:
            f.write("\n=== Constraint Satisfaction ===\n")
            f.write("\nOriginal data:\n")
            for name, score in orig_constraints.items():
                f.write(f"  {name}: {score:.4f}\n")
            f.write("\nSynthetic data:\n")
            for name, score in synth_constraints.items():
                f.write(f"  {name}: {score:.4f}\n")

    # 打印摘要
    print("\n=== Summary ===")
    print(f"ΔR = {dr:.4f}")
    if constraint_manager:
        print(f"Original constraints satisfaction: {orig_constraints.get('overall', 1.0):.4f}")
        print(f"Synthetic constraints satisfaction: {synth_constraints.get('overall', 1.0):.4f}")
    print(f"\nResults saved to {args.output}/")


if __name__ == "__main__":
    main()

