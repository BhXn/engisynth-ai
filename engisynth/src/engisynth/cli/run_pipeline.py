# import argparse
# import pandas as pd
# import yaml
# import os
# import json
# from datetime import datetime
#
# from ..config.load import load_config
# from ..data.io import read_table
# from ..generators.ctgan import CTGANAdapter
# from ..generators.constrained_ctgan import ConstrainedCTGAN
# from ..evaluation.metrics import delta_r
# from ..constraints.manager import ConstraintManager
#
#
# def create_generator(cfg):
#     """根据配置创建生成器"""
#     if cfg.generator.type == "ctgan":
#         return CTGANAdapter(
#             epochs=cfg.generator.epochs,
#             batch_size=cfg.generator.batch_size
#         )
#     elif cfg.generator.type == "constrained_ctgan":
#         # 准备噪声配置
#         noise_config = {}
#         for col, noise_cfg in cfg.noise.items():
#             noise_config[col] = noise_cfg.model_dump()
#
#         # 准备归一化配置
#         norm_config = {}
#         for col, norm_cfg in cfg.normalization.items():
#             norm_config[col] = norm_cfg.model_dump()
#
#         # 准备约束配置
#         constraints_cfg = [c.model_dump() for c in cfg.constraints]
#
#         return ConstrainedCTGAN(
#             epochs=cfg.generator.epochs,
#             batch_size=cfg.generator.batch_size,
#             constraints_cfg=constraints_cfg,
#             noise_config=noise_config,
#             normalization_config=norm_config
#         )
#     else:
#         raise ValueError(f"Unknown generator type: {cfg.generator.type}")
#
#
# def evaluate_constraints(df, constraint_manager):
#     """评估数据的约束满足情况"""
#     if constraint_manager is None:
#         return {}
#
#     return constraint_manager.evaluate(df)
#
#
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--csv", required=True, help="输入数据文件路径")
#     ap.add_argument("--cfg", required=True, help="配置文件路径")
#     ap.add_argument("--output", required=True, help="输出目录")
#     ap.add_argument("--verbose", action="store_true", help="显示详细信息")
#     args = ap.parse_args()
#
#     # 加载配置和数据
#     print("Loading configuration...")
#     cfg = load_config(args.cfg)
#
#     print(f"Reading data from {args.csv}...")
#     df = read_table(args.csv)
#     print(f"Data shape: {df.shape}")
#
#     # 创建生成器
#     print(f"Creating {cfg.generator.type} generator...")
#     gen = create_generator(cfg)
#
#     # 训练生成器
#     print("Training generator...")
#     gen.fit(df)
#
#     # 生成合成数据
#     print(f"Generating {len(df)} synthetic samples...")
#     synth = gen.sample(len(df))
#
#     if len(synth) == 0:
#         print("Error: Failed to generate valid samples")
#         return
#
#     print(f"Generated {len(synth)} valid samples")
#
#     # 评估质量
#     print("Evaluating synthetic data quality...")
#
#     # 1. 下游任务性能评估
#     task_type = "classification" if df[cfg.target].dtype == 'O' or df[cfg.target].nunique() < 10 else "regression"
#     dr = delta_r(df, synth, cfg.target, task=task_type)
#
#     # 2. 约束满足度评估
#     constraint_manager = ConstraintManager(cfg.constraints) if cfg.constraints else None
#     orig_constraints = evaluate_constraints(df, constraint_manager)
#     synth_constraints = evaluate_constraints(synth, constraint_manager)
#
#     # 创建输出目录
#     os.makedirs(args.output, exist_ok=True)
#
#     # 保存数据
#     print(f"Saving results to {args.output}...")
#     df.to_csv(f"{args.output}/original.csv", index=False)
#     synth.to_csv(f"{args.output}/synthetic.csv", index=False)
#
#     # 生成详细报告
#     report = {
#         "timestamp": datetime.now().isoformat(),
#         "config_file": args.cfg,
#         "input_file": args.csv,
#         "generator_type": cfg.generator.type,
#         "data_shape": list(df.shape),
#         "synthetic_shape": list(synth.shape),
#         "task_type": task_type,
#         "delta_r": float(dr),
#         "constraints": {
#             "original": orig_constraints,
#             "synthetic": synth_constraints
#         }
#     }
#
#     # 计算基本统计信息
#     if args.verbose:
#         report["statistics"] = {
#             "original": {
#                 col: {
#                     "mean": float(df[col].mean()) if df[col].dtype in ['int64', 'float64'] else None,
#                     "std": float(df[col].std()) if df[col].dtype in ['int64', 'float64'] else None,
#                     "min": float(df[col].min()) if df[col].dtype in ['int64', 'float64'] else None,
#                     "max": float(df[col].max()) if df[col].dtype in ['int64', 'float64'] else None,
#                 }
#                 for col in df.columns if col != cfg.target
#             },
#             "synthetic": {
#                 col: {
#                     "mean": float(synth[col].mean()) if synth[col].dtype in ['int64', 'float64'] else None,
#                     "std": float(synth[col].std()) if synth[col].dtype in ['int64', 'float64'] else None,
#                     "min": float(synth[col].min()) if synth[col].dtype in ['int64', 'float64'] else None,
#                     "max": float(synth[col].max()) if synth[col].dtype in ['int64', 'float64'] else None,
#                 }
#                 for col in synth.columns if col != cfg.target
#             }
#         }
#
#     # 保存报告
#     with open(f"{args.output}/report.json", "w") as f:
#         json.dump(report, f, indent=2)
#
#     # 简化的文本报告
#     with open(f"{args.output}/report.txt", "w") as f:
#         f.write("=== Synthetic Data Generation Report ===\n\n")
#         f.write(f"Timestamp: {report['timestamp']}\n")
#         f.write(f"Generator: {cfg.generator.type}\n")
#         f.write(f"Input shape: {df.shape}\n")
#         f.write(f"Output shape: {synth.shape}\n")
#         f.write(f"Task type: {task_type}\n")
#         f.write(f"\nΔR (relative performance difference): {dr:.4f}\n")
#         f.write("(Lower is better, 0 means perfect reproduction)\n")
#
#         if constraint_manager:
#             f.write("\n=== Constraint Satisfaction ===\n")
#             f.write("\nOriginal data:\n")
#             for name, score in orig_constraints.items():
#                 f.write(f"  {name}: {score:.4f}\n")
#             f.write("\nSynthetic data:\n")
#             for name, score in synth_constraints.items():
#                 f.write(f"  {name}: {score:.4f}\n")
#
#     # 打印摘要
#     print("\n=== Summary ===")
#     print(f"ΔR = {dr:.4f}")
#     if constraint_manager:
#         print(f"Original constraints satisfaction: {orig_constraints.get('overall', 1.0):.4f}")
#         print(f"Synthetic constraints satisfaction: {synth_constraints.get('overall', 1.0):.4f}")
#     print(f"\nResults saved to {args.output}/")
#
#
# if __name__ == "__main__":
#     main()

import argparse
import pandas as pd
import yaml
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

from ..config.load import load_config
from ..data.io import read_table
from ..evaluation.metrics import delta_r
from ..constraints.manager import ConstraintManager

# 导入所有生成器
from ..generators.factory import get_generator
from ..generators.constrained_factory import get_constrained_generator
from ..generators.ctgan import CTGANAdapter
from ..generators.constrained_ctgan import ConstrainedCTGAN
from ..generators.copulagan import CopulaGANAdapter
from ..generators.constrained_copulagan import ConstrainedCopulaGAN
from ..generators.gaussian_copula import GaussianCopulaAdapter
from ..generators.constrained_gaussian_copula import ConstrainedGaussianCopula
from ..generators.tvae import TVAEAdapter
from ..generators.constrained_tvae import ConstrainedTVAE
from ..generators.tabddpm import TabDDPMAdapter
from ..generators.constrained_tabddpm import ConstrainedTabDDPM
from ..generators.tabpfn import TabPFNGenerativeAdapter

# 生成器配置映射
GENERATOR_CONFIGS = {
    # 标准生成器
    "ctgan": {
        "class": CTGANAdapter,
        "params": ["epochs", "batch_size", "generator_dim", "discriminator_dim",
                   "generator_lr", "discriminator_lr", "pac"]
    },
    "copulagan": {
        "class": CopulaGANAdapter,
        "params": ["epochs", "batch_size", "generator_dim", "discriminator_dim",
                   "generator_lr", "discriminator_lr"]
    },
    "gaussian_copula": {
        "class": GaussianCopulaAdapter,
        "params": ["categorical_transformer", "numerical_distributions",
                   "default_distribution"]
    },
    "tvae": {
        "class": TVAEAdapter,
        "params": ["epochs", "batch_size", "encoder_dim", "decoder_dim",
                   "latent_dim", "learning_rate"]
    },
    "tabddpm": {
        "class": TabDDPMAdapter,
        "params": ["steps", "lr", "num_timesteps", "gaussian_loss_type",
                   "scheduler", "num_numerical_features"]
    },
    "tabpfn": {
        "class": TabPFNGenerativeAdapter,
        "params": ["config_name", "bootstrap_samples", "noise_scale",
                   "interpolation_frac"]
    },

    # 约束版本生成器
    "constrained_ctgan": {
        "class": ConstrainedCTGAN,
        "params": ["epochs", "batch_size", "generator_dim", "discriminator_dim",
                   "constraints_cfg", "noise_config", "normalization_config"]
    },
    "constrained_copulagan": {
        "class": ConstrainedCopulaGAN,
        "params": ["epochs", "batch_size", "generator_dim", "discriminator_dim",
                   "constraints"]
    },
    "constrained_gaussian_copula": {
        "class": ConstrainedGaussianCopula,
        "params": ["categorical_transformer", "numerical_distributions",
                   "constraints", "max_rejection_samples"]
    },
    "constrained_tvae": {
        "class": ConstrainedTVAE,
        "params": ["epochs", "batch_size", "encoder_dim", "decoder_dim",
                   "constraints", "max_rejection_samples"]
    },
    "constrained_tabddpm": {
        "class": ConstrainedTabDDPM,
        "params": ["steps", "lr", "num_timesteps", "constraints",
                   "max_rejection_samples"]
    }
}


def extract_generator_params(cfg, generator_type: str) -> Dict[str, Any]:
    """从配置中提取特定生成器所需的参数"""
    params = {}

    # 获取生成器配置中定义的参数
    if hasattr(cfg.generator, 'params') and cfg.generator.params:
        # 如果配置文件中有显式的params字段
        params.update(cfg.generator.params)

    # 获取生成器级别的参数
    generator_attrs = dir(cfg.generator)
    valid_params = GENERATOR_CONFIGS.get(generator_type, {}).get("params", [])

    for param in valid_params:
        if param in generator_attrs:
            value = getattr(cfg.generator, param, None)
            if value is not None:
                params[param] = value

    # 特殊处理约束版本的参数
    if "constrained" in generator_type:
        # 处理约束配置
        if hasattr(cfg, 'constraints') and cfg.constraints:
            if generator_type == "constrained_ctgan":
                # ConstrainedCTGAN 需要特殊格式
                params["constraints_cfg"] = [c.model_dump() for c in cfg.constraints]

                # 添加噪声配置
                if hasattr(cfg, 'noise'):
                    noise_config = {}
                    for col, noise_cfg in cfg.noise.items():
                        noise_config[col] = noise_cfg.model_dump()
                    params["noise_config"] = noise_config

                # 添加归一化配置
                if hasattr(cfg, 'normalization'):
                    norm_config = {}
                    for col, norm_cfg in cfg.normalization.items():
                        norm_config[col] = norm_cfg.model_dump()
                    params["normalization_config"] = norm_config
            else:
                # 其他约束生成器直接使用constraints
                params["constraints"] = cfg.constraints

        # 添加最大拒绝采样数
        if hasattr(cfg.generator, 'max_rejection_samples'):
            params["max_rejection_samples"] = cfg.generator.max_rejection_samples

    return params


def create_generator(cfg):
    """根据配置创建生成器（支持所有类型）"""
    generator_type = cfg.generator.type.lower()

    # 验证生成器类型
    if generator_type not in GENERATOR_CONFIGS:
        available = list(GENERATOR_CONFIGS.keys())
        raise ValueError(
            f"Unknown generator type: {generator_type}. "
            f"Available types: {', '.join(available)}"
        )

    # 获取生成器类和参数
    generator_class = GENERATOR_CONFIGS[generator_type]["class"]
    params = extract_generator_params(cfg, generator_type)

    # 打印将要使用的参数（调试用）
    print(f"Creating {generator_type} with parameters:")
    for key, value in params.items():
        if key not in ["constraints_cfg", "noise_config", "normalization_config"]:
            print(f"  {key}: {value}")

    try:
        # 使用工厂方法或直接实例化
        if "constrained" in generator_type and hasattr(cfg, 'use_factory') and cfg.use_factory:
            # 使用约束工厂方法
            base_type = generator_type.replace("constrained_", "")
            constraints = params.pop("constraints", [])
            return get_constrained_generator(base_type, constraints, **params)
        elif "constrained" not in generator_type and hasattr(cfg, 'use_factory') and cfg.use_factory:
            # 使用标准工厂方法
            return get_generator(generator_type, **params)
        else:
            # 直接实例化
            return generator_class(**params)
    except Exception as e:
        print(f"Error creating generator: {e}")
        print(f"Parameters passed: {params}")
        raise


def evaluate_constraints(df: pd.DataFrame, constraint_manager: Optional[ConstraintManager]) -> Dict:
    """评估数据的约束满足情况"""
    if constraint_manager is None:
        return {}

    try:
        return constraint_manager.evaluate(df)
    except Exception as e:
        print(f"Warning: Failed to evaluate constraints: {e}")
        return {"error": str(e)}


def generate_model_specific_metrics(df: pd.DataFrame, synth: pd.DataFrame,
                                    generator_type: str, cfg) -> Dict[str, Any]:
    """根据不同的生成器类型生成特定的评估指标"""
    metrics = {}

    # 对于 Copula 类模型，评估相关性保持
    if "copula" in generator_type.lower():
        try:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 1:
                orig_corr = df[numeric_cols].corr()
                synth_corr = synth[numeric_cols].corr()
                corr_diff = (orig_corr - synth_corr).abs().mean().mean()
                metrics["correlation_difference"] = float(corr_diff)
        except Exception as e:
            print(f"Warning: Failed to compute correlation metrics: {e}")

    # 对于 GAN 类模型，可以添加 mode collapse 检测
    if "gan" in generator_type.lower():
        try:
            # 简单的多样性度量：unique rows ratio
            orig_unique_ratio = len(df.drop_duplicates()) / len(df)
            synth_unique_ratio = len(synth.drop_duplicates()) / len(synth)
            metrics["unique_ratio_original"] = float(orig_unique_ratio)
            metrics["unique_ratio_synthetic"] = float(synth_unique_ratio)
        except Exception as e:
            print(f"Warning: Failed to compute diversity metrics: {e}")

    # 对于 TabPFN，记录使用的生成策略
    if "tabpfn" in generator_type.lower():
        if hasattr(cfg.generator, 'generation_method'):
            metrics["generation_method"] = cfg.generator.generation_method

    return metrics


def main():
    ap = argparse.ArgumentParser(
        description="运行合成数据生成流水线，支持多种生成器模型"
    )
    ap.add_argument("--csv", required=True, help="输入数据文件路径")
    ap.add_argument("--cfg", required=True, help="配置文件路径")
    ap.add_argument("--output", required=True, help="输出目录")
    ap.add_argument("--verbose", action="store_true", help="显示详细信息")
    ap.add_argument("--list-generators", action="store_true",
                    help="列出所有可用的生成器类型")
    ap.add_argument("--dry-run", action="store_true",
                    help="仅验证配置，不执行实际训练和生成")
    ap.add_argument("--n-samples", type=int, default=None,
                    help="要生成的合成样本数量。如果未指定，则使用配置文件中的设置或原始数据大小。")
    args = ap.parse_args()

    # 如果请求列出生成器，则显示并退出
    if args.list_generators:
        print("\n=== Available Generators ===\n")
        print("Standard Generators:")
        for name in GENERATOR_CONFIGS.keys():
            if "constrained" not in name:
                print(f"  - {name}")
        print("\nConstrained Generators:")
        for name in GENERATOR_CONFIGS.keys():
            if "constrained" in name:
                print(f"  - {name}")
        print("\nUse these names in your config file's generator.type field.")
        return

    # 加载配置和数据
    print("=" * 60)
    print("SYNTHETIC DATA GENERATION PIPELINE")
    print("=" * 60)
    print(f"\nLoading configuration from: {args.cfg}")
    cfg = load_config(args.cfg)

    print(f"Reading data from: {args.csv}")
    df = read_table(args.csv)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {', '.join(df.columns)}")

    # 数据基本信息
    if args.verbose:
        print("\nData types:")
        for col, dtype in df.dtypes.items():
            print(f"  {col}: {dtype}")
        print(f"\nMissing values: {df.isnull().sum().sum()}")

    # 验证目标列
    if not hasattr(cfg, 'target') or cfg.target not in df.columns:
        print(f"Warning: Target column '{cfg.target}' not found in data")
        cfg.target = df.columns[-1]  # 默认使用最后一列
        print(f"Using '{cfg.target}' as target column")

    # 创建生成器
    print(f"\n--- Generator Configuration ---")
    print(f"Type: {cfg.generator.type}")

    if args.dry_run:
        print("\nDry run mode - validating configuration only...")
        try:
            gen = create_generator(cfg)
            print("✓ Configuration is valid")
            print(f"✓ Generator created: {type(gen).__name__}")
        except Exception as e:
            print(f"✗ Configuration error: {e}")
            return 1
        print("\nDry run complete. Exiting.")
        return 0

    gen = create_generator(cfg)
    print(f"✓ Created {type(gen).__name__}")

    # 训练生成器
    print("\n--- Training Phase ---")
    print("Training generator...")
    start_time = datetime.now()
    gen.fit(df)
    train_time = (datetime.now() - start_time).total_seconds()
    print(f"✓ Training completed in {train_time:.2f} seconds")

    # 生成合成数据
    print("\n--- Generation Phase ---")
    # 优先级：命令行 --n-samples > 配置文件 cfg.generator.n_samples > 原始数据行数
    n_samples = args.n_samples if args.n_samples is not None else getattr(cfg.generator, 'n_samples', None)
    if n_samples is None:
        n_samples = len(df)
    print(f"Generating {n_samples} synthetic samples...")

    start_time = datetime.now()
    synth = gen.sample(n_samples)
    gen_time = (datetime.now() - start_time).total_seconds()

    if len(synth) == 0:
        print("✗ Error: Failed to generate valid samples")
        return 1

    print(f"✓ Generated {len(synth)} valid samples in {gen_time:.2f} seconds")

    # 评估质量
    print("\n--- Evaluation Phase ---")
    print("Evaluating synthetic data quality...")

    # 1. 下游任务性能评估
    task_type = "classification" if df[cfg.target].dtype == 'O' or df[cfg.target].nunique() < 10 else "regression"
    print(f"Task type: {task_type}")
    dr = delta_r(df, synth, cfg.target, task=task_type)
    print(f"ΔR (delta-R): {dr:.4f}")

    # 2. 约束满足度评估
    constraint_manager = ConstraintManager(cfg.constraints) if hasattr(cfg, 'constraints') and cfg.constraints else None
    orig_constraints = evaluate_constraints(df, constraint_manager)
    synth_constraints = evaluate_constraints(synth, constraint_manager)

    if constraint_manager:
        print(f"Constraints evaluated: {len(orig_constraints)} metrics")

    # 3. 模型特定指标
    model_metrics = generate_model_specific_metrics(df, synth, cfg.generator.type, cfg)
    if model_metrics:
        print(f"Model-specific metrics computed: {len(model_metrics)}")

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 保存数据
    print(f"\n--- Saving Results ---")
    print(f"Output directory: {args.output}")

    df.to_csv(f"{args.output}/original.csv", index=False)
    synth.to_csv(f"{args.output}/synthetic.csv", index=False)
    print("✓ Data files saved")

    # 生成详细报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "config_file": args.cfg,
        "input_file": args.csv,
        "generator": {
            "type": cfg.generator.type,
            "class": type(gen).__name__,
            "params": extract_generator_params(cfg, cfg.generator.type)
        },
        "data": {
            "original_shape": list(df.shape),
            "synthetic_shape": list(synth.shape),
            "columns": list(df.columns),
            "target_column": cfg.target
        },
        "performance": {
            "task_type": task_type,
            "delta_r": float(dr),
            "training_time_seconds": train_time,
            "generation_time_seconds": gen_time,
            "samples_per_second": len(synth) / gen_time if gen_time > 0 else 0
        },
        "model_specific_metrics": model_metrics,
        "constraints": {
            "original": orig_constraints,
            "synthetic": synth_constraints
        }
    }

    # 计算基本统计信息
    if args.verbose:
        print("Computing detailed statistics...")
        report["statistics"] = {
            "original": {},
            "synthetic": {}
        }

        for col in df.columns:
            if col == cfg.target:
                continue

            # 原始数据统计
            if df[col].dtype in ['int64', 'float64']:
                report["statistics"]["original"][col] = {
                    "type": "numeric",
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "median": float(df[col].median()),
                    "q25": float(df[col].quantile(0.25)),
                    "q75": float(df[col].quantile(0.75))
                }
            else:
                value_counts = df[col].value_counts()
                report["statistics"]["original"][col] = {
                    "type": "categorical",
                    "unique_values": int(df[col].nunique()),
                    "top_5_values": {str(k): int(v) for k, v in value_counts.head(5).items()}
                }

            # 合成数据统计
            if synth[col].dtype in ['int64', 'float64']:
                report["statistics"]["synthetic"][col] = {
                    "type": "numeric",
                    "mean": float(synth[col].mean()),
                    "std": float(synth[col].std()),
                    "min": float(synth[col].min()),
                    "max": float(synth[col].max()),
                    "median": float(synth[col].median()),
                    "q25": float(synth[col].quantile(0.25)),
                    "q75": float(synth[col].quantile(0.75))
                }
            else:
                value_counts = synth[col].value_counts()
                report["statistics"]["synthetic"][col] = {
                    "type": "categorical",
                    "unique_values": int(synth[col].nunique()),
                    "top_5_values": {str(k): int(v) for k, v in value_counts.head(5).items()}
                }

    # 保存JSON报告
    with open(f"{args.output}/report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("✓ JSON report saved")

    # 生成可读的文本报告
    with open(f"{args.output}/report.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("SYNTHETIC DATA GENERATION REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Timestamp: {report['timestamp']}\n")
        f.write(f"Generator: {cfg.generator.type} ({type(gen).__name__})\n")
        f.write(f"Input file: {args.csv}\n")
        f.write(f"Config file: {args.cfg}\n\n")

        f.write("--- Data Information ---\n")
        f.write(f"Original shape: {df.shape}\n")
        f.write(f"Synthetic shape: {synth.shape}\n")
        f.write(f"Target column: {cfg.target}\n")
        f.write(f"Task type: {task_type}\n\n")

        f.write("--- Performance Metrics ---\n")
        f.write(f"ΔR (relative performance difference): {dr:.4f}\n")
        f.write("(Lower is better, 0 means perfect reproduction)\n")
        f.write(f"Training time: {train_time:.2f} seconds\n")
        f.write(f"Generation time: {gen_time:.2f} seconds\n")
        f.write(f"Generation rate: {len(synth) / gen_time:.1f} samples/second\n\n")

        # 模型特定指标
        if model_metrics:
            f.write("--- Model-Specific Metrics ---\n")
            for key, value in model_metrics.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            f.write("\n")

        # 约束满足度
        if constraint_manager and orig_constraints:
            f.write("--- Constraint Satisfaction ---\n\n")
            f.write("Original data:\n")
            for name, score in orig_constraints.items():
                if name != "error":
                    f.write(f"  {name}: {score:.4f}\n")

            f.write("\nSynthetic data:\n")
            for name, score in synth_constraints.items():
                if name != "error":
                    f.write(f"  {name}: {score:.4f}\n")
            f.write("\n")

        # 详细统计（如果启用verbose）
        if args.verbose and "statistics" in report:
            f.write("--- Column Statistics ---\n\n")
            for col in df.columns:
                if col == cfg.target:
                    continue

                f.write(f"Column: {col}\n")
                orig_stats = report["statistics"]["original"].get(col, {})
                synth_stats = report["statistics"]["synthetic"].get(col, {})

                if orig_stats.get("type") == "numeric":
                    f.write("  Original  -> Synthetic\n")
                    f.write(f"  Mean:     {orig_stats['mean']:.3f} -> {synth_stats['mean']:.3f}\n")
                    f.write(f"  Std:      {orig_stats['std']:.3f} -> {synth_stats['std']:.3f}\n")
                    f.write(f"  Min:      {orig_stats['min']:.3f} -> {synth_stats['min']:.3f}\n")
                    f.write(f"  Max:      {orig_stats['max']:.3f} -> {synth_stats['max']:.3f}\n")
                else:
                    f.write(f"  Unique values: {orig_stats.get('unique_values', 'N/A')} -> "
                            f"{synth_stats.get('unique_values', 'N/A')}\n")
                f.write("\n")

    print("✓ Text report saved")

    # 打印摘要
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Generator: {cfg.generator.type}")
    print(f"ΔR Score: {dr:.4f}")
    print(f"Time: {train_time + gen_time:.2f}s total ({train_time:.1f}s train, {gen_time:.1f}s generate)")

    if constraint_manager and "overall" in orig_constraints:
        print(f"Constraint Satisfaction:")
        print(f"  Original:  {orig_constraints.get('overall', 1.0):.4f}")
        print(f"  Synthetic: {synth_constraints.get('overall', 1.0):.4f}")

    if model_metrics:
        print("Model-Specific Metrics:")
        for key, value in list(model_metrics.items())[:3]:  # 只显示前3个
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")

    print(f"\n✓ All results saved to: {args.output}/")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())