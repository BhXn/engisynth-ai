import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def generate_hardware_test_data(n_samples=200, seed=42):
    """生成模拟硬件工程数据"""
    np.random.seed(seed)

    data = {}

    # 1. 生成来料参数
    data['material_density'] = np.random.normal(2.7, 0.1, n_samples)  # g/cm³
    data['material_purity'] = np.random.beta(50, 1, n_samples) * 0.05 + 0.95  # 95%-100%
    data['supplier_id'] = np.random.choice(['S001', 'S002', 'S003'], n_samples)

    # 2. 生成制造参数
    data['temperature'] = np.random.uniform(50, 250, n_samples)  # °C
    data['pressure'] = np.random.lognormal(1.0, 0.5, n_samples)  # MPa
    data['pressure'] = np.clip(data['pressure'], 0.1, 10.0)
    data['duration'] = np.random.gamma(2, 30, n_samples)  # minutes

    # 3. 生成相关的冷却速率（与温度正相关）
    base_cooling_rate = data['temperature'] * 0.1 + np.random.normal(0, 5, n_samples)
    data['cooling_rate'] = np.clip(base_cooling_rate, 1, 50)  # °C/min

    # 4. 生成能量分布（满足守恒约束）
    total_energy = 100.0
    energy_ratios = np.random.dirichlet([2, 3, 1], n_samples)
    data['energy_heat'] = energy_ratios[:, 0] * total_energy
    data['energy_kinetic'] = energy_ratios[:, 1] * total_energy
    data['energy_potential'] = energy_ratios[:, 2] * total_energy

    # 5. 生成性能分数（与多个因素相关）
    # 基础分数
    base_score = 50.0

    # 材料密度的影响（正相关）
    density_effect = (data['material_density'] - 2.7) * 20

    # 纯度的影响（正相关）
    purity_effect = (data['material_purity'] - 0.95) * 200

    # 温度的影响（有最优值）
    optimal_temp = 150.0
    temp_effect = -0.01 * (data['temperature'] - optimal_temp) ** 2 + 10

    # 压力的影响（对数关系）
    pressure_effect = 5 * np.log(data['pressure'] + 1)

    # 供应商的影响
    supplier_effect = np.zeros(n_samples)
    supplier_effect[np.array(data['supplier_id']) == 'S001'] = 2
    supplier_effect[np.array(data['supplier_id']) == 'S002'] = 0
    supplier_effect[np.array(data['supplier_id']) == 'S003'] = -1

    # 能量分布的影响
    energy_balance = np.abs(data['energy_heat'] - 30) + \
                     np.abs(data['energy_kinetic'] - 60) + \
                     np.abs(data['energy_potential'] - 10)
    energy_effect = -0.1 * energy_balance

    # 组合所有影响并添加噪声
    data['performance_score'] = base_score + density_effect + purity_effect + \
                                temp_effect + pressure_effect + supplier_effect + \
                                energy_effect + np.random.normal(0, 3, n_samples)

    # 限制在合理范围内
    data['performance_score'] = np.clip(data['performance_score'], 0, 100)

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 添加一些缺失值（模拟真实场景）
    missing_prob = 0.02
    for col in df.columns:
        if col != 'performance_score':  # 目标变量不添加缺失值
            mask = np.random.random(n_samples) < missing_prob
            df.loc[mask, col] = np.nan

    return df


def main():
    parser = argparse.ArgumentParser(description="生成硬件工程测试数据")
    parser.add_argument("--output", default="data/hardware_test.csv", help="输出文件路径")
    parser.add_argument("--samples", type=int, default=200, help="样本数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    # 生成数据
    print(f"Generating {args.samples} samples...")
    df = generate_hardware_test_data(n_samples=args.samples, seed=args.seed)

    # 显示数据摘要
    print("\nData Summary:")
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"\nFeature statistics:")
    print(df.describe())

    # 验证约束
    print("\nConstraint validation:")

    # 能量守恒
    energy_sum = df[['energy_heat', 'energy_kinetic', 'energy_potential']].sum(axis=1)
    energy_error = (energy_sum - 100.0).abs()
    print(f"Energy conservation error: mean={energy_error.mean():.6f}, max={energy_error.max():.6f}")

    # 温度范围
    temp_valid = (df['temperature'] >= 20) & (df['temperature'] <= 300)
    print(f"Temperature in range: {temp_valid.sum()}/{len(df)}")

    # 材料纯度范围
    purity_valid = (df['material_purity'] >= 0.95) & (df['material_purity'] <= 1.0)
    print(f"Purity in range: {purity_valid.sum()}/{len(df)}")

    # 相关性
    corr_density_score = df['material_density'].corr(df['performance_score'])
    print(f"Correlation (density-score): {corr_density_score:.3f}")

    # 保存数据
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nData saved to {output_path}")


if __name__ == "__main__":
    main()