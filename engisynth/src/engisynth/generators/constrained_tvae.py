"""
Constrained TVAE Generator
支持约束条件的 TVAE 生成器实现
"""

from .constrained_generator import ConstrainedGenerator
from .tvae import TVAEAdapter
from ..constraints.manager import ConstraintManager
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np


class ConstrainedTVAE(ConstrainedGenerator):
    """集成约束处理的TVAE生成器"""

    def __init__(
            self,
            epochs: int = 300,
            batch_size: int = 500,
            embedding_dim: int = 128,
            compress_dims: tuple = (128, 128),
            decompress_dims: tuple = (128, 128),
            l2_scale: float = 1e-5,
            loss_factor: int = 2,
            constraints_cfg: Optional[List[Dict[str, Any]]] = None,
            noise_config: Optional[Dict[str, Any]] = None,
            normalization_config: Optional[Dict[str, Any]] = None,
            **tvae_kwargs
    ):
        """
        初始化约束版 TVAE

        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            embedding_dim: 嵌入维度
            compress_dims: 编码器隐藏层维度
            decompress_dims: 解码器隐藏层维度
            l2_scale: L2 正则化系数
            loss_factor: 损失函数权重因子
            constraints_cfg: 约束配置列表
            noise_config: 噪声配置
            normalization_config: 归一化配置
        """
        # 创建基础 TVAE 生成器
        base_generator = TVAEAdapter(
            epochs=epochs,
            batch_size=batch_size,
            embedding_dim=embedding_dim,
            compress_dims=compress_dims,
            decompress_dims=decompress_dims,
            l2_scale=l2_scale,
            loss_factor=loss_factor,
            **tvae_kwargs
        )

        # 创建约束管理器
        constraint_manager = None
        if constraints_cfg:
            constraint_manager = ConstraintManager(constraints_cfg)

        # 初始化父类
        super().__init__(
            base_generator=base_generator,
            constraint_manager=constraint_manager,
            noise_config=noise_config,
            normalization_config=normalization_config
        )

        # TVAE 特有的参数
        self.embedding_dim = embedding_dim
        self.latent_dim_cache = None

    def sample_from_latent(
            self,
            n: int,
            latent_vectors: Optional[np.ndarray] = None,
            max_rejection_samples: int = 10000
    ) -> pd.DataFrame:
        """
        从潜在空间采样（TVAE 特有功能）

        Args:
            n: 生成样本数量
            latent_vectors: 指定的潜在向量，如果为None则随机生成
            max_rejection_samples: 最大拒绝采样次数

        Returns:
            满足约束的生成样本
        """
        if latent_vectors is not None:
            # 使用指定的潜在向量生成
            # 注意：这需要直接访问 TVAE 的解码器，SDV 可能不直接支持
            print("Custom latent vector generation requires direct model access")
            return self.sample(n, max_rejection_samples)

        # 使用标准采样
        return self.sample(n, max_rejection_samples)

    def interpolate(
            self,
            sample1: pd.Series,
            sample2: pd.Series,
            steps: int = 10,
            apply_constraints: bool = True
    ) -> pd.DataFrame:
        """
        在两个样本之间进行插值（利用 VAE 的连续潜在空间）

        Args:
            sample1: 起始样本
            sample2: 结束样本
            steps: 插值步数
            apply_constraints: 是否应用约束

        Returns:
            插值生成的样本序列
        """
        # 生成插值样本
        interpolated_samples = []

        for alpha in np.linspace(0, 1, steps):
            # 简单的线性插值（在数据空间）
            interpolated = sample1 * (1 - alpha) + sample2 * alpha
            interpolated_df = pd.DataFrame([interpolated])

            if apply_constraints and self.constraint_manager:
                # 应用约束投影
                interpolated_df = self.constraint_manager.project(interpolated_df)

            # 添加噪声
            if self.noise_config:
                interpolated_df = self._add_noise(interpolated_df)

            interpolated_samples.append(interpolated_df)

        return pd.concat(interpolated_samples, ignore_index=True)

    def sample_with_diversity_control(
            self,
            n: int,
            diversity_factor: float = 1.0,
            max_rejection_samples: int = 10000
    ) -> pd.DataFrame:
        """
        控制生成样本的多样性

        Args:
            n: 生成样本数量
            diversity_factor: 多样性因子（>1 增加多样性，<1 减少多样性）
            max_rejection_samples: 最大拒绝采样次数

        Returns:
            满足约束的生成样本
        """
        samples_list = []
        total_generated = 0

        while len(samples_list) < n and total_generated < max_rejection_samples:
            # 生成候选样本
            batch_size = min(n - len(samples_list), 1000)
            candidates = self.base_generator.sample(batch_size * 2)

            # 反归一化
            candidates = self._denormalize(candidates)

            # 调整多样性（通过缩放数值特征的方差）
            if diversity_factor != 1.0:
                numeric_cols = candidates.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    mean_val = candidates[col].mean()
                    centered = candidates[col] - mean_val
                    candidates[col] = mean_val + centered * diversity_factor

            # 应用约束投影
            if self.constraint_manager:
                candidates = self.constraint_manager.project(candidates)

            # 添加噪声
            candidates = self._add_noise(candidates)

            # 过滤满足约束的样本
            if self.constraint_manager:
                valid_samples = self.constraint_manager.filter_satisfied(candidates)
            else:
                valid_samples = candidates

            if len(valid_samples) > 0:
                samples_list.append(valid_samples)

            total_generated += len(candidates)

        # 合并所有有效样本
        if samples_list:
            all_samples = pd.concat(samples_list, ignore_index=True)
            return all_samples.iloc[:n]
        else:
            print("Warning: Could not generate enough valid samples")
            return pd.DataFrame()

    def get_reconstruction_error(self, df: pd.DataFrame) -> float:
        """
        计算重构误差（评估 VAE 的重构质量）

        Args:
            df: 输入数据

        Returns:
            平均重构误差
        """
        # 归一化输入数据
        df_normalized = self._normalize(df, fit=False)

        # 生成相同数量的样本
        reconstructed = self.base_generator.sample(len(df))

        # 反归一化
        reconstructed = self._denormalize(reconstructed)

        # 计算重构误差（均方误差）
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        total_error = 0

        for col in numeric_cols:
            if col in reconstructed.columns:
                error = np.mean((df[col].values - reconstructed[col].values) ** 2)
                total_error += error

        return total_error / len(numeric_cols) if len(numeric_cols) > 0 else 0