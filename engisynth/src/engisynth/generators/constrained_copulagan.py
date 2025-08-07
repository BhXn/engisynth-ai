"""
Constrained CopulaGAN Generator
支持约束条件的 CopulaGAN 生成器实现
"""

from .constrained_generator import ConstrainedGenerator
from .copulagan import CopulaGANAdapter
from ..constraints.manager import ConstraintManager
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats


class ConstrainedCopulaGAN(ConstrainedGenerator):
    """集成约束处理的 CopulaGAN 生成器"""

    def __init__(
            self,
            epochs: int = 300,
            batch_size: int = 500,
            generator_dim: tuple = (256, 256),
            discriminator_dim: tuple = (256, 256),
            generator_lr: float = 2e-4,
            discriminator_lr: float = 2e-4,
            discriminator_steps: int = 1,
            log_frequency: bool = True,
            constraints_cfg: Optional[List[Dict[str, Any]]] = None,
            noise_config: Optional[Dict[str, Any]] = None,
            normalization_config: Optional[Dict[str, Any]] = None,
            dependency_config: Optional[Dict[str, Any]] = None,
            **copulagan_kwargs
    ):
        """
        初始化约束版 CopulaGAN

        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            generator_dim: 生成器隐藏层维度
            discriminator_dim: 判别器隐藏层维度
            generator_lr: 生成器学习率
            discriminator_lr: 判别器学习率
            discriminator_steps: 判别器训练步数
            log_frequency: 是否使用对数频率编码
            constraints_cfg: 约束配置列表
            noise_config: 噪声配置
            normalization_config: 归一化配置
            dependency_config: 依赖关系配置
        """
        # 创建基础 CopulaGAN 生成器
        base_generator = CopulaGANAdapter(
            epochs=epochs,
            batch_size=batch_size,
            generator_dim=generator_dim,
            discriminator_dim=discriminator_dim,
            generator_lr=generator_lr,
            discriminator_lr=discriminator_lr,
            discriminator_steps=discriminator_steps,
            log_frequency=log_frequency,
            **copulagan_kwargs
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

        # CopulaGAN 特有的依赖关系配置
        self.dependency_config = dependency_config or {}
        self.correlation_matrix = None
        self.copula_params = None

    def fit(self, df: pd.DataFrame, **kwargs):
        """
        训练生成器，同时学习变量间的依赖关系

        Args:
            df: 训练数据
            **kwargs: 额外参数
        """
        # 学习相关性结构
        self._learn_dependencies(df)

        # 调用父类的 fit 方法
        super().fit(df, **kwargs)

    def _learn_dependencies(self, df: pd.DataFrame):
        """
        学习数据中的依赖关系和相关性结构

        Args:
            df: 输入数据
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 1:
            # 计算相关性矩阵
            self.correlation_matrix = df[numeric_cols].corr()

            # 学习 Copula 参数（简化版）
            self.copula_params = {}
            for col in numeric_cols:
                # 估计边缘分布参数
                data = df[col].dropna()

                # 尝试拟合不同的分布
                distributions = ['norm', 'gamma', 'beta', 'expon']
                best_dist = None
                best_params = None
                best_ks_stat = float('inf')

                for dist_name in distributions:
                    try:
                        dist = getattr(stats, dist_name)
                        params = dist.fit(data)
                        ks_stat, _ = stats.kstest(data, dist_name, args=params)

                        if ks_stat < best_ks_stat:
                            best_ks_stat = ks_stat
                            best_dist = dist_name
                            best_params = params
                    except:
                        continue

                self.copula_params[col] = {
                    'distribution': best_dist,
                    'params': best_params
                }

    def sample_with_dependency_preservation(
            self,
            n: int,
            correlation_strength: float = 1.0,
            max_rejection_samples: int = 10000
    ) -> pd.DataFrame:
        """
        生成保持依赖关系的样本

        Args:
            n: 生成样本数量
            correlation_strength: 相关性强度（0-1，1表示完全保持原始相关性）
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

            # 强化依赖关系
            if self.correlation_matrix is not None and correlation_strength > 0:
                candidates = self._enforce_correlations(candidates, correlation_strength)

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

    def _enforce_correlations(
            self,
            df: pd.DataFrame,
            strength: float = 1.0
    ) -> pd.DataFrame:
        """
        强化变量间的相关性

        Args:
            df: 输入数据
            strength: 相关性强度

        Returns:
            调整后的数据
        """
        df_adjusted = df.copy()
        numeric_cols = [col for col in self.correlation_matrix.columns if col in df.columns]

        if len(numeric_cols) > 1:
            # 获取当前相关性
            current_corr = df[numeric_cols].corr()

            # 计算目标相关性（混合原始和当前）
            target_corr = (self.correlation_matrix.loc[numeric_cols, numeric_cols] * strength +
                           current_corr * (1 - strength))

            # 使用 Cholesky 分解调整相关性
            try:
                # 标准化数据
                data_matrix = df[numeric_cols].values
                mean = np.mean(data_matrix, axis=0)
                std = np.std(data_matrix, axis=0)
                standardized = (data_matrix - mean) / (std + 1e-8)

                # Cholesky 分解
                L = np.linalg.cholesky(target_corr)

                # 调整数据
                adjusted = standardized @ L.T

                # 反标准化
                df_adjusted[numeric_cols] = adjusted * std + mean
            except np.linalg.LinAlgError:
                # 如果 Cholesky 分解失败，使用简单的线性调整
                for i, col1 in enumerate(numeric_cols):
                    for j, col2 in enumerate(numeric_cols):
                        if i < j:
                            target_corr_value = target_corr.iloc[i, j]
                            current_corr_value = current_corr.iloc[i, j]

                            if abs(target_corr_value - current_corr_value) > 0.1:
                                # 简单的线性调整
                                adjustment = (target_corr_value - current_corr_value) * strength
                                df_adjusted[col2] = df_adjusted[col2] + adjustment * df_adjusted[col1]

        return df_adjusted

    def sample_with_marginal_constraints(
            self,
            n: int,
            marginal_constraints: Optional[Dict[str, Tuple[float, float]]] = None,
            max_rejection_samples: int = 10000
    ) -> pd.DataFrame:
        """
        生成满足边缘分布约束的样本

        Args:
            n: 生成样本数量
            marginal_constraints: 边缘分布约束，格式为 {column: (min_percentile, max_percentile)}
            max_rejection_samples: 最大拒绝采样次数

        Returns:
            满足约束的生成样本
        """
        samples = self.sample(n, max_rejection_samples)

        if marginal_constraints:
            for col, (min_pct, max_pct) in marginal_constraints.items():
                if col in samples.columns and col in self.copula_params:
                    # 获取分布参数
                    dist_info = self.copula_params[col]
                    if dist_info['distribution'] and dist_info['params']:
                        dist = getattr(stats, dist_info['distribution'])

                        # 计算分位数对应的值
                        min_val = dist.ppf(min_pct, *dist_info['params'])
                        max_val = dist.ppf(max_pct, *dist_info['params'])

                        # 裁剪到指定范围
                        samples[col] = samples[col].clip(min_val, max_val)

        return samples

    def get_dependency_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算生成数据的依赖关系指标

        Args:
            df: 生成的数据

        Returns:
            依赖关系指标字典
        """
        metrics = {}

        # 计算相关性差异
        if self.correlation_matrix is not None:
            numeric_cols = [col for col in self.correlation_matrix.columns if col in df.columns]
            if len(numeric_cols) > 1:
                generated_corr = df[numeric_cols].corr()
                corr_diff = abs(self.correlation_matrix.loc[numeric_cols, numeric_cols] - generated_corr)

                metrics['mean_correlation_difference'] = corr_diff.mean().mean()
                metrics['max_correlation_difference'] = corr_diff.max().max()

                # 计算秩相关系数（Spearman）
                spearman_original = pd.DataFrame()
                spearman_generated = pd.DataFrame()

                for col in numeric_cols:
                    spearman_generated[col] = df[col].rank()

                metrics['spearman_correlation'] = spearman_generated.corr()

        # 计算互信息（如果可能）
        try:
            from sklearn.feature_selection import mutual_info_regression

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                mi_scores = {}
                for col in numeric_cols:
                    X = df[numeric_cols].drop(columns=[col])
                    y = df[col]
                    mi = mutual_info_regression(X, y, random_state=42)
                    mi_scores[col] = mi.mean()

                metrics['mutual_information'] = mi_scores
        except ImportError:
            pass

        return metrics