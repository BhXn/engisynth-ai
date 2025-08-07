"""
Constrained GaussianCopula Generator
支持约束条件的 GaussianCopula 生成器实现
"""

from .constrained_generator import ConstrainedGenerator
from .gaussian_copula import GaussianCopulaAdapter
from ..constraints.manager import ConstraintManager
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, multivariate_normal


class ConstrainedGaussianCopula(ConstrainedGenerator):
    """集成约束处理的 GaussianCopula 生成器"""

    def __init__(
            self,
            numerical_distributions: Optional[Dict[str, str]] = None,
            default_distribution: str = 'beta',
            min_value_clip: Optional[float] = None,
            max_value_clip: Optional[float] = None,
            constraints_cfg: Optional[List[Dict[str, Any]]] = None,
            noise_config: Optional[Dict[str, Any]] = None,
            normalization_config: Optional[Dict[str, Any]] = None,
            correlation_method: str = 'pearson',
            **gaussian_copula_kwargs
    ):
        """
        初始化约束版 GaussianCopula

        Args:
            numerical_distributions: 数值列的分布类型字典
            default_distribution: 默认分布类型
            min_value_clip: 数值的最小值裁剪
            max_value_clip: 数值的最大值裁剪
            constraints_cfg: 约束配置列表
            noise_config: 噪声配置
            normalization_config: 归一化配置
            correlation_method: 相关性计算方法 ('pearson', 'spearman', 'kendall')
        """
        # 创建基础 GaussianCopula 生成器
        base_generator = GaussianCopulaAdapter(
            numerical_distributions=numerical_distributions,
            default_distribution=default_distribution,
            min_value_clip=min_value_clip,
            max_value_clip=max_value_clip,
            **gaussian_copula_kwargs
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

        # GaussianCopula 特有参数
        self.correlation_method = correlation_method
        self.correlation_matrix = None
        self.marginal_distributions = {}
        self.copula_correlation = None

    def fit(self, df: pd.DataFrame, **kwargs):
        """
        训练生成器，学习边缘分布和相关性结构

        Args:
            df: 训练数据
            **kwargs: 额外参数
        """
        # 学习边缘分布
        self._fit_marginals(df)

        # 学习相关性结构
        self._fit_correlation(df)

        # 调用父类的 fit 方法
        super().fit(df, **kwargs)

    def _fit_marginals(self, df: pd.DataFrame):
        """
        拟合边缘分布

        Args:
            df: 输入数据
        """
        self.marginal_distributions = {}

        for col in df.columns:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                # 数值列：拟合分布
                data = df[col].dropna().values

                # 尝试多种分布
                distributions = {
                    'norm': stats.norm,
                    'gamma': stats.gamma,
                    'beta': stats.beta,
                    'expon': stats.expon,
                    'lognorm': stats.lognorm,
                    'uniform': stats.uniform
                }

                best_dist = None
                best_params = None
                best_ks_stat = float('inf')

                for dist_name, dist_obj in distributions.items():
                    try:
                        # 拟合分布
                        params = dist_obj.fit(data)

                        # Kolmogorov-Smirnov 检验
                        ks_stat, _ = stats.kstest(data, lambda x: dist_obj.cdf(x, *params))

                        if ks_stat < best_ks_stat:
                            best_ks_stat = ks_stat
                            best_dist = dist_name
                            best_params = params
                    except:
                        continue

                self.marginal_distributions[col] = {
                    'type': 'continuous',
                    'distribution': best_dist,
                    'params': best_params,
                    'ks_statistic': best_ks_stat
                }
            else:
                # 分类列：计算频率分布
                value_counts = df[col].value_counts(normalize=True)
                self.marginal_distributions[col] = {
                    'type': 'categorical',
                    'categories': value_counts.index.tolist(),
                    'probabilities': value_counts.values.tolist()
                }

    def _fit_correlation(self, df: pd.DataFrame):
        """
        拟合相关性结构

        Args:
            df: 输入数据
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 1:
            # 计算相关性矩阵
            if self.correlation_method == 'pearson':
                self.correlation_matrix = df[numeric_cols].corr(method='pearson')
            elif self.correlation_method == 'spearman':
                self.correlation_matrix = df[numeric_cols].corr(method='spearman')
            elif self.correlation_method == 'kendall':
                self.correlation_matrix = df[numeric_cols].corr(method='kendall')

            # 转换为 copula 相关性（使用正态分位数变换）
            n = len(df)
            transformed_data = pd.DataFrame()

            for col in numeric_cols:
                # 转换为均匀分布
                uniform_data = df[col].rank() / (n + 1)
                # 转换为标准正态分布
                transformed_data[col] = norm.ppf(uniform_data)

            # 计算 copula 相关性
            self.copula_correlation = transformed_data.corr()

    def sample_with_exact_marginals(
            self,
            n: int,
            preserve_marginals: bool = True,
            max_rejection_samples: int = 10000
    ) -> pd.DataFrame:
        """
        生成精确保持边缘分布的样本

        Args:
            n: 生成样本数量
            preserve_marginals: 是否精确保持边缘分布
            max_rejection_samples: 最大拒绝采样次数

        Returns:
            满足约束的生成样本
        """
        if not preserve_marginals:
            return self.sample(n, max_rejection_samples)

        samples_list = []
        total_generated = 0

        while len(samples_list) < n and total_generated < max_rejection_samples:
            batch_size = min(n - len(samples_list), 1000)

            # 生成 copula 样本
            if self.copula_correlation is not None:
                # 使用高斯 copula 生成相关的均匀分布
                numeric_cols = list(self.copula_correlation.columns)

                # 生成多元正态分布样本
                mean = np.zeros(len(numeric_cols))
                samples_normal = multivariate_normal.rvs(
                    mean=mean,
                    cov=self.copula_correlation.values,
                    size=batch_size
                )

                # 转换为均匀分布
                samples_uniform = norm.cdf(samples_normal)

                # 创建数据框
                candidates = pd.DataFrame()

                # 应用边缘分布的逆变换
                for i, col in enumerate(numeric_cols):
                    if col in self.marginal_distributions:
                        dist_info = self.marginal_distributions[col]
                        if dist_info['type'] == 'continuous':
                            dist_name = dist_info['distribution']
                            params = dist_info['params']
                            if dist_name and params:
                                dist_obj = getattr(stats, dist_name)
                                # 逆变换采样
                                candidates[col] = dist_obj.ppf(samples_uniform[:, i], *params)

                # 处理分类列
                for col, dist_info in self.marginal_distributions.items():
                    if dist_info['type'] == 'categorical':
                        categories = dist_info['categories']
                        probabilities = dist_info['probabilities']
                        candidates[col] = np.random.choice(
                            categories,
                            size=batch_size,
                            p=probabilities
                        )
            else:
                # 如果没有相关性结构，使用基础生成器
                candidates = self.base_generator.sample(batch_size)

            # 反归一化
            candidates = self._denormalize(candidates)

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

    def sample_with_correlation_adjustment(
            self,
            n: int,
            target_correlation: Optional[pd.DataFrame] = None,
            adjustment_strength: float = 1.0,
            max_rejection_samples: int = 10000
    ) -> pd.DataFrame:
        """
        生成具有调整相关性的样本

        Args:
            n: 生成样本数量
            target_correlation: 目标相关性矩阵
            adjustment_strength: 调整强度（0-1）
            max_rejection_samples: 最大拒绝采样次数

        Returns:
            满足约束的生成样本
        """
        # 如果提供了目标相关性，临时替换
        original_correlation = self.copula_correlation

        if target_correlation is not None:
            # 混合原始和目标相关性
            if self.copula_correlation is not None:
                self.copula_correlation = (
                        self.copula_correlation * (1 - adjustment_strength) +
                        target_correlation * adjustment_strength
                )
            else:
                self.copula_correlation = target_correlation

        # 生成样本
        samples = self.sample_with_exact_marginals(n, True, max_rejection_samples)

        # 恢复原始相关性
        self.copula_correlation = original_correlation

        return samples

    def get_distribution_quality_metrics(self, generated_df: pd.DataFrame) -> Dict[str, Any]:
        """
        评估生成数据的分布质量

        Args:
            generated_df: 生成的数据

        Returns:
            质量指标字典
        """
        metrics = {}

        for col in generated_df.columns:
            if col in self.marginal_distributions:
                dist_info = self.marginal_distributions[col]

                if dist_info['type'] == 'continuous':
                    # 连续变量：KS 检验
                    dist_name = dist_info['distribution']
                    params = dist_info['params']

                    if dist_name and params:
                        dist_obj = getattr(stats, dist_name)
                        ks_stat, p_value = stats.kstest(
                            generated_df[col].dropna(),
                            lambda x: dist_obj.cdf(x, *params)
                        )

                        metrics[f"{col}_ks_statistic"] = ks_stat
                        metrics[f"{col}_ks_pvalue"] = p_value

                        # 计算分位数差异
                        quantiles = [0.25, 0.5, 0.75]
                        for q in quantiles:
                            expected_q = dist_obj.ppf(q, *params)
                            actual_q = generated_df[col].quantile(q)
                            metrics[f"{col}_q{int(q * 100)}_diff"] = abs(expected_q - actual_q)

                elif dist_info['type'] == 'categorical':
                    # 分类变量：卡方检验
                    expected_probs = dist_info['probabilities']
                    categories = dist_info['categories']

                    observed_counts = generated_df[col].value_counts()
                    expected_counts = np.array(expected_probs) * len(generated_df)

                    # 确保类别对齐
                    observed = []
                    expected = []
                    for cat, exp_count in zip(categories, expected_counts):
                        if cat in observed_counts.index:
                            observed.append(observed_counts[cat])
                            expected.append(exp_count)

                    if len(observed) > 0:
                        chi2_stat, p_value = stats.chisquare(observed, expected)
                        metrics[f"{col}_chi2_statistic"] = chi2_stat
                        metrics[f"{col}_chi2_pvalue"] = p_value

        # 相关性保持度
        if self.correlation_matrix is not None:
            numeric_cols = [col for col in self.correlation_matrix.columns
                            if col in generated_df.columns]
            if len(numeric_cols) > 1:
                generated_corr = generated_df[numeric_cols].corr(method=self.correlation_method)
                corr_diff = abs(self.correlation_matrix.loc[numeric_cols, numeric_cols] -
                                generated_corr)

                metrics['correlation_preservation'] = 1 - corr_diff.mean().mean()
                metrics['max_correlation_deviation'] = corr_diff.max().max()

        return metrics

    def adaptive_sample(
            self,
            n: int,
            quality_threshold: float = 0.8,
            max_attempts: int = 5,
            max_rejection_samples: int = 10000
    ) -> pd.DataFrame:
        """
        自适应采样，直到达到质量阈值

        Args:
            n: 生成样本数量
            quality_threshold: 质量阈值（0-1）
            max_attempts: 最大尝试次数
            max_rejection_samples: 每次尝试的最大拒绝采样次数

        Returns:
            满足质量要求的生成样本
        """
        best_samples = None
        best_quality = 0

        for attempt in range(max_attempts):
            # 生成样本
            samples = self.sample_with_exact_marginals(n, True, max_rejection_samples)

            if len(samples) == 0:
                continue

            # 评估质量
            metrics = self.get_distribution_quality_metrics(samples)

            # 计算综合质量分数
            quality_scores = []
            for key, value in metrics.items():
                if 'pvalue' in key:
                    # p值越大越好（表示分布越相似）
                    quality_scores.append(min(value, 1.0))
                elif 'preservation' in key:
                    # 相关性保持度
                    quality_scores.append(value)
                elif 'statistic' in key or 'diff' in key or 'deviation' in key:
                    # 统计量和差异越小越好
                    quality_scores.append(1 / (1 + value))

            if quality_scores:
                avg_quality = np.mean(quality_scores)

                if avg_quality > best_quality:
                    best_quality = avg_quality
                    best_samples = samples

                if avg_quality >= quality_threshold:
                    print(f"Quality threshold reached: {avg_quality:.3f}")
                    return samples

        print(f"Best quality achieved: {best_quality:.3f}")
        return best_samples if best_samples is not None else pd.DataFrame()