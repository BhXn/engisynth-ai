import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from .base import BaseGenerator
from ..constraints.manager_constraint import ConstraintManager


class ConstrainedGenerator(BaseGenerator):
    """支持约束的生成器基类"""

    def __init__(
            self,
            base_generator: BaseGenerator,
            constraint_manager: Optional[ConstraintManager] = None,
            noise_config: Optional[Dict[str, Any]] = None,
            normalization_config: Optional[Dict[str, Any]] = None
    ):
        self.base_generator = base_generator
        self.constraint_manager = constraint_manager
        self.noise_config = noise_config or {}
        self.normalization_config = normalization_config or {}

        # 存储归一化参数
        self.normalization_params = {}

    def fit(self, df: pd.DataFrame, **kwargs):
        """训练生成器"""
        # 1. 数据归一化
        df_normalized = self._normalize(df, fit=True)

        # 2. 训练基础生成器
        self.base_generator.fit(df_normalized, **kwargs)

    def sample(self, n: int, max_rejection_samples: int = 10000) -> pd.DataFrame:
        """生成满足约束的样本"""
        samples_list = []
        total_generated = 0

        while len(samples_list) < n and total_generated < max_rejection_samples:
            # 1. 生成候选样本
            batch_size = min(n - len(samples_list), 1000)
            candidates = self.base_generator.sample(batch_size * 2)  # 生成更多以提高成功率

            # 2. 反归一化
            candidates = self._denormalize(candidates)

            # 3. 应用约束投影
            if self.constraint_manager:
                candidates = self.constraint_manager.project(candidates)

            # 4. 添加噪声
            candidates = self._add_noise(candidates)

            # 5. 过滤满足约束的样本
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
            return all_samples.iloc[:n]  # 返回所需数量
        else:
            print("Warning: Could not generate enough valid samples")
            return pd.DataFrame()

    def _normalize(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """归一化数据"""
        df_norm = df.copy()

        if fit:
            self.normalization_params = {}

        # 根据配置进行归一化
        for col, config in self.normalization_config.items():
            if col not in df.columns:
                continue

            method = config.get('method', 'minmax')

            if fit:
                if method == 'minmax':
                    min_val = df[col].min()
                    max_val = df[col].max()
                    self.normalization_params[col] = {
                        'method': 'minmax',
                        'min': min_val,
                        'max': max_val
                    }
                elif method == 'zscore':
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    self.normalization_params[col] = {
                        'method': 'zscore',
                        'mean': mean_val,
                        'std': std_val
                    }
                elif method == 'log':
                    self.normalization_params[col] = {
                        'method': 'log',
                        'offset': config.get('offset', 1e-6)
                    }

            # 应用归一化
            if col in self.normalization_params:
                params = self.normalization_params[col]
                if params['method'] == 'minmax':
                    min_val = params['min']
                    max_val = params['max']
                    if max_val > min_val:
                        df_norm[col] = (df[col] - min_val) / (max_val - min_val)
                elif params['method'] == 'zscore':
                    mean_val = params['mean']
                    std_val = params['std']
                    if std_val > 0:
                        df_norm[col] = (df[col] - mean_val) / std_val
                elif params['method'] == 'log':
                    offset = params['offset']
                    df_norm[col] = np.log(df[col] + offset)

        return df_norm

    def _denormalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """反归一化数据"""
        df_denorm = df.copy()

        for col, params in self.normalization_params.items():
            if col not in df.columns:
                continue

            if params['method'] == 'minmax':
                min_val = params['min']
                max_val = params['max']
                df_denorm[col] = df[col] * (max_val - min_val) + min_val
            elif params['method'] == 'zscore':
                mean_val = params['mean']
                std_val = params['std']
                df_denorm[col] = df[col] * std_val + mean_val
            elif params['method'] == 'log':
                offset = params['offset']
                df_denorm[col] = np.exp(df[col]) - offset

        return df_denorm

    def _add_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加噪声以模拟测量误差"""
        df_noisy = df.copy()

        for col, config in self.noise_config.items():
            if col not in df.columns:
                continue

            noise_type = config.get('type', 'gaussian')

            if noise_type == 'gaussian':
                # 高斯噪声
                std = config.get('std', 0.01)
                noise = np.random.normal(0, std, size=len(df))
                if config.get('relative', False):
                    # 相对噪声
                    df_noisy[col] = df[col] * (1 + noise)
                else:
                    # 绝对噪声
                    df_noisy[col] = df[col] + noise

            elif noise_type == 'uniform':
                # 均匀噪声
                scale = config.get('scale', 0.01)
                noise = np.random.uniform(-scale, scale, size=len(df))
                if config.get('relative', False):
                    df_noisy[col] = df[col] * (1 + noise)
                else:
                    df_noisy[col] = df[col] + noise

            elif noise_type == 'quantization':
                # 量化噪声
                levels = config.get('levels', 100)
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    # 量化到指定的级别
                    step = (max_val - min_val) / levels
                    df_noisy[col] = np.round((df[col] - min_val) / step) * step + min_val

        return df_noisy