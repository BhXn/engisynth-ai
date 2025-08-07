"""
Ensemble Constrained Generator
集成约束生成器，组合多个约束生成器以提高生成质量和鲁棒性
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Union
from .constrained_generator import ConstrainedGenerator


class EnsembleConstrainedGenerator(ConstrainedGenerator):
    """
    集成约束生成器

    通过组合多个约束生成器，提供更鲁棒和高质量的数据生成能力。
    支持加权投票、多数投票等多种集成策略。
    """

    def __init__(self,
                 generators: List[ConstrainedGenerator],
                 weights: Optional[List[float]] = None,
                 voting: str = 'weighted',
                 min_consensus: float = 0.5,
                 diversity_threshold: float = 0.1,
                 quality_metric: str = 'constraint_satisfaction',
                 **kwargs):
        """
        初始化集成约束生成器

        Args:
            generators: 约束生成器列表
            weights: 各生成器权重，为None时使用均匀权重
            voting: 投票策略 ('weighted', 'majority', 'best', 'adaptive')
            min_consensus: 最小共识阈值
            diversity_threshold: 多样性阈值
            quality_metric: 质量评估指标
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)

        if not generators:
            raise ValueError("至少需要一个生成器")

        self.generators = generators
        self.n_generators = len(generators)

        # 设置权重
        if weights is None:
            self.weights = np.ones(self.n_generators) / self.n_generators
        else:
            if len(weights) != self.n_generators:
                raise ValueError("权重数量必须与生成器数量一致")
            self.weights = np.array(weights)
            self.weights = self.weights / np.sum(self.weights)  # 归一化

        self.voting = voting
        self.min_consensus = min_consensus
        self.diversity_threshold = diversity_threshold
        self.quality_metric = quality_metric

        # 性能追踪
        self.generator_performance = {}
        self.generation_history = []

    def fit(self, df: pd.DataFrame, **kwargs):
        """
        训练所有集成生成器

        Args:
            df: 训练数据
            **kwargs: 额外参数
        """
        # 为每个生成器拟合数据
        for i, generator in enumerate(self.generators):
            try:
                generator.fit(df, **kwargs)
                self.generator_performance[i] = {
                    'fitted': True,
                    'error': None,
                    'quality_score': 1.0
                }
            except Exception as e:
                self.generator_performance[i] = {
                    'fitted': False,
                    'error': str(e),
                    'quality_score': 0.0
                }
                print(f"警告：生成器 {i} 训练失败: {e}")

        # 检查是否有至少一个生成器训练成功
        successful_generators = sum(1 for perf in self.generator_performance.values()
                                   if perf['fitted'])

        if successful_generators == 0:
            raise RuntimeError("所有生成器训练失败")

        self._fitted = True

    def sample(self, n: int,
               max_rejection_samples: int = 10000,
               diversity_bonus: bool = True,
               **kwargs) -> pd.DataFrame:
        """
        使用集成策略生成数据

        Args:
            n: 生成样本数
            max_rejection_samples: 最大拒绝采样次数
            diversity_bonus: 是否使用多样性奖励
            **kwargs: 额外参数

        Returns:
            生成的数据DataFrame
        """
        if not self._fitted:
            raise ValueError("必须先调用 fit() 方法")

        # 根据投票策略选择生成方法
        if self.voting == 'weighted':
            return self._weighted_sample(n, max_rejection_samples, **kwargs)
        elif self.voting == 'majority':
            return self._majority_sample(n, max_rejection_samples, **kwargs)
        elif self.voting == 'best':
            return self._best_sample(n, max_rejection_samples, **kwargs)
        elif self.voting == 'adaptive':
            return self._adaptive_sample(n, max_rejection_samples, **kwargs)
        else:
            raise ValueError(f"未知的投票策略: {self.voting}")

    def _weighted_sample(self, n: int, max_rejection_samples: int, **kwargs) -> pd.DataFrame:
        """加权采样策略"""
        successful_generators = [
            (i, gen) for i, gen in enumerate(self.generators)
            if self.generator_performance[i]['fitted']
        ]

        if not successful_generators:
            raise RuntimeError("没有可用的生成器")

        # 根据权重分配样本数
        samples_per_generator = []
        remaining_n = n

        for i, (idx, gen) in enumerate(successful_generators[:-1]):
            gen_samples = int(n * self.weights[idx])
            samples_per_generator.append((idx, gen, gen_samples))
            remaining_n -= gen_samples

        # 最后一个生成器获得剩余样本
        if successful_generators:
            last_idx, last_gen = successful_generators[-1]
            samples_per_generator.append((last_idx, last_gen, remaining_n))

        # 生成样本并合并
        all_samples = []
        for idx, gen, num_samples in samples_per_generator:
            if num_samples > 0:
                try:
                    samples = gen.sample(num_samples, max_rejection_samples, **kwargs)
                    all_samples.append(samples)
                except Exception as e:
                    print(f"警告：生成器 {idx} 采样失败: {e}")

        if not all_samples:
            raise RuntimeError("所有生成器采样失败")

        # 合并并打乱
        result = pd.concat(all_samples, ignore_index=True)
        return result.sample(frac=1, random_state=42).reset_index(drop=True)

    def _majority_sample(self, n: int, max_rejection_samples: int, **kwargs) -> pd.DataFrame:
        """多数投票采样策略"""
        successful_generators = [
            gen for i, gen in enumerate(self.generators)
            if self.generator_performance[i]['fitted']
        ]

        # 每个生成器生成相同数量的样本
        generator_samples = []
        for gen in successful_generators:
            try:
                samples = gen.sample(n, max_rejection_samples, **kwargs)
                generator_samples.append(samples)
            except Exception as e:
                print(f"警告：生成器采样失败: {e}")

        if not generator_samples:
            raise RuntimeError("所有生成器采样失败")

        # 如果只有一个成功的生成器，直接返回其结果
        if len(generator_samples) == 1:
            return generator_samples[0]

        # 通过样本相似性进行多数投票
        return self._consensus_selection(generator_samples)

    def _best_sample(self, n: int, max_rejection_samples: int, **kwargs) -> pd.DataFrame:
        """选择最优生成器策略"""
        # 根据性能选择最好的生成器
        best_idx = max(
            (i for i in range(self.n_generators)
             if self.generator_performance[i]['fitted']),
            key=lambda i: self.generator_performance[i]['quality_score'],
            default=None
        )

        if best_idx is None:
            raise RuntimeError("没有可用的生成器")

        return self.generators[best_idx].sample(n, max_rejection_samples, **kwargs)

    def _adaptive_sample(self, n: int, max_rejection_samples: int, **kwargs) -> pd.DataFrame:
        """自适应采样策略"""
        # 动态调整权重和策略
        self._update_performance_scores()

        # 如果有显著最优生成器，使用best策略
        scores = [self.generator_performance[i]['quality_score']
                 for i in range(self.n_generators)
                 if self.generator_performance[i]['fitted']]

        if scores and (max(scores) - np.mean(scores)) > self.diversity_threshold:
            return self._best_sample(n, max_rejection_samples, **kwargs)
        else:
            return self._weighted_sample(n, max_rejection_samples, **kwargs)

    def _consensus_selection(self, samples_list: List[pd.DataFrame]) -> pd.DataFrame:
        """基于共识选择最佳样本组合"""
        if len(samples_list) == 1:
            return samples_list[0]

        # 简化的共识算法：选择第一个样本作为基准
        base_samples = samples_list[0]

        # 可以在这里实现更复杂的共识算法
        # 例如：基于特征分布的相似性评分

        return base_samples

    def _update_performance_scores(self):
        """更新生成器性能分数"""
        # 基于历史生成质量更新性能分数
        # 这里可以实现更复杂的性能评估逻辑
        for i in range(self.n_generators):
            if self.generator_performance[i]['fitted']:
                # 简单的性能衰减，可以根据实际情况调整
                current_score = self.generator_performance[i]['quality_score']
                self.generator_performance[i]['quality_score'] = max(0.1, current_score * 0.95)

    def get_generator_status(self) -> Dict[str, Any]:
        """获取各生成器状态"""
        return {
            'total_generators': self.n_generators,
            'fitted_generators': sum(1 for perf in self.generator_performance.values()
                                   if perf['fitted']),
            'performance': self.generator_performance.copy(),
            'weights': self.weights.tolist(),
            'voting_strategy': self.voting
        }

    def set_weights(self, new_weights: List[float]):
        """动态调整生成器权重"""
        if len(new_weights) != self.n_generators:
            raise ValueError("权重数量必须与生成器数量一致")

        self.weights = np.array(new_weights)
        self.weights = self.weights / np.sum(self.weights)

    def remove_generator(self, index: int):
        """移除指定索引的生成器"""
        if not 0 <= index < self.n_generators:
            raise IndexError("生成器索引超出范围")

        self.generators.pop(index)
        self.weights = np.delete(self.weights, index)
        self.weights = self.weights / np.sum(self.weights)  # 重新归一化

        # 更新性能记录
        new_performance = {}
        for i, perf in self.generator_performance.items():
            if i < index:
                new_performance[i] = perf
            elif i > index:
                new_performance[i-1] = perf

        self.generator_performance = new_performance
        self.n_generators -= 1

    def add_generator(self, generator: ConstrainedGenerator, weight: float = None):
        """添加新的生成器"""
        self.generators.append(generator)

        if weight is None:
            # 均匀分配权重
            new_weight = 1.0 / (self.n_generators + 1)
            self.weights = self.weights * (self.n_generators / (self.n_generators + 1))
            self.weights = np.append(self.weights, new_weight)
        else:
            # 归一化所有权重
            total_weight = np.sum(self.weights) + weight
            self.weights = np.append(self.weights, weight) / total_weight

        self.generator_performance[self.n_generators] = {
            'fitted': False,
            'error': None,
            'quality_score': 1.0
        }

        self.n_generators += 1

    def evaluate_diversity(self, samples: pd.DataFrame) -> float:
        """评估生成样本的多样性"""
        # 简单的多样性评估，可以扩展为更复杂的指标
        if samples.empty:
            return 0.0

        # 基于数值列的标准差计算多样性
        numeric_cols = samples.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return 0.5  # 默认值

        diversity_scores = []
        for col in numeric_cols:
            col_std = samples[col].std()
            col_range = samples[col].max() - samples[col].min()
            if col_range > 0:
                diversity_scores.append(col_std / col_range)

        return np.mean(diversity_scores) if diversity_scores else 0.0