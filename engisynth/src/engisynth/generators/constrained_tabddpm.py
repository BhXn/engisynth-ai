"""
Constrained TabDDPM Generator
支持约束条件的 TabDDPM 扩散模型生成器实现
"""

from .constrained_generator import ConstrainedGenerator
from .tabddpm import TabDDPMAdapter
from ..constraints.manager import ConstraintManager
from typing import Optional, Dict, Any, List, Callable
import pandas as pd
import numpy as np
import torch


class ConstrainedTabDDPM(ConstrainedGenerator):
    """集成约束处理的 TabDDPM 生成器"""

    def __init__(
            self,
            num_timesteps: int = 1000,
            gaussian_loss_type: str = 'mse',
            scheduler: str = 'cosine',
            model_type: str = 'mlp',
            model_params: Optional[Dict[str, Any]] = None,
            num_epochs: int = 1000,
            batch_size: int = 256,
            learning_rate: float = 1e-3,
            device: str = 'auto',
            seed: Optional[int] = None,
            constraints_cfg: Optional[List[Dict[str, Any]]] = None,
            noise_config: Optional[Dict[str, Any]] = None,
            normalization_config: Optional[Dict[str, Any]] = None,
            guidance_config: Optional[Dict[str, Any]] = None,
            **tabddpm_kwargs
    ):
        """
        初始化约束版 TabDDPM

        Args:
            num_timesteps: 扩散步数
            gaussian_loss_type: 损失类型 ('mse' or 'kl')
            scheduler: 噪声调度器类型 ('linear' or 'cosine')
            model_type: 模型架构 ('mlp' or 'resnet')
            model_params: 模型参数字典
            num_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            device: 计算设备
            seed: 随机种子
            constraints_cfg: 约束配置列表
            noise_config: 噪声配置
            normalization_config: 归一化配置
            guidance_config: 引导采样配置
        """
        # 创建基础 TabDDPM 生成器
        base_generator = TabDDPMAdapter(
            num_timesteps=num_timesteps,
            gaussian_loss_type=gaussian_loss_type,
            scheduler=scheduler,
            model_type=model_type,
            model_params=model_params,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            seed=seed,
            **tabddpm_kwargs
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

        # TabDDPM 特有的引导配置
        self.guidance_config = guidance_config or {}
        self.num_timesteps = num_timesteps
        self.device = base_generator.device
        self.guidance_scale = self.guidance_config.get('scale', 1.0)

    def sample_with_guidance(
            self,
            n: int,
            guidance_fn: Optional[Callable] = None,
            guidance_scale: float = 1.0,
            max_rejection_samples: int = 10000
    ) -> pd.DataFrame:
        """
        使用引导函数的采样（Guided Diffusion）

        Args:
            n: 生成样本数量
            guidance_fn: 引导函数，接收样本并返回梯度
            guidance_scale: 引导强度
            max_rejection_samples: 最大拒绝采样次数

        Returns:
            满足约束的生成样本
        """
        if guidance_fn is None and self.constraint_manager is not None:
            # 使用约束作为引导
            guidance_fn = self._constraint_guidance

        samples_list = []
        total_generated = 0

        while len(samples_list) < n and total_generated < max_rejection_samples:
            batch_size = min(n - len(samples_list), 100)

            # 如果有引导函数，使用引导扩散
            if guidance_fn is not None:
                candidates = self._guided_sampling(batch_size, guidance_fn, guidance_scale)
            else:
                # 标准采样
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

    def _guided_sampling(
            self,
            n: int,
            guidance_fn: Callable,
            guidance_scale: float
    ) -> pd.DataFrame:
        """
        执行引导采样

        Args:
            n: 样本数量
            guidance_fn: 引导函数
            guidance_scale: 引导强度

        Returns:
            生成的样本
        """
        # 注意：这是一个简化的实现
        # 完整的引导扩散需要修改扩散过程的每一步

        # 生成初始样本
        samples = self.base_generator.sample(n)

        # 应用引导（简化版：在最终样本上应用梯度）
        if guidance_scale > 0:
            # 计算引导梯度
            samples_tensor = torch.FloatTensor(samples.values).to(self.device)
            samples_tensor.requires_grad = True

            # 计算引导分数
            guidance_score = guidance_fn(samples_tensor)

            # 计算梯度
            if guidance_score.requires_grad:
                gradients = torch.autograd.grad(
                    outputs=guidance_score.sum(),
                    inputs=samples_tensor,
                    create_graph=False
                )[0]

                # 应用梯度
                with torch.no_grad():
                    guided_samples = samples_tensor + guidance_scale * gradients
                    samples = pd.DataFrame(
                        guided_samples.cpu().numpy(),
                        columns=samples.columns
                    )

        return samples

    def _constraint_guidance(self, x: torch.Tensor) -> torch.Tensor:
        """
        基于约束的引导函数

        Args:
            x: 输入张量

        Returns:
            约束满足度分数
        """
        # 将张量转换为 DataFrame
        x_df = pd.DataFrame(x.detach().cpu().numpy())

        # 计算约束满足度
        if self.constraint_manager:
            satisfaction_scores = []
            for _, row in x_df.iterrows():
                score = self.constraint_manager.compute_satisfaction_score(
                    pd.DataFrame([row])
                )
                satisfaction_scores.append(score)

            # 返回分数张量
            return torch.FloatTensor(satisfaction_scores).to(self.device)

        return torch.zeros(len(x)).to(self.device)

    def sample_with_annealing(
            self,
            n: int,
            temperature_schedule: Optional[List[float]] = None,
            max_rejection_samples: int = 10000
    ) -> pd.DataFrame:
        """
        使用温度退火的采样

        Args:
            n: 生成样本数量
            temperature_schedule: 温度调度列表
            max_rejection_samples: 最大拒绝采样次数

        Returns:
            满足约束的生成样本
        """
        if temperature_schedule is None:
            # 默认温度调度：从高到低
            temperature_schedule = np.linspace(2.0, 0.5, 5)

        all_samples = []
        samples_per_temp = n // len(temperature_schedule)

        for temp in temperature_schedule:
            # 调整噪声水平
            temp_noise_config = self.noise_config.copy() if self.noise_config else {}
            for col in temp_noise_config:
                if 'std' in temp_noise_config[col]:
                    temp_noise_config[col]['std'] *= temp
                elif 'scale' in temp_noise_config[col]:
                    temp_noise_config[col]['scale'] *= temp

            # 临时更新噪声配置
            original_noise = self.noise_config
            self.noise_config = temp_noise_config

            # 生成样本
            samples = self.sample(samples_per_temp, max_rejection_samples)
            all_samples.append(samples)

            # 恢复原始噪声配置
            self.noise_config = original_noise

        # 合并所有样本
        if all_samples:
            combined = pd.concat(all_samples, ignore_index=True)
            return combined.iloc[:n]

        return pd.DataFrame()

    def progressive_sampling(
            self,
            n: int,
            stages: int = 3,
            refinement_steps: int = 100,
            max_rejection_samples: int = 10000
    ) -> pd.DataFrame:
        """
        渐进式采样，逐步细化生成的样本

        Args:
            n: 生成样本数量
            stages: 细化阶段数
            refinement_steps: 每阶段的细化步数
            max_rejection_samples: 最大拒绝采样次数

        Returns:
            满足约束的生成样本
        """
        # 初始采样
        samples = self.base_generator.sample(n * 2)

        for stage in range(stages):
            # 反归一化
            samples = self._denormalize(samples)

            # 计算约束满足度
            if self.constraint_manager:
                satisfaction_scores = []
                for _, row in samples.iterrows():
                    score = self.constraint_manager.compute_satisfaction_score(
                        pd.DataFrame([row])
                    )
                    satisfaction_scores.append(score)

                # 选择最好的样本
                satisfaction_scores = np.array(satisfaction_scores)
                top_indices = np.argsort(satisfaction_scores)[-n:]
                samples = samples.iloc[top_indices]

                # 应用约束投影
                samples = self.constraint_manager.project(samples)

            # 添加递减的噪声进行细化
            noise_scale = 1.0 / (stage + 1)
            if self.noise_config:
                temp_noise_config = {}
                for col, config in self.noise_config.items():
                    temp_config = config.copy()
                    if 'std' in temp_config:
                        temp_config['std'] *= noise_scale
                    elif 'scale' in temp_config:
                        temp_config['scale'] *= noise_scale
                    temp_noise_config[col] = temp_config

                # 应用噪声
                original_noise = self.noise_config
                self.noise_config = temp_noise_config
                samples = self._add_noise(samples)
                self.noise_config = original_noise

            # 如果不是最后一个阶段，生成更多候选样本
            if stage < stages - 1:
                # 基于当前样本生成变体
                variants = []
                for _ in range(2):
                    variant = samples.copy()
                    # 添加小扰动
                    numeric_cols = variant.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        variant[col] += np.random.normal(0, 0.01, len(variant))
                    variants.append(variant)

                samples = pd.concat([samples] + variants, ignore_index=True)

        # 最终过滤
        if self.constraint_manager:
            samples = self.constraint_manager.filter_satisfied(samples)

        return samples.iloc[:n] if len(samples) >= n else samples

    def sample_with_conditioning(
            self,
            n: int,
            conditions: Dict[str, Any],
            conditioning_strength: float = 1.0,
            max_rejection_samples: int = 10000
    ) -> pd.DataFrame:
        """
        条件生成（类似于 Classifier-Free Guidance）

        Args:
            n: 生成样本数量
            conditions: 条件字典，格式为 {column: value}
            conditioning_strength: 条件强度
            max_rejection_samples: 最大拒绝采样次数

        Returns:
            满足条件和约束的生成样本
        """
        samples_list = []
        total_generated = 0

        while len(samples_list) < n and total_generated < max_rejection_samples:
            batch_size = min(n - len(samples_list), 100)

            # 生成无条件样本
            unconditional_samples = self.base_generator.sample(batch_size)

            # 生成条件样本（通过后处理实现）
            conditional_samples = unconditional_samples.copy()
            for col, value in conditions.items():
                if col in conditional_samples.columns:
                    # 混合条件值和生成值
                    if isinstance(value, (int, float)):
                        # 数值条件
                        original_values = conditional_samples[col].values
                        conditional_samples[col] = (
                                value * conditioning_strength +
                                original_values * (1 - conditioning_strength)
                        )
                    else:
                        # 分类条件
                        mask = np.random.random(len(conditional_samples)) < conditioning_strength
                        conditional_samples.loc[mask, col] = value

            # 反归一化
            candidates = self._denormalize(conditional_samples)

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
            print("Warning: Could not generate enough valid samples with conditions")
            return pd.DataFrame()

    def get_diffusion_metrics(self, generated_df: pd.DataFrame, original_df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算扩散模型特有的质量指标

        Args:
            generated_df: 生成的数据
            original_df: 原始训练数据

        Returns:
            质量指标字典
        """
        metrics = {}

        # 计算 FID-like 分数（简化版）
        numeric_cols = generated_df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            # 计算特征统计
            gen_mean = generated_df[numeric_cols].mean()
            gen_cov = generated_df[numeric_cols].cov()

            orig_mean = original_df[numeric_cols].mean()
            orig_cov = original_df[numeric_cols].cov()

            # 计算 Frechet 距离（简化版）
            mean_diff = np.sum((gen_mean - orig_mean) ** 2)

            # 计算协方差差异（使用 Frobenius 范数）
            cov_diff = np.linalg.norm(gen_cov - orig_cov, 'fro')

            metrics['mean_difference'] = float(mean_diff)
            metrics['covariance_difference'] = float(cov_diff)

            # 简化的 FID 分数
            metrics['simplified_fid'] = float(mean_diff + cov_diff)

        # 计算覆盖率和密度
        from sklearn.neighbors import NearestNeighbors

        if len(numeric_cols) > 0:
            # 准备数据
            gen_data = generated_df[numeric_cols].values
            orig_data = original_df[numeric_cols].values

            # 计算覆盖率（生成样本覆盖了多少训练样本的邻域）
            nbrs = NearestNeighbors(n_neighbors=5).fit(orig_data)
            distances, _ = nbrs.kneighbors(gen_data)
            threshold = np.percentile(distances, 95)

            coverage_distances, _ = nbrs.kneighbors(orig_data)
            coverage = np.mean(np.min(coverage_distances, axis=0) < threshold)
            metrics['coverage'] = float(coverage)

            # 计算密度（生成样本的聚集程度）
            gen_nbrs = NearestNeighbors(n_neighbors=5).fit(gen_data)
            gen_distances, _ = gen_nbrs.kneighbors(gen_data)
            density = 1.0 / (1.0 + np.mean(gen_distances))
            metrics['density'] = float(density)

        # 约束满足率
        if self.constraint_manager:
            satisfaction_rate = self.constraint_manager.compute_satisfaction_rate(generated_df)
            metrics['constraint_satisfaction_rate'] = satisfaction_rate

        return metrics