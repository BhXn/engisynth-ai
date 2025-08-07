"""
Constrained Generator Factory
统一创建和管理约束版本的生成器
"""

from typing import Dict, Type, Optional, Any, List
from .base import BaseGenerator
from .constrained_generator import ConstrainedGenerator
from .constrained_ctgan import ConstrainedCTGAN
from .constrained_tvae import ConstrainedTVAE
from .constrained_copulagan import ConstrainedCopulaGAN
from .constrained_gaussian_copula import ConstrainedGaussianCopula
from .constrained_tabddpm import ConstrainedTabDDPM
from ..constraints.manager import ConstraintManager


class ConstrainedGeneratorFactory:
    """约束生成器工厂类"""

    # 注册所有可用的约束生成器
    _generators: Dict[str, Type[ConstrainedGenerator]] = {
        'constrained_ctgan': ConstrainedCTGAN,
        'constrained_tvae': ConstrainedTVAE,
        'constrained_copulagan': ConstrainedCopulaGAN,
        'constrained_gaussian_copula': ConstrainedGaussianCopula,
        'constrained_tabddpm': ConstrainedTabDDPM,
    }

    # 生成器描述信息
    _descriptions = {
        'constrained_ctgan': {
            'name': 'Constrained CTGAN',
            'description': '支持约束的条件表格GAN',
            'features': ['约束投影', '拒绝采样', '噪声注入', '归一化处理'],
            'best_for': '需要严格约束满足的混合类型数据生成'
        },
        'constrained_tvae': {
            'name': 'Constrained TVAE',
            'description': '支持约束的变分自编码器',
            'features': ['潜在空间采样', '插值生成', '多样性控制', '重构误差评估'],
            'best_for': '需要可控生成和快速迭代的场景'
        },
        'constrained_copulagan': {
            'name': 'Constrained CopulaGAN',
            'description': '支持约束和依赖关系保持的生成器',
            'features': ['依赖关系保持', '边缘分布约束', '相关性强化', '互信息计算'],
            'best_for': '具有复杂相关性且需要约束的数据'
        },
        'constrained_gaussian_copula': {
            'name': 'Constrained GaussianCopula',
            'description': '支持约束的统计生成方法',
            'features': ['精确边缘分布', '相关性调整', '分布质量评估', '自适应采样'],
            'best_for': '需要快速生成且有分布约束的场景'
        },
        'constrained_tabddpm': {
            'name': 'Constrained TabDDPM',
            'description': '支持约束的扩散模型',
            'features': ['引导采样', '温度退火', '渐进细化', '条件生成'],
            'best_for': '追求最高质量且有复杂约束的生成任务'
        }
    }

    @classmethod
    def create(cls,
               generator_type: str,
               constraints_cfg: Optional[List[Dict[str, Any]]] = None,
               noise_config: Optional[Dict[str, Any]] = None,
               normalization_config: Optional[Dict[str, Any]] = None,
               **kwargs) -> ConstrainedGenerator:
        """
        创建指定类型的约束生成器

        Args:
            generator_type: 生成器类型
            constraints_cfg: 约束配置列表
            noise_config: 噪声配置
            normalization_config: 归一化配置
            **kwargs: 传递给生成器的其他参数

        Returns:
            ConstrainedGenerator: 约束生成器实例
        """
        # 添加 'constrained_' 前缀如果没有
        if not generator_type.startswith('constrained_'):
            generator_type = f'constrained_{generator_type}'

        generator_type = generator_type.lower()

        if generator_type not in cls._generators:
            available = ', '.join(cls._generators.keys())
            raise ValueError(
                f"Unknown generator type: {generator_type}. "
                f"Available types: {available}"
            )

        generator_class = cls._generators[generator_type]

        # 创建生成器实例
        return generator_class(
            constraints_cfg=constraints_cfg,
            noise_config=noise_config,
            normalization_config=normalization_config,
            **kwargs
        )

    @classmethod
    def create_with_preset(cls,
                           generator_type: str,
                           preset: str = 'default',
                           **kwargs) -> ConstrainedGenerator:
        """
        使用预设配置创建约束生成器

        Args:
            generator_type: 生成器类型
            preset: 预设名称 ('default', 'strict', 'relaxed', 'high_quality')
            **kwargs: 额外参数

        Returns:
            ConstrainedGenerator: 配置好的约束生成器
        """
        # 预设配置
        presets = {
            'default': {
                'noise_config': {
                    'default': {'type': 'gaussian', 'std': 0.01, 'relative': False}
                },
                'normalization_config': {
                    'default': {'method': 'minmax'}
                }
            },
            'strict': {
                'noise_config': {},  # 无噪声
                'normalization_config': {
                    'default': {'method': 'zscore'}
                },
                'max_rejection_samples': 50000
            },
            'relaxed': {
                'noise_config': {
                    'default': {'type': 'gaussian', 'std': 0.05, 'relative': True}
                },
                'normalization_config': {
                    'default': {'method': 'minmax'}
                },
                'max_rejection_samples': 5000
            },
            'high_quality': {
                'noise_config': {
                    'default': {'type': 'gaussian', 'std': 0.001, 'relative': False}
                },
                'normalization_config': {
                    'default': {'method': 'zscore'}
                },
                'epochs': 1000,  # 更多训练轮数
                'batch_size': 512  # 更大批次
            }
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")

        # 合并预设配置和用户参数
        config = presets[preset].copy()
        config.update(kwargs)

        return cls.create(generator_type, **config)

    @classmethod
    def recommend_constrained_generator(cls,
                                        constraint_complexity: str = 'medium',
                                        data_size: str = 'medium',
                                        quality_priority: bool = True,
                                        has_gpu: bool = False) -> str:
        """
        推荐合适的约束生成器

        Args:
            constraint_complexity: 约束复杂度 ('simple', 'medium', 'complex')
            data_size: 数据规模 ('small', 'medium', 'large')
            quality_priority: 是否优先考虑质量
            has_gpu: 是否有GPU

        Returns:
            str: 推荐的约束生成器类型
        """
        # 推荐逻辑
        if constraint_complexity == 'simple':
            if data_size == 'small':
                return 'constrained_gaussian_copula'
            else:
                return 'constrained_tvae'

        elif constraint_complexity == 'complex':
            if quality_priority and has_gpu:
                return 'constrained_tabddpm'
            else:
                return 'constrained_copulagan'

        else:  # medium
            if data_size == 'large' and not quality_priority:
                return 'constrained_tvae'
            elif quality_priority:
                return 'constrained_ctgan'
            else:
                return 'constrained_copulagan'

    @classmethod
    def create_ensemble(cls,
                        generator_types: List[str],
                        weights: Optional[List[float]] = None,
                        voting: str = 'weighted',
                        **kwargs) -> 'EnsembleConstrainedGenerator':
        """
        创建集成约束生成器

        Args:
            generator_types: 生成器类型列表
            weights: 各生成器权重
            voting: 投票方式 ('weighted', 'majority', 'best')
            **kwargs: 生成器参数

        Returns:
            EnsembleConstrainedGenerator: 集成生成器
        """
        from .ensemble_constrained import EnsembleConstrainedGenerator

        generators = []
        for gen_type in generator_types:
            gen = cls.create(gen_type, **kwargs)
            generators.append(gen)

        return EnsembleConstrainedGenerator(
            generators=generators,
            weights=weights,
            voting=voting
        )

    @classmethod
    def list_generators(cls) -> Dict[str, Dict[str, Any]]:
        """列出所有可用的约束生成器"""
        return cls._descriptions.copy()

    @classmethod
    def get_generator_info(cls, generator_type: str) -> Dict[str, Any]:
        """获取指定约束生成器的详细信息"""
        if not generator_type.startswith('constrained_'):
            generator_type = f'constrained_{generator_type}'

        generator_type = generator_type.lower()

        if generator_type not in cls._descriptions:
            raise ValueError(f"No information available for generator: {generator_type}")

        return cls._descriptions[generator_type].copy()

    @classmethod
    def validate_constraints(cls,
                             constraints_cfg: List[Dict[str, Any]],
                             sample_data: Optional[Any] = None) -> Dict[str, Any]:
        """
        验证约束配置的有效性

        Args:
            constraints_cfg: 约束配置列表
            sample_data: 样本数据（可选）

        Returns:
            Dict: 验证结果
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'constraint_count': len(constraints_cfg)
        }

        # 验证约束配置格式
        required_keys = ['type', 'columns']

        for i, constraint in enumerate(constraints_cfg):
            # 检查必需键
            for key in required_keys:
                if key not in constraint:
                    validation_results['valid'] = False
                    validation_results['errors'].append(
                        f"Constraint {i}: Missing required key '{key}'"
                    )

            # 检查约束类型
            valid_types = ['range', 'categorical', 'dependency', 'custom']
            if constraint.get('type') not in valid_types:
                validation_results['warnings'].append(
                    f"Constraint {i}: Unknown type '{constraint.get('type')}'"
                )

            # 如果提供了样本数据，检查列是否存在
            if sample_data is not None and hasattr(sample_data, 'columns'):
                for col in constraint.get('columns', []):
                    if col not in sample_data.columns:
                        validation_results['warnings'].append(
                            f"Constraint {i}: Column '{col}' not found in sample data"
                        )

        return validation_results


# ============================================================================
# 便捷函数 - 提供顶层函数接口
# ============================================================================

def get_constrained_generator(
    generator_type: str,
    constraints: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> ConstrainedGenerator:
    """
    便捷函数：创建指定类型的约束生成器

    Args:
        generator_type: 生成器类型（可以带或不带 'constrained_' 前缀）
        constraints: 约束配置列表
        **kwargs: 传递给生成器的其他参数

    Returns:
        ConstrainedGenerator: 约束生成器实例

    Examples:
        >>> from generators.constrained_factory import get_constrained_generator
        >>> constraints = [
        ...     {'type': 'range', 'columns': ['age'], 'min': 18, 'max': 100}
        ... ]
        >>> gen = get_constrained_generator('ctgan', constraints, epochs=300)
        >>> gen.fit(train_df)
        >>> synth = gen.sample(1000)
    """
    # 处理 constraints 参数名称的兼容性
    if 'constraints_cfg' not in kwargs and constraints is not None:
        kwargs['constraints_cfg'] = constraints

    return ConstrainedGeneratorFactory.create(generator_type, **kwargs)


def list_available() -> Dict[str, Dict[str, Any]]:
    """
    列出所有可用的约束生成器

    Returns:
        Dict: 约束生成器信息字典

    Examples:
        >>> from generators.constrained_factory import list_available
        >>> generators = list_available()
        >>> for name, info in generators.items():
        ...     print(f"{name}: {info['description']}")
    """
    return ConstrainedGeneratorFactory.list_generators()


def get_generator_info(generator_type: str) -> Dict[str, Any]:
    """
    获取指定约束生成器的详细信息

    Args:
        generator_type: 生成器类型

    Returns:
        Dict: 约束生成器详细信息

    Examples:
        >>> from generators.constrained_factory import get_generator_info
        >>> info = get_generator_info('constrained_ctgan')
        >>> print(info['description'])
    """
    return ConstrainedGeneratorFactory.get_generator_info(generator_type)