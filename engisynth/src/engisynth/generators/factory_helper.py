"""
Factory Helper Functions
为工厂类提供便捷的顶层函数
"""

from typing import Optional, List, Dict, Any
from .factory import GeneratorFactory
from .constrained_factory import ConstrainedGeneratorFactory
from .base import BaseGenerator
from .constrained_generator import ConstrainedGenerator


def get_generator(generator_type: str, **kwargs) -> BaseGenerator:
    """
    便捷函数：创建指定类型的生成器

    Args:
        generator_type: 生成器类型 (ctgan, tvae, copulagan, gaussian_copula, tabddpm, tabpfn)
        **kwargs: 传递给生成器构造函数的参数

    Returns:
        BaseGenerator: 生成器实例

    Examples:
        >>> gen = get_generator('ctgan', epochs=300, batch_size=64)
        >>> gen.fit(train_df)
        >>> synth = gen.sample(1000)
    """
    return GeneratorFactory.create(generator_type, **kwargs)


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


def list_available_generators(include_constrained: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    列出所有可用的生成器及其信息

    Args:
        include_constrained: 是否包含约束版本的生成器

    Returns:
        Dict: 生成器信息字典

    Examples:
        >>> generators = list_available_generators()
        >>> for name, info in generators.items():
        ...     print(f"{name}: {info['description']}")
    """
    generators = GeneratorFactory.list_generators()

    if include_constrained:
        constrained = ConstrainedGeneratorFactory.list_generators()
        generators.update(constrained)

    return generators


def recommend_generator(
        data_size: str = 'medium',
        data_type: str = 'mixed',
        priority: str = 'quality',
        needs_constraints: bool = False,
        has_gpu: bool = False
) -> str:
    """
    根据条件推荐合适的生成器

    Args:
        data_size: 数据规模 ('small', 'medium', 'large')
        data_type: 数据类型 ('numerical', 'categorical', 'mixed')
        priority: 优先级 ('speed', 'quality', 'balance')
        needs_constraints: 是否需要约束支持
        has_gpu: 是否有GPU

    Returns:
        str: 推荐的生成器类型

    Examples:
        >>> gen_type = recommend_generator(
        ...     data_size='large',
        ...     priority='quality',
        ...     has_gpu=True
        ... )
        >>> print(f"Recommended: {gen_type}")
    """
    if needs_constraints:
        # 获取约束生成器推荐
        constraint_complexity = 'complex' if priority == 'quality' else 'simple'
        quality_priority = priority == 'quality'

        return ConstrainedGeneratorFactory.recommend_constrained_generator(
            constraint_complexity=constraint_complexity,
            data_size=data_size,
            quality_priority=quality_priority,
            has_gpu=has_gpu
        )
    else:
        # 获取标准生成器推荐
        return GeneratorFactory.recommend_generator(
            data_size=data_size,
            data_type=data_type,
            priority=priority,
            has_gpu=has_gpu
        )


def create_generator_with_preset(
        generator_type: str,
        preset: str = 'default',
        use_constraints: bool = False,
        **kwargs
) -> BaseGenerator:
    """
    使用预设配置创建生成器

    Args:
        generator_type: 生成器类型
        preset: 预设名称 ('default', 'strict', 'relaxed', 'high_quality')
        use_constraints: 是否使用约束版本
        **kwargs: 额外参数

    Returns:
        BaseGenerator: 配置好的生成器

    Examples:
        >>> gen = create_generator_with_preset(
        ...     'ctgan',
        ...     preset='high_quality',
        ...     use_constraints=True
        ... )
    """
    if use_constraints:
        return ConstrainedGeneratorFactory.create_with_preset(
            generator_type,
            preset=preset,
            **kwargs
        )
    else:
        # 对于标准生成器，实现简单的预设逻辑
        presets = {
            'default': {},
            'high_quality': {
                'epochs': 1000,
                'batch_size': 512
            },
            'fast': {
                'epochs': 100,
                'batch_size': 128
            },
            'balanced': {
                'epochs': 300,
                'batch_size': 256
            }
        }

        if preset in presets:
            config = presets[preset].copy()
            config.update(kwargs)
            return GeneratorFactory.create(generator_type, **config)
        else:
            return GeneratorFactory.create(generator_type, **kwargs)


def validate_generator_config(
        generator_type: str,
        config: Dict[str, Any],
        is_constrained: bool = False
) -> Dict[str, Any]:
    """
    验证生成器配置的有效性

    Args:
        generator_type: 生成器类型
        config: 配置字典
        is_constrained: 是否为约束生成器

    Returns:
        Dict: 验证结果

    Examples:
        >>> config = {'epochs': 300, 'batch_size': 64}
        >>> result = validate_generator_config('ctgan', config)
        >>> if result['valid']:
        ...     gen = get_generator('ctgan', **config)
    """
    validation_result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }

    # 检查生成器类型是否存在
    try:
        if is_constrained:
            ConstrainedGeneratorFactory.get_generator_info(generator_type)
        else:
            GeneratorFactory.get_generator_info(generator_type)
    except ValueError as e:
        validation_result['valid'] = False
        validation_result['errors'].append(str(e))
        return validation_result

    # 基本参数验证
    if 'epochs' in config:
        if not isinstance(config['epochs'], int) or config['epochs'] <= 0:
            validation_result['errors'].append("'epochs' must be a positive integer")
            validation_result['valid'] = False

    if 'batch_size' in config:
        if not isinstance(config['batch_size'], int) or config['batch_size'] <= 0:
            validation_result['errors'].append("'batch_size' must be a positive integer")
            validation_result['valid'] = False

    # 约束配置验证
    if is_constrained and 'constraints_cfg' in config:
        constraints_result = ConstrainedGeneratorFactory.validate_constraints(
            config['constraints_cfg']
        )
        if not constraints_result['valid']:
            validation_result['valid'] = False
            validation_result['errors'].extend(constraints_result['errors'])
        validation_result['warnings'].extend(constraints_result.get('warnings', []))

    return validation_result


# 导出便捷函数
__all__ = [
    'get_generator',
    'get_constrained_generator',
    'list_available_generators',
    'recommend_generator',
    'create_generator_with_preset',
    'validate_generator_config'
]