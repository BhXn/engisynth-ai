"""
Generator Factory
统一的生成器工厂类，用于创建和管理不同的表格数据生成器
"""

from typing import Dict, Type, Optional, Any
from .base import BaseGenerator
from .ctgan import CTGANAdapter
from .tvae import TVAEAdapter
from .copulagan import CopulaGANAdapter
from .gaussian_copula import GaussianCopulaAdapter
from .tabddpm import TabDDPMAdapter


class GeneratorFactory:
    """生成器工厂类 - 统一创建和管理各种生成器"""

    # 注册所有可用的生成器
    _generators: Dict[str, Type[BaseGenerator]] = {
        'ctgan': CTGANAdapter,
        'tvae': TVAEAdapter,
        'copulagan': CopulaGANAdapter,
        'gaussian_copula': GaussianCopulaAdapter,
        'tabddpm': TabDDPMAdapter,
    }

    # 生成器描述信息
    _descriptions = {
        'ctgan': {
            'name': 'CTGAN',
            'description': '条件表格GAN，使用模式特定归一化处理混合数据类型',
            'pros': ['处理混合数据类型效果好', '生成质量高', '支持条件生成'],
            'cons': ['训练时间较长', '需要较大数据集'],
            'best_for': '混合类型数据，需要高质量生成'
        },
        'tvae': {
            'name': 'TVAE',
            'description': '基于变分自编码器的表格数据生成器',
            'pros': ['训练速度快', '稳定性好', '内存占用少'],
            'cons': ['生成多样性可能不如GAN'],
            'best_for': '快速原型开发，中小规模数据集'
        },
        'copulagan': {
            'name': 'CopulaGAN',
            'description': '结合Copula统计方法和GAN的混合模型',
            'pros': ['捕捉变量依赖关系好', '理论基础扎实', '适合相关性强的数据'],
            'cons': ['计算复杂度较高'],
            'best_for': '具有复杂相关性的数据'
        },
        'gaussian_copula': {
            'name': 'GaussianCopula',
            'description': '基于高斯Copula的纯统计方法',
            'pros': ['训练极快', '可解释性强', '数学基础扎实'],
            'cons': ['假设数据服从特定分布', '可能无法捕捉复杂模式'],
            'best_for': '快速基准测试，简单数据分布'
        },
        'tabddpm': {
            'name': 'TabDDPM',
            'description': '基于扩散模型的表格数据生成器',
            'pros': ['生成质量极高', '理论创新', '适合复杂分布'],
            'cons': ['训练时间长', '计算资源需求大', '需要额外依赖'],
            'best_for': '追求最高生成质量，复杂数据分布'
        }
    }

    @classmethod
    def create(cls,
               generator_type: str,
               **kwargs) -> BaseGenerator:
        """
        创建指定类型的生成器

        Args:
            generator_type: 生成器类型 (ctgan, tvae, copulagan, gaussian_copula, tabddpm)
            **kwargs: 传递给生成器构造函数的参数

        Returns:
            BaseGenerator: 生成器实例

        Raises:
            ValueError: 如果生成器类型不存在
        """
        generator_type = generator_type.lower()

        if generator_type not in cls._generators:
            available = ', '.join(cls._generators.keys())
            raise ValueError(
                f"Unknown generator type: {generator_type}. "
                f"Available types: {available}"
            )

        generator_class = cls._generators[generator_type]
        return generator_class(**kwargs)

    @classmethod
    def register(cls,
                 name: str,
                 generator_class: Type[BaseGenerator],
                 description: Optional[Dict[str, Any]] = None):
        """
        注册新的生成器类型

        Args:
            name: 生成器名称
            generator_class: 生成器类
            description: 生成器描述信息
        """
        cls._generators[name.lower()] = generator_class
        if description:
            cls._descriptions[name.lower()] = description

    @classmethod
    def list_generators(cls) -> Dict[str, Dict[str, Any]]:
        """
        列出所有可用的生成器及其描述

        Returns:
            Dict: 生成器信息字典
        """
        return cls._descriptions.copy()

    @classmethod
    def get_generator_info(cls, generator_type: str) -> Dict[str, Any]:
        """
        获取指定生成器的详细信息

        Args:
            generator_type: 生成器类型

        Returns:
            Dict: 生成器详细信息
        """
        generator_type = generator_type.lower()
        if generator_type not in cls._descriptions:
            raise ValueError(f"No information available for generator: {generator_type}")

        return cls._descriptions[generator_type].copy()

    @classmethod
    def recommend_generator(cls,
                            data_size: str = 'medium',
                            data_type: str = 'mixed',
                            priority: str = 'quality',
                            has_gpu: bool = False) -> str:
        """
        根据条件推荐合适的生成器

        Args:
            data_size: 数据规模 ('small', 'medium', 'large')
            data_type: 数据类型 ('numerical', 'categorical', 'mixed')
            priority: 优先级 ('speed', 'quality', 'balance')
            has_gpu: 是否有GPU

        Returns:
            str: 推荐的生成器类型
        """
        # 推荐逻辑
        if priority == 'speed':
            if data_size == 'small':
                return 'gaussian_copula'
            else:
                return 'tvae'

        elif priority == 'quality':
            if has_gpu and data_size != 'small':
                return 'tabddpm'
            elif data_type == 'mixed':
                return 'ctgan'
            else:
                return 'copulagan'

        else:  # balance
            if data_size == 'small':
                return 'tvae'
            elif data_type == 'mixed':
                return 'ctgan'
            else:
                return 'copulagan'


# ============================================================================
# 便捷函数 - 提供顶层函数接口
# ============================================================================

def get_generator(generator_type: str, **kwargs) -> BaseGenerator:
    """
    便捷函数：创建指定类型的生成器

    Args:
        generator_type: 生成器类型 (ctgan, tvae, copulagan, gaussian_copula, tabddpm)
        **kwargs: 传递给生成器构造函数的参数

    Returns:
        BaseGenerator: 生成器实例

    Examples:
        >>> from generators.factory import get_generator
        >>> gen = get_generator('ctgan', epochs=300, batch_size=64)
        >>> gen.fit(train_df)
        >>> synth = gen.sample(1000)
    """
    return GeneratorFactory.create(generator_type, **kwargs)


def list_available() -> Dict[str, Dict[str, Any]]:
    """
    列出所有可用的生成器

    Returns:
        Dict: 生成器信息字典

    Examples:
        >>> from generators.factory import list_available
        >>> generators = list_available()
        >>> for name, info in generators.items():
        ...     print(f"{name}: {info['description']}")
    """
    return GeneratorFactory.list_generators()


def get_generator_info(generator_type: str) -> Dict[str, Any]:
    """
    获取指定生成器的详细信息

    Args:
        generator_type: 生成器类型

    Returns:
        Dict: 生成器详细信息

    Examples:
        >>> from generators.factory import get_generator_info
        >>> info = get_generator_info('ctgan')
        >>> print(info['description'])
    """
    return GeneratorFactory.get_generator_info(generator_type)