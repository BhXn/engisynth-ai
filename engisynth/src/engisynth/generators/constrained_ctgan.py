from .constrained_generator import ConstrainedGenerator
from .ctgan import CTGANAdapter
from ..constraints.manager import ConstraintManager
from typing import Optional, Dict, Any, List
import pandas as pd


class ConstrainedCTGAN(ConstrainedGenerator):
    """集成约束处理的CTGAN生成器"""

    def __init__(
            self,
            epochs: int = 300,
            batch_size: int = 64,
            constraints_cfg: Optional[List[Dict[str, Any]]] = None,
            noise_config: Optional[Dict[str, Any]] = None,
            normalization_config: Optional[Dict[str, Any]] = None,
            **ctgan_kwargs
    ):
        # 创建基础CTGAN生成器
        base_generator = CTGANAdapter(
            epochs=epochs,
            batch_size=batch_size,
            **ctgan_kwargs
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

    def sample_with_conditions(
            self,
            n: int,
            conditions: Optional[Dict[str, Any]] = None,
            max_rejection_samples: int = 10000
    ) -> pd.DataFrame:
        """生成满足条件的样本"""
        if conditions:
            # 如果有条件，使用条件生成
            # 注意：SDV 的 CTGAN 支持条件生成，但需要在 sample 时传入 conditions
            print("Conditional generation is not yet implemented")

        return self.sample(n, max_rejection_samples)