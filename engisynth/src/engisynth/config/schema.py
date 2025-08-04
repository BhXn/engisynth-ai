from typing import Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, validator

class ConstraintCfg(BaseModel):
    """约束配置"""
    type: Literal["sum_equal", "range", "ratio", "monotonic", "correlation"]
    cols: List[str]
    value: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    values: Optional[List[str]] = None
    tol: Optional[float] = None
    direction: Optional[Literal["increasing", "decreasing"]] = None

class NoiseCfg(BaseModel):
    """噪声配置"""
    type: Literal["gaussian", "uniform", "quantization"] = "gaussian"
    std: Optional[float] = 0.01  # 用于高斯噪声
    scale: Optional[float] = 0.01  # 用于均匀噪声
    levels: Optional[int] = 100  # 用于量化噪声
    relative: bool = False  # 是否为相对噪声

class NormalizationCfg(BaseModel):
    """归一化配置"""
    method: Literal["minmax", "zscore", "log"] = "minmax"
    offset: Optional[float] = 1e-6  # 用于log变换

class GeneratorCfg(BaseModel):
    """生成器配置"""
    type: Literal["ctgan", "tvae", "tabddpm", "constrained_ctgan"] = "constrained_ctgan"
    epochs: int = 300
    batch_size: int = 64
    max_rejection_samples: int = 10000

class ProjectCfg(BaseModel):
    """项目配置"""
    dataset: str
    target: str
    feature_types: Dict[str, Literal["numerical", "categorical", "boolean"]]
    constraints: List[ConstraintCfg] = []
    noise: Dict[str, NoiseCfg] = {}  # 列名到噪声配置的映射
    normalization: Dict[str, NormalizationCfg] = {}  # 列名到归一化配置的映射
    generator: GeneratorCfg = Field(default_factory=lambda: GeneratorCfg())

    @validator("target")
    def target_in_features(cls, v, values):
        if "feature_types" in values and v not in values["feature_types"]:
            raise ValueError("target not found in feature_types")
        return v