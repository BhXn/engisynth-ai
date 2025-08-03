from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, validator

class ConstraintCfg(BaseModel):
    type: Literal["sum_equal", "range", "category_allowed"]
    cols: List[str]
    value: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    values: Optional[List[str]] = None

class GeneratorCfg(BaseModel):
    type: Literal["ctgan", "tvae", "tabddpm"]
    epochs: int = 300
    batch_size: int = 64

class ProjectCfg(BaseModel):
    dataset: str
    target: str
    feature_types: Dict[str, Literal["numerical", "categorical", "boolean"]]
    constraints: List[ConstraintCfg] = []
    generator: GeneratorCfg = GeneratorCfg(type="ctgan")

    @validator("target")
    def target_in_features(cls, v, values):
        if "feature_types" in values and v not in values["feature_types"]:
            raise ValueError("target not found in feature_types")
        return v

