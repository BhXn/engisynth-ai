
# Usage
### Example: Run pipeline

```bash
export PYTHONPATH=src
python src/engisynth/cli/run_pipeline.py \
    --csv data/iris_small.csv \
    --cfg configs/iris_small.yaml \
    --output outputs/iris_run1
```




## 新增功能概述


### 1. 约束系统
- **求和约束(SumEqual)**: 确保指定列的和等于给定值（如能量守恒）
- **范围约束(RangeConstraint)**: 限制数值在物理可行范围内
- **比例约束(RatioConstraint)**: 保持列之间的比例关系
- **单调性约束(MonotonicConstraint)**: 确保变量间的单调关系
- **相关性约束(CorrelationConstraint)**: 维持特定的相关系数

### 2. 物理量归一化
- **MinMax归一化**: 将数据缩放到[0,1]区间
- **Z-score标准化**: 转换为均值0、标准差1的分布
- **对数变换**: 处理偏态分布的数据

### 3. 噪声注入
- **高斯噪声**: 模拟正态分布的测量误差
- **均匀噪声**: 模拟均匀分布的误差
- **量化噪声**: 模拟数字化测量的量化效应


### 4. 自动约束识别
...

## 使用方法

### 1. 生成测试数据

```bash
python scripts/generate_test_data.py --output data/hardware_test.csv --samples 200
```

### 2. 配置文件示例

创建配置文件 `configs/hardware_example.yaml`：

```yaml
dataset: "hardware_module_data"
target: "performance_score"

feature_types:
  temperature: numerical
  pressure: numerical
  material_purity: numerical
  performance_score: numerical

constraints:
  - type: range
    cols: [temperature]
    min: 20.0
    max: 300.0
  
  - type: sum_equal
    cols: [energy_heat, energy_kinetic, energy_potential]
    value: 100.0
    tol: 0.01

noise:
  temperature:
    type: gaussian
    std: 2.0
    relative: false

normalization:
  pressure:
    method: log
    offset: 0.01

generator:
  type: constrained_ctgan
  epochs: 500
  batch_size: 32
```

### 3. 运行管道

```bash
python -m engisynth.cli.run_pipeline \
    --csv data/dataset-uci.csv \
    --cfg configs/dataset-uci_autogen.yaml \
    --output results/dataset-uci \
    --n-samples 1000
```


### 4. 半自动识别约束
```bash
python engisynth/scripts/auto_detect_constraints.py \
    --csv engisynth/data/hardware_test.csv \
    --out engisynth/configs/hardware_test_autogen.yaml \
    --target "performance_score"
```
## 约束处理流程

1. **数据归一化**: 根据配置对原始数据进行归一化处理
2. **模型训练**: 使用归一化后的数据训练CTGAN
3. **样本生成**: 从CTGAN生成候选样本
4. **反归一化**: 将生成的样本转换回原始尺度
5. **约束投影**: 使用迭代投影算法调整数据以满足约束
6. **噪声注入**: 添加配置的噪声以模拟测量误差
7. **约束验证**: 过滤出满足所有约束的有效样本

## 输出说明

运行后会在输出目录生成：

- `original.csv`: 原始数据
- `synthetic.csv`: 生成的合成数据
- `report.json`: 详细的评估报告（JSON格式）
- `report.txt`: 简化的文本报告

报告包含：
- ΔR指标：衡量合成数据在下游任务上的性能差异
- 约束满足度：每个约束的满足程度评分
- 统计信息：原始和合成数据的基本统计量对比

## 高级用法

### 自定义约束

创建新的约束类：

```python
from engisynth.constraints.base import Constraint
import pandas as pd

class CustomConstraint(Constraint):
    def __init__(self, param):
        self.param = param
    
    def is_satisfied(self, df: pd.DataFrame) -> pd.Series:
        # 实现约束检查逻辑
        pass
    
    def project(self, df: pd.DataFrame) -> pd.DataFrame:
        # 实现投影逻辑（可选）
        return df
```

### 扩展噪声模型

在 `ConstrainedGenerator._add_noise` 方法中添加新的噪声类型。

## 注意事项

1. **约束冲突**: 某些约束组合可能相互冲突，导致无法生成有效样本
2. **性能考虑**: 复杂约束和严格的容差会增加生成时间
3. **样本拒绝**: 如果约束过于严格，可能需要增加 `max_rejection_samples`

## 最佳实践

1. **逐步添加约束**: 从简单约束开始，逐步增加复杂度
2. **合理设置容差**: 过小的容差可能导致无法生成样本
3. **监控约束满足度**: 使用报告中的约束评分来调整参数
4. **平衡质量与效率**: 权衡生成质量和计算时间

## 数据集介绍
### 硬件模块数据集
- **数据来源**: 脚本生成的硬件模块性能数据
- **特征**: 温度、压力、材料纯度等
- **目标**: 预测性能评分
- **约束**: 温度范围、能量守恒等
- **噪声**: 高斯噪声、对数变换等
### Iris数据集
- **数据来源**: 经典的Iris花卉数据集
- **特征**: 花萼长度、花萼宽度、花瓣
- **目标**: 预测花卉种类
- **约束**: 无
### secondary_datasets
- **数据来源**: UCI Machine Learning Repository
- **特征**: 各种物理属性
- **目标**: 预测是否有毒
- **约束**: 暂无
### dataset-uci
- **数据来源**: UCI Machine Learning Repository
- **特征**: 病人的各种属性
- **目标**: Gallstone Status
- **约束**: 暂无
