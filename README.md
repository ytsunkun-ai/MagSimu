# MagSimu / softmag-predictor

软磁粉芯多尺度电磁性能预测研究原型：**输入校验 → 粒径分布 → DEM 压实 → 接触网络 → MicroMagnetic.jl 桥接 → 均匀化 → 损耗 → 绝缘耐压 → 校准 → 不确定度 → Markdown 报告**。

> **重要：`configs/` 中的材料、绝缘、损耗和耐压数据全部是 D 级占位值，不是真实 FeSiCr 材料记录。**
> 本仓库用于验证代码、数据契约和计算流程，不能直接用于材料选型、产品设计或对外性能声明。

## 快速开始

Windows PowerShell：

```powershell
uv venv .venv
uv pip install --python .venv\Scripts\python.exe -e .
.venv\Scripts\python.exe -m softmag validate --project configs/project.yaml
.venv\Scripts\python.exe -m softmag run-all --project configs/project.yaml --run-id demo
```

或者使用可编辑安装后的命令：

```powershell
softmag run-all --project configs/project.yaml
```

输出在 `results/<run_id>/`，其中：

```text
input_contract.json
missing_parameters.json
manifest.json
particles/
dem/
structure/
hysteresis/
homogenization/
losses/
dielectric/
calibration/
uncertainty/
reports/report.md
run_results.json
```

运行测试：

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## 目录结构

```text
configs/                输入 YAML（材料、DEM、微磁、实验、项目编排）
python/softmag/         Python 包
julia/                  MicroMagnetic.jl 作业脚本和 Project.toml
tests/                  单元与端到端测试
data/raw/               synthetic placeholder CSV（不可用于定量结论）
results/                每次运行的隔离输出
```

## 主要模块

| 模块 | 作用 |
|---|---|
| `schemas.py` | SI 输入契约、证据等级、缺失参数报告、Python→Julia 数据契约 |
| `particle_generator.py` | 数分布/体积分布显式换算，对数正态分位数拟合与采样 |
| `dem_runner.py` | 周期立方盒子中的简化 DEM 压实、接触表、外部结构导入 |
| `microstructure.py` | 机械/近场/电/磁网络，电阻/磁阻网络求解，RVE 筛查 |
| `micromagnetics.py` | Python→Julia 作业、MicroMagnetic.jl 调用、明确的 tanh 回退 |
| `homogenization.py` | Maxwell–Garnett / Bruggeman / 对数混合、相平均 B–H、小回线 |
| `bh_curve.py` | 回线闭合性、升/降支、磁滞能量、磁导率定义、矫顽力/剩磁 |
| `eddy_loss.py` | 孤立球低频涡流基线、粒径积分、集肤深度检查、Steinmetz 拟合 |
| `dielectric.py` | 随机绝缘厚度/缺陷、最弱瓶颈路径、Weibull 统计 |
| `calibration.py` | tanh 回线、Steinmetz、Weibull、损耗分离校准 |
| `uncertainty.py` | 只传播输入中**声明**的不确定度；情景扫描与概率区间分开 |
| `report.py` | 带 `model_status`、假设和限制的 Markdown 报告 |

## MicroMagnetic.jl

Julia 不是运行 Python 回退流程的必需条件。真实微磁接口需要 Julia ≥ 1.11：

```powershell
julia --project=julia julia/setup.jl
julia --project=julia julia/hysteresis.jl --input job.json --output outdir
```

`python/softmag/micromagnetics.py` 生成 `job.json` 并调用 `julia/hysteresis.jl`。如果 Julia 或
MicroMagnetic.jl 不可用，`auto` 模式会回退到 **tanh 唯象大回线**，并在每个输出中标记：

- `engine = python_tanh_fallback`
- `model_status = fallback_reduced_order`
- 明确的 `fallback_reason`

回退不是微磁学结果，不能替代真实 MicroMagnetic.jl 计算。

## 数据契约与单位

- 内部全部使用 SI：m、A/m、T、Hz、Ω·m、J/m³、W/m³。
- 粒径分位数必须声明 `number` 或 `volume` 基；DEM 采样按数基，体积统计单独报告。
- `magnetic_volume_fraction` 是体积分数，不是质量分数，也不自动等于相对密度。
- 机械接触、近场接触和电接触分开；**机械接触默认不电导通**。
- 电接触由缺陷/损伤概率采样；低缺陷率下不逾渗是预期物理结果。
- 击穿电压是接触网络的最弱瓶颈路径统计量，不是最大局部电场，也不是产品耐压保证。
- 微磁 5 nm 网格只代表小体积，不是 20 µm 全颗粒。

## 示例数据

`data/raw/example_*.csv` 由 `data/make_placeholders.py` 生成，列中标有
`synthetic_placeholder_generated_for_code_verification`。它们只用于验证拟合和报告流程。
替换为实测数据前，不得把校准结果解释为材料性能。

## 已知限制

- DEM 是简化几何原型，不是经过实验验证的接触动力学 DEM。
- 涡流模型是孤立球低频基线；集肤深度不满足时会标记 `outside_validity`。
- 磁滞损耗使用给定 B_peak 的归一化唯象小回线；必须用实测损耗 map 校准。
- 不确定度只传播显式声明的输入不确定度，不包含 DEM/微磁模型形式误差。
- 没有温度场、应力场、频域 Maxwell 全场求解或局部放电模型。

## License

MIT。引用或用于研究时，请保留输入证据等级、`model_status` 和限制说明。
