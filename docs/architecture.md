# 架构与数据流

```text
configs/material.yaml
        │
        ├─ schemas.py ── SI 校验 / 证据等级 / 缺失参数
        │
        ├─ particle_generator.py ── 数/体积基粒径分布
        │
        ├─ dem_runner.py ── 周期盒子压实 ──► particles.csv / contacts.csv
        │
        ├─ microstructure.py ── 机械/近场/电/磁网络 + RVE
        │
        ├─ micromagnetics.py ── job.json ──► julia/hysteresis.jl
        │                         └──────► loop.csv / result.json
        │                         └──────► tanh fallback（明确标记）
        │
        ├─ homogenization.py ── 混合律 + 磁路网络 + 相平均 B–H
        │
        ├─ eddy_loss.py / bh_curve.py ── 损耗与回线指标
        │
        ├─ dielectric.py ── 缺陷/绝缘 Monte Carlo + 最弱瓶颈路径
        │
        ├─ calibration.py ── 回线 / Steinmetz / Weibull
        │
        ├─ uncertainty.py ── 仅传播已声明不确定度
        │
        └─ report.py ── model_status / 假设 / 限制 / 复现清单
```

## 关键分离

1. **几何接触 ≠ 电接触**：接触表只描述几何间隙；电接触另按缺陷/损伤概率采样。
2. **混合模型 ≠ 微结构网络**：Bruggeman/MG/对数混合是无序有效介质近似；磁路网络依赖 DEM 接触拓扑。报告同时列出，不能把二者混为一个“精确值”。
3. **微磁学 ≠ 唯象回线**：Julia 成功时使用 `loop.csv`；Julia 不可用时使用 `tanh` fallback，并标为 `fallback_reduced_order`。
4. **大回线 ≠ 给定 B_peak 的损耗**：损耗使用给定 B_peak 的闭合小回线，不能用 Hmax=400 kA/m 的大回线面积直接乘频率。
5. **最弱路径统计 ≠ 最大局部电场**：击穿电压来自 minimax bottleneck path 的 Monte Carlo 分布。

## 每次运行的产物

- `input_contract.json`：规范化输入与派生体积分数来源。
- `missing_parameters.json`：缺失参数，不用默认值伪装。
- `manifest.json`：配置哈希、软件版本、随机种子、Git revision（若有）。
- 各步骤 `summary.json` 与 CSV。
- `reports/report.md`：面向读者的报告。

## 状态语义

| 状态 | 含义 |
|---|---|
| `direct` | 在声明范围内直接计算 |
| `conditional` | 依赖明确假设/模型选择 |
| `needs_calibration` | 尚未用独立实验校准 |
| `outside_validity` | 超出公式适用范围（如集肤深度） |
| `fallback_reduced_order` | 降阶回退，不是高保真结果 |
| `demo_only` | 仅用于代码验证的 D 级演示 |
| `unavailable` | 输入不足或计算失败，未填造数值 |
