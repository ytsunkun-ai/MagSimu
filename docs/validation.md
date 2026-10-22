# 验证与使用边界

## 当前已自动测试

- SI 单位往返转换、幅值约定。
- 粒径分位数在体积基下保持 D50；数基 D50 小于体积基 D50。
- 小规模 DEM 达到目标体积分数并产生接触表。
- 串联/并联电阻网络、非逾渗电极。
- tanh 大回线闭合与面积；给定 B_peak 小回线峰值/闭合。
- 孤立球涡流 d² 标度、分布积分、Steinmetz 往返。
- minimax 最弱路径、Weibull 参数范围。
- tanh 回线三参数拟合参数回收。
- 端到端 `run-all`（小 DEM + fallback + 校准 + 报告）。

运行：

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## 真实 Julia / MicroMagnetic.jl 验证

已在 Windows 11 + Julia 1.11.9 + MicroMagnetic.jl 0.6.0 上完成：

- `julia/setup.jl` 按 `julia/Project.toml` / `julia/Manifest.toml` 实例化并预编译；
- `softmag julia-check` 强制真实 Julia、关闭 fallback，跑通 8×4×2 `box` 与
  16×16×8 `sphere`/`cluster` 的 5 场点完整回线；
- Julia 写出 `loop.csv` 与 `result.json`，Python 回读并计算
  `B = μ0(H + φM)`，闭合误差为 0；
- 形状未覆盖任何磁性单元时明确报错，不输出 NaN；
- Python 默认测试与真实 Julia 集成测试通过。

命令：

```powershell
julia --startup-file=no --project=julia julia/setup.jl
.venv\Scripts\python.exe -m softmag julia-check --project configs/project.yaml
$env:SOFTMAG_TEST_JULIA = "1"
.venv\Scripts\python.exe -m pytest -q tests/test_julia_integration.py
```

该烟测只证明接口、单位、状态和回读契约成立，**不证明网格收敛、RVE 代表性或材料
参数的真实性**。默认 64×16×16 / 41 场点完整作业未在本次验证中跑完；不得据烟测宣称
定量精度。部署步骤、输出契约和故障排查见
[`docs/deployment.md`](deployment.md)。

## 必须补充的实验验证

1. 粒径分布：激光衍射/图像分析，明确 number/volume 基。
2. DEM：压坯密度、SEM/CT 配位数、接触邻近距离分布。
3. 微磁学：真实颗粒/团簇的 TEM 晶粒、Ms、K、交换长度、阻尼。
4. 磁导率：初始、振幅、增量磁导率分别测量，注明频率/偏置/温度。
5. 损耗：B–f–T–偏置 map；至少分离磁滞/涡流/剩余损耗。
6. 绝缘：涂层厚度分布、缺陷统计、局部击穿/耐压分布。
7. 校准：独立 train/test 批次，禁止用同一批数据既拟合又宣称预测。

## 不得使用的结果

- 用 D 级占位材料参数计算出的任何“FeSiCr 性能”。
- 把机械接触数当作电接触数。
- 把最大局部电场当作击穿电压。
- 把 tanh fallback 结果当作 MicroMagnetic.jl 微磁学结果。
- 把 5 nm 子体积网格结果外推到 20 µm 全颗粒。
- 把平均粒径代替分布后的损耗/磁导率。
