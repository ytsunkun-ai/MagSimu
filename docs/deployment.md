# Julia + MicroMagnetic.jl 本地部署与使用指南

本文档面向在本仓库中启用**真实 MicroMagnetic.jl 微磁学路径**的用户。它覆盖
Windows 本地安装、环境固定、真实作业验证、日常命令、Python/Julia 数据契约、
故障排查和卸载。材料参数、绝缘参数、损耗与耐压数据仍全部是 D 级占位值，不能把
本文的烟测结果解释为真实 FeSiCr 性能。

## 1. 已验证环境与边界

本次部署在以下环境完成并实际运行了 MicroMagnetic.jl：

| 组件 | 已验证值 | 说明 |
|---|---|---|
| 操作系统 | Windows 11 10.0.26100 | x64 |
| Julia | 1.11.9 | 官方 Windows x64 zip |
| MicroMagnetic.jl | 0.6.0 | 由 `julia/Project.toml` + `julia/Manifest.toml` 固定 |
| JSON.jl | 1.12.1 | 作业 JSON 读写 |
| Python | 3.12.15 | 项目 `.venv` |
| Julia depot | `C:\Users\ytsun\.julia` | 用户级缓存与包 |
| Julia 安装目录 | `C:\Users\ytsun\AppData\Local\Programs\Julia-1.11.9` | 本次便携安装路径 |

真实契约烟测结果：

```text
status:   PASSED
engine:   MicroMagnetic.jl
package:  MicroMagnetic.jl 0.6.0
loop:     10 points
max |B - mu0(H+phi M)|: 0.000e+00 T
```

该烟测为 `8×4×2` 网格、5 个场点、完整回线，只验证：

- Julia 和固定版本的 MicroMagnetic.jl 可加载；
- Python 生成 `job.json`，Julia 读取并写出 `loop.csv`、`result.json`；
- H、M 单位换算与 `B = μ0(H + φM)` 闭合；
- 回退路径未被误当成真实结果。

它**不验证**网格收敛、RVE 代表性、材料参数真实性或全颗粒定量精度。默认完整配置
`64×16×16 / 41` 场点也属于代表性子体积，不是 20 µm 全颗粒。

## 2. 组件关系

```text
configs/project.yaml
        │
        ├─ Python: softmag run-all
        │       ├─ DEM / 接触网络 / 均匀化 / 损耗 / 耐压 / 报告
        │       └─ softmag.micromagnetics
        │                │  写 job.json
        │                ▼
        │          julia/hysteresis.jl
        │                │  MicroMagnetic.jl
        │                ▼
        │          loop.csv + result.json
        │                │
        └────────────────┴─ 回读、单位换算、B = μ0(H + φM)
```

Python 不解析 MicroMagnetic 的 Julia 内部结构，只依赖文件契约。这样 Julia 不可用时
可以明确回退；Julia 可用时也不会把回退结果混入真实结果。

## 3. 前置条件

- Windows 10/11 x64，至少约 2 GB 可用磁盘空间（Julia、包和 depot 会继续增长）。
- 可访问 Julia 官方下载站和包 registry；企业网络可能需要代理。
- PowerShell 5.1 或 PowerShell 7。
- 仓库已放在本机，例如 `E:\python\micrMagnetic`。
- 已创建 Python 虚拟环境（见第 4 节）。
- 可选：Git，用于查看版本和推送；运行模型本身不要求 Git。

以下命令默认从仓库根目录执行：

```powershell
Set-Location E:\python\micrMagnetic
```

路径含空格或中文时请保留引号。本仓库路径为纯 ASCII，可减少 Windows 工具链问题。

## 4. 准备 Python 环境

如果还没有 `.venv`：

```powershell
uv venv .venv
uv pip install --python .venv\Scripts\python.exe -e .
```

验证：

```powershell
.venv\Scripts\python.exe -m softmag validate --project configs/project_quickstart.yaml
.venv\Scripts\python.exe -m pytest -q
```

默认单元测试不依赖 Julia。结果应为 `23 passed, 3 skipped` 左右；3 个 skip 是
`box`/`sphere`/`cluster` 三个真实 Julia 集成测试（见第 7.4 节）。具体数量会随后续
测试增加而变化，以实际输出为准。

## 5. 安装 Julia

### 5.1 推荐：官方 Windows 安装包

1. 打开 <https://julialang.org/downloads/>。
2. 选择 Windows x86_64 的 1.11.x 安装包（`julia-1.11.x-win64.exe`）。
3. 安装时勾选 **Add Julia to PATH**。
4. 关闭并重新打开 PowerShell，让 PATH 生效。

```powershell
julia --version
# julia version 1.11.9
```

如果安装器没有加入 PATH，手动加入当前用户 PATH：

```powershell
$juliaBin = "$env:LOCALAPPDATA\Programs\Julia-1.11.9\bin"
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$parts = @($userPath -split ";" | Where-Object { $_ })
if ($parts -notcontains $juliaBin) {
    [Environment]::SetEnvironmentVariable("Path", (($parts + $juliaBin) -join ";"), "User")
}
$env:Path = "$juliaBin;$env:Path"
julia --version
```

如果实际版本目录不同，把 `Julia-1.11.9` 改成安装器创建的目录。

### 5.2 可复现：便携 zip（本次验证采用）

不需要管理员权限，解压到当前用户程序目录：

```powershell
$ver = "1.11.9"
$zip = "$env:TEMP\julia-$ver-win64.zip"
$dest = "$env:LOCALAPPDATA\Programs\Julia-$ver"
curl.exe -L --fail --retry 3 -o $zip `
  "https://julialang-s3.julialang.org/bin/winnt/x64/1.11/julia-$ver-win64.zip"
(Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
# 1ae5f16cb10cf23089e002626304c467a8d14ba750a7e02dff32df7cdec85074
Expand-Archive -LiteralPath $zip -DestinationPath $env:TEMP\julia-extract -Force
Move-Item "$env:TEMP\julia-extract\julia-$ver" $dest
```

然后按 5.1 节把 `$dest\bin` 加入 PATH。此步骤只执行一次；以后升级 Julia 时保留旧目录
直到新环境验证完成。

### 5.3 检查 Julia 与 depot

```powershell
julia --version
julia -e 'println(Sys.BINDIR); println(DEPOT_PATH[1])'
```

预期分别显示 Julia 版本、`...\Julia-1.11.9\bin` 和 `C:\Users\<用户>\.julia`。
如果 PowerShell 仍找不到 `julia`，使用 `where.exe julia` 检查实际命中路径，或直接用
完整路径调用。

## 6. 初始化仓库的 Julia 环境

`julia/Project.toml` 固定兼容范围，`julia/Manifest.toml` 锁定本次验证过的精确版本。
首次运行会下载约 66 个依赖并预编译：

```powershell
julia --startup-file=no --project=julia julia/setup.jl
```

预期最后输出类似：

```text
Julia environment ready in E:\python\micrMagnetic\julia
Julia:         1.11.9
MicroMagnetic: 0.6.0
JSON:          1.12.1
```

`setup.jl` 的行为：

1. 激活 `julia/` 项目；
2. 按 `Project.toml` / `Manifest.toml` 执行 `Pkg.instantiate()`；
3. 在临时目录中预编译，避免 `_precompile_*`、OVF、TXT 等副产物污染仓库；
4. 加载 `MicroMagnetic` 和 `JSON`，打印版本。

**不要**手工把 `MicroMagnetic` 兼容范围改成 `"0.7"`。当前 General registry 中
MicroMagnetic.jl 可用版本为 0.3.1–0.6.0；写成 0.7 会得到
`Unsatisfiable requirements ... no versions left`。本仓库已在 0.6.0 上验证接口。

如果 Manifest 丢失，`Pkg.instantiate()` 会在 `"0.6"` 兼容范围内重新解析并生成新
Manifest；这仍然是可运行的，但不再保证和本次验证逐字节一致。要复现验证环境，请从
仓库恢复 `julia/Manifest.toml`。

## 7. 验证安装

### 7.1 Julia 与包版本

```powershell
julia --startup-file=no --project=julia -e `
  'using Pkg, MicroMagnetic; println(VERSION); println(pkgversion(MicroMagnetic)); Pkg.status()'
```

预期看到 `1.11.9`、`0.6.0`，以及 `JSON`、`MicroMagnetic` 两项。

### 7.2 Python 单元测试

```powershell
.venv\Scripts\python.exe -m pytest -q
```

默认 Julia 集成测试会 skip；这属于设计行为，不是失败。

### 7.3 真实契约烟测

这是验证 Julia 部署是否成功的主要命令：

```powershell
.venv\Scripts\python.exe -m softmag julia-check --project configs/project.yaml
```

默认输出目录为 `results/julia_check/`。成功时应看到：

```text
status:   PASSED
engine:   MicroMagnetic.jl
package:  MicroMagnetic.jl 0.6.0
loop:     10 points
max |B - mu0(H+phi M)|: 0.000e+00 T
```

命令会强制 `mode=julia` 并关闭 fallback。找不到 Julia、Julia 报错、输出缺列或单位
不闭合都会返回非零退出码，不会用 tanh 回退掩盖失败。

可用参数：

| 参数 | 默认 | 用途 |
|---|---|---|
| `--output` | `results/julia_check` | 指定烟测输出目录 |
| `--mesh-nx/--mesh-ny/--mesh-nz` | `8/4/2` | 网格；仅为契约检查 |
| `--shape` | `box` | `box` / `sphere` / `cluster`，验证形状分支 |
| `--field-steps` | `5` | 完整回线的正向场点数；总点数为 2 倍 |
| `--hmax` | `1.0e5` A/m | 最大外加场 |
| `--timeout` | `600` s | Julia 子进程超时 |

例如用稍大的 `16×8×4` 验证多一点的并行/FFT 路径：

```powershell
.venv\Scripts\python.exe -m softmag julia-check `
  --project configs/project.yaml --mesh-nx 16 --mesh-ny 8 --mesh-nz 4
```

`sphere` 和 `cluster` 分支需要足够细的网格，否则形状可能不覆盖任何网格单元。默认
`8×4×2` 只用于 `box`；验证这两个分支请用例如 `16×16×8`：

```powershell
.venv\Scripts\python.exe -m softmag julia-check `
  --project configs/project.yaml --shape cluster `
  --mesh-nx 16 --mesh-ny 16 --mesh-nz 8
```

脚本在形状未覆盖任何磁性单元时会明确报错，不会输出 NaN 回线。

输出目录中的关键文件：

| 文件 | 生产者 | 内容 |
|---|---|---|
| `job.json` | Python | 材料、网格、形状、扫场、仿真参数 |
| `loop.csv` | Julia | `H_A_m,M_phase_A_m` |
| `result.json` | Julia | 引擎、版本、网格、警告、形状 |
| `julia_stdout.txt` / `julia_stderr.txt` | Python | 子进程日志；MicroMagnetic 日志多在 stderr |
| `loop.csv`（回读后重写） | Python | `H_A_m,M_A_m,M_macro_A_m,B_T` |
| `summary.json` | Python | 管道摘要、B–H 指标、假设 |
| `julia_check.json` | Python | 烟测检查表、误差、软件版本、结论 |

### 7.4 可选：把真实 Julia 检查纳入 pytest

```powershell
$env:SOFTMAG_TEST_JULIA = "1"
.venv\Scripts\python.exe -m pytest -q tests/test_julia_integration.py
```

没有设置该环境变量时，集成测试会 skip；设置后会在临时目录分别运行
`box`、`sphere`、`cluster` 三个形状的 `julia-check`，检查引擎、点数、B 闭合和
输出文件。该测试仍是烟测，不是网格收敛测试。

## 8. 日常使用

### 8.1 输入校验

```powershell
.venv\Scripts\python.exe -m softmag validate --project configs/project.yaml
```

会打印材料 ID、证据等级和缺失参数。D 级输入会给出警告，这是预期行为。

### 8.2 快速全流程（不需要 Julia，明确 fallback）

```powershell
.venv\Scripts\python.exe -m softmag run-all `
  --project configs/project_quickstart.yaml --run-id quickstart
```

`configs/project_quickstart.yaml` 引用 `configs/micromagnetics_fallback.yaml`，因此
不会启动 Julia。结果的 `engine` 必须是 `python_tanh_fallback`，`model_status` 必须是
`fallback_reduced_order`。这只用于检查数据流、报告和校准流程。

### 8.3 真实 Julia 全流程

确认烟测通过后：

```powershell
.venv\Scripts\python.exe -m softmag run-all `
  --project configs/project.yaml --run-id demo_julia
```

`configs/project.yaml` 引用 `configs/micromagnetics.yaml`，默认：

```yaml
mode: "auto"                 # 优先 Julia；失败时按 allow_fallback_models 决定
mesh_nx: 64
mesh_ny: 16
mesh_nz: 16
n_field_steps: 41
hmax_A_m: 4.0e5
timeout_s: 3600
```

注意：

- 默认 `64×16×16` 不是 20 µm 全颗粒，只是代表性子体积；禁止把结果直接外推到全颗粒。
- `41` 个场点生成 82 个弛豫段，运行时间取决于收敛步数、CPU、网格和 MKL/FFT 路径。
  本指南不虚报完整配置的耗时；先用 `julia-check` 和较小网格估计本机成本。
- 第一次调用会有 Julia JIT 编译开销，后续进程会快一些。
- 若要在失败时允许回退，保留 `solver.allow_fallback_models: true`；若要求严格真实结果，
  在材料配置中把它设为 `false`，此时 Julia 失败会使该步骤变为不可用或直接报错，而不会
  伪造数值。
- 要**强制**回退，把 `configs/micromagnetics.yaml` 的 `mode` 改为 `fallback`，并保留
  `engine`/`model_status` 标记。不要把 fallback 结果改名为 MicroMagnetic 结果。

### 8.4 手工运行一次 Julia 作业

`julia-check` 生成的 `job.json` 可以单独交给 Julia：

```powershell
julia --startup-file=no --project=julia julia/hysteresis.jl `
  --input results\julia_check\job.json `
  --output results\julia_check_manual
```

然后检查：

```powershell
Get-Content results\julia_check_manual\loop.csv -TotalCount 5
Get-Content results\julia_check_manual\result.json
```

脚本接受 `--input` 和 `--output`，单位约定见第 9 节。它不需要从仓库根目录启动，但
`--project=julia` 的相对路径要在仓库根目录下解释；最稳妥是使用绝对路径：

```powershell
julia --startup-file=no --project=E:\python\micrMagnetic\julia `
  E:\python\micrMagnetic\julia\hysteresis.jl `
  --input E:\python\micrMagnetic\results\julia_check\job.json `
  --output E:\python\micrMagnetic\results\julia_check_manual
```

### 8.5 输出和状态

每次 `run-all` 写入 `results/<run_id>/`。关键目录：

```text
hysteresis/job.json
hysteresis/loop.csv
hysteresis/result.json
hysteresis/summary.json
hysteresis/julia_stdout.txt
hysteresis/julia_stderr.txt
homogenization/
losses/
dielectric/
calibration/
uncertainty/
reports/report.md
```

状态语义：

| 状态 | 含义 |
|---|---|
| `conditional` | 在声明的假设和子体积/模型范围内计算 |
| `fallback_reduced_order` | tanh 唯象回退，不是微磁学 |
| `outside_validity` | 超出公式适用范围（如集肤深度） |
| `demo_only` | 仅用于 D 级代码演示 |
| `unavailable` | 输入不足或计算失败；未填造数值 |

## 9. Python ↔ Julia 数据契约

### 9.1 单位与符号

- H：A/m
- M：A/m；`M_phase_A_m` 是磁性相内磁化强度
- φ：磁性体积分数，无量纲
- `M_macro_A_m = φ · M_phase_A_m`
- B：T，且 `B = μ0 (H + M_macro_A_m)`，`μ0 = 4π×10⁻⁷ H/m`
- 网格长度：m；时间：s；频率：Hz

MicroMagnetic.jl 的 `hysteresis` 返回的 H 单位是 mT（其内部 `mT = 1e-3/μ0`），
`average_m` 返回归一化磁化强度。`julia/hysteresis.jl` 在边界上显式转换：

```julia
H_A_m = H_mT .* (1.0e-3 / MU0)
M_A_m = m .* Ms
```

Python 随后计算宏观量，避免在 Julia 侧混用相磁化强度和宏观磁感应强度。

### 9.2 `job.json` 关键字段

```json
{
  "schema_version": "1.0",
  "run_id": "...",
  "material": {
    "Ms_A_m": 1250000.0,
    "exchange_A_J_m": 1e-11,
    "anisotropy_K_J_m3": 8000.0,
    "damping_alpha": 0.3,
    "parameter_quality": "D"
  },
  "mesh": {
    "nx": 64, "ny": 16, "nz": 16,
    "dx_m": 5e-9, "dy_m": 5e-9, "dz_m": 5e-9,
    "extent_m": [3.2e-7, 8e-8, 8e-8],
    "note": "代表性子体积，不是完整颗粒"
  },
  "shape": {"model": "box"},
  "sweep": {
    "H_A_m": [-400000, -390000, "..."],
    "direction": [1.0, 0.0, 0.0],
    "full_loop": true
  },
  "simulation": {
    "demag": true,
    "m0": [-1.0, 0.0, 0.0],
    "stopping_dmdt": 0.01,
    "driver": "LLG",
    "integrator": "BS23"
  },
  "microstructure": {
    "magnetic_volume_fraction": 0.6
  }
}
```

注意：

- `Ms_A_m`、交换常数和 anisotropy 缺失时，Python 在构建作业时直接报错，不静默填
  “合理默认值”。需要严格真实作业时请把参数补全；只做流程检查可显式使用 fallback。
- `shape=box` 是颗粒内部子体积；`sphere` 也只是网格可容纳的代表性球；`cluster` 是
  示意性颗粒簇，不是 DEM 团聚体。
- `demag` 会显著增加 FFT/卷积成本。

### 9.3 Julia 输出

`loop.csv`：

```text
H_A_m,M_phase_A_m
-1.000000000000e+05,-1.245983157509e+06
...
```

`result.json` 至少包含：

```json
{
  "schema_version": "1.0",
  "model_status": "conditional",
  "engine": "MicroMagnetic.jl",
  "package_version": "0.6.0",
  "julia_version": "1.11.9",
  "mesh": {"...": "..."},
  "shape": {"model": "box", "description": "..."},
  "n_points": 10,
  "Ms_A_m": 1250000.0,
  "warnings": ["..."]
}
```

Python 只接受同时具备以下条件的输出：

1. 有 `H_A_m`，且有 `M_phase_A_m` 或 `B_T`；
2. 至少 5 个有效点；
3. Julia 退出码为 0 且 `loop.csv` 存在；
4. `result.json` 可读（可选元数据，但正常作业应存在）；
5. 回退没有被关闭时才会在异常后回退；`julia-check` 和严格模式不会回退。

## 10. 配置项速查

`configs/micromagnetics.yaml`：

| 键 | 作用 | 注意 |
|---|---|---|
| `mode` | `auto` / `julia` / `fallback` | 严格真实用 `julia` 并关闭 fallback |
| `julia_executable` | Julia 命令或绝对路径 | PATH 未刷新时可填绝对路径 |
| `timeout_s` | 子进程超时 | 完整扫场要大；超时会报错/按策略回退 |
| `particle_shape` | `box` / `sphere` / `cluster` | 都不是完整颗粒 |
| `mesh_nx/ny/nz` | 子体积网格 | 与 dx/dy/dz 共同决定解析尺度 |
| `dx_m/dy_m/dz_m` | 网格步长 | 5 nm 不能覆盖 20 µm 全颗粒 |
| `Ms_A_m` / `exchange_A_J_m` / `anisotropy_K_J_m3` / `damping_alpha` | 覆盖材料值 | `null` 时取材料配置；缺失会报错 |
| `demag` | 退磁场 | FFT 成本较高 |
| `m0` | 初始磁化方向 | 必须是三维非零向量 |
| `hmax_A_m` | 最大扫场 | 单位 A/m |
| `n_field_steps` | 正向场点数 | 完整回线总点数为 2 倍 |
| `stopping_dmdt` | 弛豫停止阈值 | 越小越慢、通常越接近收敛 |
| `sweep` | `full_loop` / `up_only` | Julia 始终计算完整回线，up_only 由 `julia/hysteresis.jl` 截断 |
| `fallback_model` | 唯一允许值 `tanh_major_loop` | 状态必须标记为降阶 |

项目级：

- `configs/project_quickstart.yaml`：显式 fallback 的快速全流程；
- `configs/project.yaml`：真实 Julia `auto` 路径，默认完整子体积配置；
- `configs/material.yaml`：材料和证据等级；D 级占位；
- `configs/experiment.yaml`：实验校准 CSV 路径。

## 11. 资源与性能

本次本机（Windows 11、CPU、预热后的 Julia 缓存）：

| 作业 | 规模 | 观察 |
|---|---|---|
| 直接 Julia 烟测 | 8×4×2，5 场点 | 约 5–8 s，含进程启动 |
| `julia-check` | 16×8×4，5 场点 | 约 8 s |
| 默认完整配置 | 64×16×16，41 场点 | 本次未跑完，不能给出可信时长 |

影响耗时的主要因素：

1. 发送/退磁 FFT 与网格元胞数；
2. 每个场点的 LLG 收敛步数（受 `stopping_dmdt`、阻尼、各向异性影响）；
3. 完整回线的 2 倍场点数；
4. 首次 JIT 编译（显著但通常只发生一次）；
5. CPU 核数、MKL 路径、内存带宽。

生产使用前必须另做网格收敛和 RVE 收敛；`julia-check` 不是收敛证明。

## 12. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `julia : 无法将“julia”项识别...` | PATH 未刷新/未加入 | 新开终端；用 5.1 节命令或绝对路径 |
| `Unsatisfiable requirements ... MicroMagnetic ... 0.7` | compat 写成 0.7，registry 无此版本 | 保持 `MicroMagnetic = "0.6"`；恢复 `Project.toml`/`Manifest.toml` |
| `julia-check` 找不到 Julia | 子进程 PATH 与交互 shell 不同 | 先 `where.exe julia`；在 YAML 写 `julia_executable` 绝对路径 |
| 下载 registry/包超时 | 网络/代理 | 设置 `HTTPS_PROXY`/`HTTP_PROXY`，或重试 `julia/setup.jl` |
| `Pkg.precompile()` 失败 | 缓存损坏、磁盘/内存不足 | 释放空间；重试 `Pkg.instantiate(); Pkg.precompile()`；再查完整 stderr |
| `loop.csv` 不存在 | Julia 异常、超时、输出目录不可写 | 查看 `julia_stderr.txt`；手工运行 8.4 节命令 |
| `engine = python_tanh_fallback` | Julia 不可用/运行失败且允许回退 | 看 `fallback_reason`；严格模式请关闭回退并修好 Julia |
| `UnicodeDecodeError` / 日志乱码 | 子进程编码不是 UTF-8 | 本仓库已用 UTF-8 捕获；若二次开发请保留 `encoding="utf-8"` |
| `julia-check` 超时 | 网格/场点/收敛阈值过大 | 降低 `--mesh-*`、`--field-steps`，或增大 `--timeout` |
| 运行后仓库出现 `_precompile_*` / `softmag_llg.txt` | 旧版本或手工在仓库根目录直接跑 MicroMagnetic | 删除这些副产物；用当前脚本（临时工作目录）和 `.gitignore` |
| `B` 与 `μ0(H+φM)` 不闭合 | 单位/φ 口径错误 | 用 `julia-check` 检查；不要手工改 loop.csv 的物理含义 |

诊断时优先保留：

```text
results/<run_id>/hysteresis/job.json
results/<run_id>/hysteresis/julia_stderr.txt
results/<run_id>/hysteresis/julia_stdout.txt
results/<run_id>/hysteresis/result.json
```

不要把 D 级占位材料、烟测网格或 fallback 结果贴成“FeSiCr 实测性能”。

## 13. 更新与卸载

### 更新 Julia 包

只有在明确需要时更新。更新会改变 Manifest，必须重新跑 `julia-check`、完整测试和
至少一个小规模收敛检查，然后提交新的 Manifest。

```powershell
julia --startup-file=no --project=julia -e `
  'using Pkg; Pkg.update(); Pkg.precompile()'
julia --startup-file=no --project=julia julia/setup.jl
```

不要盲目把 MicroMagnetic 主版本升级到尚未在 registry 发布的版本；先确认版本和 API。

### 卸载 Julia

1. 从当前用户 PATH 中删除 Julia 的 `bin` 目录；
2. 删除 `%LOCALAPPDATA%\Programs\Julia-1.11.9`；
3. 如需清理包缓存，删除 `%USERPROFILE%\.julia`（这会删除所有 Julia 项目的 depot，
   不只本项目）；
4. 不要删除仓库中的 `julia/Project.toml`、`julia/Manifest.toml` 和脚本，它们是可复现
   环境的一部分。

## 14. 验证记录

2026-10-23 本地验证：

- Julia 1.11.9 安装并加入当前用户 PATH；
- `julia/setup.jl` 成功加载 MicroMagnetic 0.6.0、JSON 1.12.1；
- 真实作业 `box` 8×4×2、`sphere`/`cluster` 16×16×8，均为 5 场点完整回线并通过；
- `loop.csv` 10 点，`result.json` 引擎为 `MicroMagnetic.jl`；
- `max |B - μ0(H + φM)| = 0`，`M_macro = φ M_phase` 校验通过；
- 形状未覆盖任何磁性单元时会明确报错，不输出 NaN；
- Python 默认测试通过，真实 Julia 集成测试通过（设置 `SOFTMAG_TEST_JULIA=1`，覆盖三种形状）；
- 未运行默认 64×16×16 / 41 场点完整作业；其耗时不作承诺。

验证命令：

```powershell
julia --startup-file=no --project=julia julia/setup.jl
.venv\Scripts\python.exe -m softmag julia-check --project configs/project.yaml
.venv\Scripts\python.exe -m pytest -q
$env:SOFTMAG_TEST_JULIA = "1"
.venv\Scripts\python.exe -m pytest -q tests/test_julia_integration.py
```

## 15. 使用边界

- 所有材料、绝缘、损耗、耐压和实验数据仍是 D 级占位值，不能用于材料选型或产品声明。
- 微磁网格是代表性子体积，不是 20 µm 全颗粒；不能直接把结果乘颗粒数。
- 未自动证明网格收敛、RVE 代表性或实验可辨识性。
- 机械接触不等于电接触；低缺陷率下电网络不逾渗是预期结果。
- 击穿电压来自最弱瓶颈路径的 Monte Carlo 统计，不是最大局部电场。
- tanh fallback 不是微磁学；真实 MicroMagnetic.jl 结果是 `conditional`，不是已校准的
  材料性能。

需要定量结论时，请替换占位数据，补充网格/RVE 收敛、独立实验校准和不确定度分析，
并在报告中保留完整 `model_status`、假设和限制。
