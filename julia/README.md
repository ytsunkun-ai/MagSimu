# Julia / MicroMagnetic.jl

`Project.toml` 固定 MicroMagnetic.jl 0.7.x（需要 Julia ≥ 1.11）和 JSON。

```powershell
julia --project=julia julia/setup.jl
```

`hysteresis.jl` 接收 Python 生成的 `job.json`，输出 `loop.csv`（SI 单位）和 `result.json`：

```powershell
julia --project=julia julia/hysteresis.jl --input results/demo/hysteresis/job.json --output results/demo/hysteresis
```

Python 端不会假设 Julia 一定存在。若 Julia 未安装或作业失败，`auto` 模式会回退到
`python_tanh_fallback`，并把 `model_status` 标为 `fallback_reduced_order`。

## 限制

- 网格是代表性子体积，不是完整颗粒。5 nm 网格无法覆盖 20 µm 颗粒。
- `box` 表示颗粒内部子体积；`sphere`/`cluster` 被裁剪到网格可容纳尺寸，仅用于接口验证。
- 本脚本没有自动网格收敛、温度/应力耦合或颗粒间相互作用。
