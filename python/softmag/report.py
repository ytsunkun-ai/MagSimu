"""Markdown report generation with explicit model-status and limitation sections."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from .io_utils import ensure_dir, timestamp_utc, write_json


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e4):
            return f"{value:.{digits}e}"
        return f"{value:.{digits}g}"
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            return str(value)
    return str(value)


def _status(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, dict):
        return str(value.get("model_status", "n/a"))
    return str(value)


def _table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def _warnings(*items: Any) -> list[str]:
    out: list[str] = []
    for item in items:
        if not item:
            continue
        if isinstance(item, dict):
            item = item.get("warnings", [])
        if isinstance(item, str):
            out.append(item)
        else:
            out.extend(str(x) for x in item)
    return list(dict.fromkeys(out))


def generate_report(
    results: dict[str, Any],
    *,
    outdir: str | Path,
    title: str = "软磁粉芯多尺度电磁性能预测报告",
) -> dict[str, str]:
    out = ensure_dir(outdir)
    pred = results.get("prediction") or {}
    material = pred.get("material") or {}
    quality = material.get("parameter_quality")
    demo = str(quality).upper() in {"D", "UNKNOWN"}
    run_id = results.get("run_id") or pred.get("run_id")
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- 运行 ID：`{run_id}`")
    lines.append(f"- 生成时间（UTC）：`{timestamp_utc()}`")
    lines.append(f"- 输入材料：`{material.get('material_id', 'n/a')}`")
    lines.append(f"- 材料证据等级：`{quality}`")
    if demo:
        lines.append("")
        lines.append(
            "> **重要声明：材料参数为 D 级占位值（非真实 FeSiCr 数据）。** "
            "本报告只能证明代码/流程可运行，不能用于材料选型、器件设计或对外性能声明。"
        )
    lines.append("")

    # Executive summary
    hom = results.get("homogenization") or {}
    mm = results.get("micromagnetics") or {}
    losses = results.get("losses") or {}
    diel = results.get("dielectric") or {}
    cal = results.get("calibration") or {}
    unc = results.get("uncertainty") or {}
    lines.append("## 1. 结果摘要")
    lines.append("")
    summary_rows = [
        ("有效相对磁导率", hom.get("effective_relative_permeability"), _status(hom)),
        (
            "磁滞损耗（W/m³）",
            losses.get("hysteresis_loss_W_m3"),
            _status(losses),
        ),
        (
            "颗粒涡流损耗（W/m³）",
            losses.get("eddy_loss_W_m3"),
            _status(losses.get("eddy_loss") or losses),
        ),
        (
            "总损耗（W/m³）",
            losses.get("total_loss_W_m3"),
            _status(losses),
        ),
        (
            "击穿电压中位数（V）",
            (diel.get("breakdown_voltage_V") or {}).get("p50"),
            _status(diel),
        ),
        (
            "击穿场中位数（V/m）",
            (diel.get("breakdown_field_V_m") or {}).get("p50"),
            _status(diel.get("breakdown_field_V_m") or diel),
        ),
        ("回线拟合 R²", (cal.get("fits", {}).get("loop") or {}).get("r2_all"), _status(cal)),
    ]
    lines.append(_table(["指标", "数值", "model_status"], summary_rows))
    lines.append("")
    overall = _warnings(
        pred.get("warnings"),
        results.get("warnings"),
        mm.get("warnings"),
        hom.get("warnings"),
        losses.get("warnings"),
        diel.get("warnings"),
        cal.get("warnings"),
        unc.get("warnings"),
    )
    if overall:
        lines.append("**跨层警告（必须与数值一起阅读）：**")
        lines.append("")
        for w in overall[:20]:
            lines.append(f"- {w}")
        lines.append("")

    # Inputs
    lines.append("## 2. 输入与证据等级")
    lines.append("")
    lines.append(
        _table(
            ["项目", "值", "备注"],
            [
                ("材料", material.get("material_id"), material.get("composition")),
                ("参数质量", quality, material.get("parameter_source")),
                ("来源类型", material.get("source_type"), material.get("source_ref")),
                ("粒径 d10/d50/d90 (m)", f"{_fmt(pred.get('particle', {}).get('d10_m'))} / {_fmt(pred.get('particle', {}).get('d50_m'))} / {_fmt(pred.get('particle', {}).get('d90_m'))}", f"basis={pred.get('particle', {}).get('quantile_basis')}"),
                ("频率 / B 峰值", f"{_fmt(pred.get('excitation', {}).get('frequency_hz'))} Hz / {_fmt(pred.get('excitation', {}).get('flux_density_peak_T'))} T", "SI"),
            ],
        )
    )
    missing = results.get("missing_parameters") or {}
    if missing:
        lines.append("")
        lines.append("**缺失参数清单（未用默认值伪造）**")
        lines.append("")
        for layer, items in missing.items():
            lines.append(f"- `{layer}`: {', '.join(items) if items else '无'}")
    lines.append("")

    # Particle / DEM / structure
    lines.append("## 3. 粒径、DEM 与微结构")
    lines.append("")
    psum = results.get("particles") or {}
    lines.append("### 3.1 粒径分布")
    lines.append("")
    ps = psum.get("size_summary") or results.get("dem", {}).get("summary", {}).get("particle_size_summary") or {}
    dist = ps.get("distribution") or {}
    stats_v = dist.get("stats_volume") or {}
    stats_n = dist.get("stats_number") or {}
    lines.append(
        _table(
            ["统计量", "数值"],
            [
                ("分布类型", dist.get("kind") or ps.get("distribution_type")),
                ("输入基", dist.get("basis") or pred.get("particle", {}).get("quantile_basis")),
                ("体积基准 d50 (m)", stats_v.get("d50_m") or (ps.get("volume") or {}).get("d50_m")),
                ("数基准 d50 (m)", stats_n.get("d50_m") or (ps.get("number") or {}).get("d50_m")),
                ("拟合最大相对误差", (dist.get("fit_diagnostics") or {}).get("max_abs_relative_error")),
            ],
        )
    )
    lines.append("")
    dem = results.get("dem") or {}
    dems = dem.get("summary") or {}
    lines.append("### 3.2 DEM")
    lines.append("")
    lines.append(
        _table(
            ["项目", "值"],
            [
                ("颗粒数", dem.get("n_particles") or dems.get("n_particles")),
                ("盒子边长 (m)", dem.get("box_length_m") or dems.get("box_length_m")),
                ("颗粒体积分数", dems.get("achieved_particle_volume_fraction") or dems.get("particle_volume_fraction")),
                ("接触数", dem.get("n_contacts") or dems.get("n_contacts")),
                ("机械接触数", dems.get("n_mechanical_contacts")),
                ("最大重叠（相对）", (dems.get("relaxation") or {}).get("max_overlap_rel")),
                ("引擎状态", dems.get("engine_status")),
                ("model_status", dems.get("model_status")),
            ],
        )
    )
    st = results.get("structure") or {}
    lines.append("")
    lines.append("### 3.3 结构网络")
    lines.append("")
    lines.append(
        _table(
            ["网络", "逾渗", "数值", "备注"],
            [
                ("机械接触", (st.get("mechanical_network") or {}).get("percolating"), (st.get("coordination_number_mechanical") or {}).get("mean"), "gap <= 0"),
                ("近场接触", (st.get("near_contact_network") or {}).get("percolating"), (st.get("coordination_number_near") or {}).get("mean"), f"threshold={_fmt(st.get('near_gap_threshold_m'))} m"),
                ("电接触", (st.get("electrical") or {}).get("percolating"), (st.get("electrical") or {}).get("n_conductive_contacts"), "缺陷/损伤概率采样；不默认导通"),
                ("磁路", (st.get("magnetic") or {}).get("percolating"), (st.get("magnetic") or {}).get("n_active_edges"), "集中磁阻网络"),
            ],
        )
    )
    lines.append("")

    # Micromagnetics
    lines.append("## 4. 微磁学 / 回线")
    lines.append("")
    params = mm.get("parameters") or {}
    param_text = (
        f"Ms={_fmt(params.get('Ms_A_m'))} A/m, Hc={_fmt(params.get('Hc_A_m'))} A/m, "
        f"a={_fmt(params.get('a_A_m'))} A/m"
        if params
        else "n/a"
    )
    lines.append(
        _table(
            ["项目", "值", "状态"],
            [
                ("引擎", mm.get("engine"), _status(mm)),
                ("回线参数", param_text, _status(mm)),
                ("大回线面积（仅诊断，不用于 B_peak 损耗）", mm.get("loop_energy_density_J_m3_per_cycle"), _status(mm)),
                ("B_peak 小回线面积 (J/m³/cycle)", (losses.get("hysteresis_minor_loop") or {}).get("loop_energy_density_J_m3_per_cycle"), _status(losses.get("hysteresis_minor_loop") or {})),
                ("初始磁导率（模型）", mm.get("initial_permeability_model"), _status(mm)),
                ("网格", mm.get("mesh"), _status(mm)),
            ],
        )
    )
    if mm.get("fallback_reason"):
        lines.append("")
        lines.append(f"- 回退原因：{mm['fallback_reason']}")
    lines.append("")

    # Homogenization / losses
    lines.append("## 5. 均匀化与损耗")
    lines.append("")
    lines.append(
        _table(
            ["模型", "有效相对磁导率", "说明"],
            [
                ("Bruggeman", (hom.get("mixing") or {}).get("models", {}).get("bruggeman"), "对称有效介质"),
                ("Maxwell–Garnett", (hom.get("mixing") or {}).get("models", {}).get("maxwell_garnett"), "稀疏夹杂近似"),
                ("对数混合", (hom.get("mixing") or {}).get("models", {}).get("logarithmic"), "趋势参考"),
                ("DEM 磁路", (hom.get("network") or {}).get("effective_relative_permeability"), "接触磁阻网络"),
                ("采用值", hom.get("effective_relative_permeability"), hom.get("effective_permeability_source")),
            ],
        )
    )
    lines.append("")
    lines.append(
        _table(
            ["损耗项", "W/m³", "状态"],
            [
                ("磁滞", losses.get("hysteresis_loss_W_m3"), _status(losses)),
                ("颗粒涡流", losses.get("eddy_loss_W_m3"), _status(losses.get("eddy_loss") or losses)),
                ("颗粒间涡流", losses.get("interparticle_eddy_loss_W_m3"), _status(losses.get("interparticle_eddy") or losses)),
                ("总计", losses.get("total_loss_W_m3"), _status(losses)),
            ],
        )
    )
    if losses.get("total_is_partial"):
        lines.append("")
        lines.append(
            "> 总损耗只对可用分量求和；缺失分量没有用零替代。"
            "请查看 `losses/summary.json` 的 `total_is_partial` 和各子模型状态。"
        )
    if losses.get("steinmetz"):
        sm = losses["steinmetz"]
        lines.append("")
        lines.append(
            f"Steinmetz 拟合：`P = {_fmt(sm.get('k'))} f^{_fmt(sm.get('alpha'))} B^{_fmt(sm.get('beta'))}`，"
            f"R²={_fmt(sm.get('r2'))}，n={_fmt(sm.get('n_points'))}。"
        )
    lines.append("")

    # Dielectric
    lines.append("## 6. 绝缘与耐压")
    lines.append("")
    bv = diel.get("breakdown_voltage_V") or {}
    bf = diel.get("breakdown_field_V_m") or {}
    lines.append(
        _table(
            ["量", "p10", "p50", "p90", "Weibull k / λ"],
            [
                ("击穿电压 (V)", bv.get("p10"), bv.get("p50"), bv.get("p90"), f"{_fmt(bv.get('shape_k'))} / {_fmt(bv.get('scale_lambda'))}"),
                ("击穿场 (V/m)", bf.get("p10"), bf.get("p50"), bf.get("p90"), f"{_fmt(bf.get('shape_k'))} / {_fmt(bf.get('scale_lambda'))}"),
            ],
        )
    )
    lines.append("")
    lines.append(
        "> 该值是**接触网络最弱瓶颈路径**的统计分布，不是最大局部电场，也不是实测击穿电压。"
    )
    lines.append("")

    # Calibration
    lines.append("## 7. 校准")
    lines.append("")
    fits = cal.get("fits") or {}
    rows = []
    for name, fit in fits.items():
        rows.append(
            (
                name,
                _status(fit),
                (fit.get("r2_all") or fit.get("r2")) if isinstance(fit, dict) else None,
                fit.get("data_quality") if isinstance(fit, dict) else None,
            )
        )
    lines.append(_table(["拟合", "状态", "R²", "数据质量"], rows) if rows else "未找到校准数据。")
    lines.append("")

    # Uncertainty
    lines.append("## 8. 不确定度")
    lines.append("")
    outs = unc.get("outputs") or {}
    urows = []
    for name, stat in outs.items():
        urows.append((name, stat.get("p05"), stat.get("p50"), stat.get("p95"), _status(stat)))
    lines.append(_table(["输出", "p05", "p50", "p95", "状态"], urows) if urows else "未运行不确定度分析。")
    lines.append("")
    lines.append(
        "> 不确定度中的磁导率按 Bruggeman 混合模型重算，因此中位数与第 5 节采用的 DEM 磁路值不同；"
        "这是模型形式差异，不是数值噪声。"
    )
    if unc.get("diameter_scenarios"):
        lines.append("")
        lines.append("粒径情景（非概率区间）：")
        lines.append("")
        lines.append(_table(["d50 因子", "d50 (m)", "涡流损耗 (W/m³)"], [(s.get("d50_factor"), s.get("d50_m"), s.get("eddy_loss_W_m3")) for s in unc["diameter_scenarios"]]))
    lines.append("")

    # Limitations
    lines.append("## 9. 限制与禁止用法")
    lines.append("")
    for item in [
        "D 级材料参数只用于代码验证；不得作为真实材料性能结论。",
        "DEM 是简化几何原型，不是经过验证的接触动力学 DEM；必须用密度、SEM/CT 和配位数实验校验。",
        "机械接触不默认电导通；低缺陷率下电网络不逾渗是物理结果，不是数值失败。",
        "微磁学网格只覆盖代表性子体积，不是 20 µm 全颗粒；不得用 5 nm 网格冒充全颗粒。",
        "tanh 回退模型不是微磁学结果，必须与 MicroMagnetic.jl 结果分开报告。",
        "击穿电压由统计网络模型给出，不等于最大局部电场对应的电压，也不是产品耐压保证。",
        "所有外推、频率/温度/应力条件变化都必须重新验证。",
    ]:
        lines.append(f"- {item}")
    lines.append("")

    # Reproducibility
    lines.append("## 10. 可复现性与文件")
    lines.append("")
    manifest = results.get("manifest") or {}
    lines.append(
        _table(
            ["项目", "值"],
            [
                ("Git revision", manifest.get("git_revision")),
                ("Python", (manifest.get("software") or {}).get("python_version")),
                (
                    "NumPy",
                    ((manifest.get("software") or {}).get("packages") or {}).get("numpy"),
                ),
                ("Julia", (manifest.get("software") or {}).get("julia_version")),
                ("运行目录", results.get("run_dir")),
            ],
        )
    )
    lines.append("")
    lines.append("主要输出文件：`input_contract.json`、`particles/`、`dem/`、`structure/`、`hysteresis/`、`homogenization/`、`losses/`、`dielectric/`、`calibration/`、`uncertainty/`。")
    lines.append("")
    lines.append("---")
    lines.append("状态约定：`direct` 直接计算；`conditional` 依赖明确假设；`needs_calibration` 未校准；`outside_validity` 超出适用范围；`fallback_reduced_order` 降阶回退；`demo_only` 仅演示；`unavailable` 不可用。")
    lines.append("")
    path = out / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    write_json(out / "report_data.json", results)
    return {"markdown": str(path), "json": str(out / "report_data.json")}
