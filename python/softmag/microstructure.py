"""Microstructure extraction, contact networks and network transport solvers.

The DEM structure is a geometric scaffold.  Mechanical contact does **not**
imply electrical contact.  This module therefore keeps three networks separate:

* mechanical contact network (gap <= 0),
* near-contact network (gap <= a user-defined tolerance),
* electrical network, where each contact is conductive only with a
  defect/damage probability derived from the insulation model.

The electrical and magnetic network solvers use a scalar potential on a graph
with electrode slabs.  Contacts that wrap across the x-periodic boundary are
excluded for transport, because the electrodes are open surfaces at x=0 and x=L.
The y/z directions remain periodic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from .dem_runner import Structure
from .schemas import InsulationParameters
from .units import MU0


# ---------------------------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------------------------
def coordination_numbers(
    structure: Structure, *, gap_threshold_m: float = 0.0
) -> np.ndarray:
    n = structure.n_particles
    z = np.zeros(n, dtype=int)
    if structure.contacts.empty:
        return z
    gap = structure.contacts["surface_gap_m"].to_numpy(dtype=float)
    ii = structure.contacts.loc[gap <= gap_threshold_m, "particle_i"].to_numpy(int)
    jj = structure.contacts.loc[gap <= gap_threshold_m, "particle_j"].to_numpy(int)
    np.add.at(z, ii, 1)
    np.add.at(z, jj, 1)
    return z


def _percolation_stats(
    n: int,
    i: np.ndarray,
    j: np.ndarray,
    *,
    centers: np.ndarray | None = None,
    box_length_m: float | None = None,
    span_fraction_threshold: float = 0.75,
) -> dict[str, Any]:
    """Connected components plus a geometric spanning test.

    ``percolating`` means that at least one component spans at least
    ``span_fraction_threshold`` of the periodic box in one axis.  It is **not**
    the same as "all particles form one component" and it is not an electrode
    transport result.  Transport percolation is reported separately by
    :func:`electrical_network`.
    """
    labels = union_find_labels(n, i, j)
    if n == 0:
        return {
            "n_components": 0,
            "largest_component_fraction": 0.0,
            "percolating": False,
            "spanning_axes": [],
            "note": "空结构",
        }
    sizes = np.bincount(labels)
    largest = int(sizes.max()) if sizes.size else 0
    largest_label = int(np.argmax(sizes)) if sizes.size else -1
    spanning_axes: list[str] = []
    largest_span: dict[str, float] = {}
    if centers is not None and box_length_m is not None and float(box_length_m) > 0:
        centers = np.asarray(centers, dtype=float)
        if centers.shape != (n, 3):
            raise ValueError("centers 必须为 (n,3)")
        L = float(box_length_m)
        # Compute extents per component.  A periodic cluster with points near
        # both x=0 and x=L has a large raw span; this is sufficient for the
        # geometric screening used here.
        for label in np.unique(labels):
            mask = labels == label
            if int(mask.sum()) < 2:
                continue
            pts = centers[mask]
            spans = (pts.max(axis=0) - pts.min(axis=0)) / L
            if label == largest_label:
                largest_span = {
                    "x": float(spans[0]),
                    "y": float(spans[1]),
                    "z": float(spans[2]),
                }
            for axis, value in zip(("x", "y", "z"), spans):
                if float(value) >= span_fraction_threshold and axis not in spanning_axes:
                    spanning_axes.append(axis)
    return {
        "n_components": int(sizes.size),
        "largest_component_size": largest,
        "largest_component_fraction": float(largest / n),
        "largest_component_span_fraction": largest_span,
        "percolating": bool(spanning_axes),
        "spanning_axes": spanning_axes,
        "percolation_definition": (
            f"组件在至少一个轴上跨度 >= {span_fraction_threshold:.2f} L；"
            "不是电极间输运逾渗"
        ),
    }


def structure_summary(
    structure: Structure,
    *,
    insulation_thickness_m: float | None = None,
    near_gap_m: float | None = None,
) -> dict[str, Any]:
    """Compute geometric and topological descriptors of a DEM structure."""
    p = structure.particles
    c = structure.contacts
    n = structure.n_particles
    L = structure.box_length_m
    volumes = p["volume_m3"].to_numpy(float) if "volume_m3" in p else (
        (4.0 / 3.0) * math.pi * p["radius_m"].to_numpy(float) ** 3
    )
    phi = float(volumes.sum() / L**3)
    d = 2.0 * p["radius_m"].to_numpy(float)
    if near_gap_m is None:
        near_gap_m = (
            max(3.0 * insulation_thickness_m, 0.02 * float(np.median(d)))
            if insulation_thickness_m is not None
            else 0.02 * float(np.median(d))
        )
    z_mech = coordination_numbers(structure, gap_threshold_m=0.0)
    z_near = coordination_numbers(structure, gap_threshold_m=near_gap_m)
    if c.empty:
        gap_mech = np.empty(0)
        gap_all = np.empty(0)
        gap_i = np.empty(0, dtype=int)
        gap_j = np.empty(0, dtype=int)
    else:
        gap_all = c["surface_gap_m"].to_numpy(float)
        mask = gap_all <= 0.0
        gap_mech = gap_all[mask]
        gap_i = c.loc[mask, "particle_i"].to_numpy(int)
        gap_j = c.loc[mask, "particle_j"].to_numpy(int)
    centers = structure.centers
    mech_perc = _percolation_stats(
        n, gap_i, gap_j, centers=centers, box_length_m=L
    )
    near_mask = gap_all <= near_gap_m
    near_perc = _percolation_stats(
        n,
        c.loc[near_mask, "particle_i"].to_numpy(int) if not c.empty else np.empty(0, int),
        c.loc[near_mask, "particle_j"].to_numpy(int) if not c.empty else np.empty(0, int),
        centers=centers,
        box_length_m=L,
    )
    surface_area = float(np.sum(4.0 * math.pi * p["radius_m"].to_numpy(float) ** 2))
    n_mech = int(np.sum(gap_all <= 0.0)) if gap_all.size else 0
    return {
        "schema_version": "1.0",
        "n_particles": n,
        "box_length_m": float(L),
        "particle_volume_fraction": phi,
        "porosity_fraction": float(1.0 - phi),
        "mean_diameter_m": float(np.mean(d)),
        "volume_weighted_mean_diameter_m": float(
            np.sum(volumes * d) / np.sum(volumes)
        ),
        "specific_surface_area_m2_m3": surface_area / L**3,
        "n_contacts": int(len(c)),
        "n_mechanical_contacts": n_mech,
        "n_near_contacts": int(np.sum(near_mask)) if gap_all.size else 0,
        "near_gap_threshold_m": float(near_gap_m),
        "mechanical_surface_gap": {
            "mean_m": float(np.mean(gap_mech)) if gap_mech.size else None,
            "min_m": float(np.min(gap_mech)) if gap_mech.size else None,
            "max_m": float(np.max(gap_mech)) if gap_mech.size else None,
        },
        "surface_gap_quantiles_m": (
            {
                f"p{int(q * 100)}": float(np.quantile(gap_all, q))
                for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
            }
            if gap_all.size
            else {}
        ),
        "coordination_number_mechanical": {
            "mean": float(z_mech.mean()) if n else None,
            "std": float(z_mech.std(ddof=1)) if n > 1 else None,
            "min": int(z_mech.min()) if n else None,
            "max": int(z_mech.max()) if n else None,
            "histogram": _histogram(z_mech),
        },
        "coordination_number_near": {
            "mean": float(z_near.mean()) if n else None,
            "std": float(z_near.std(ddof=1)) if n > 1 else None,
            "histogram": _histogram(z_near),
        },
        "mechanical_network": mech_perc,
        "near_contact_network": near_perc,
        "model_status": "conditional",
        "assumptions": [
            "结构来自简化几何 DEM 或用户导入的外部 DEM",
            "机械接触不等于电接触",
            "配位数对接触判定阈值敏感，报告阈值时必须同时给出",
        ],
        "warnings": list(structure.summary.get("warnings", [])),
    }


def _histogram(values: np.ndarray) -> dict[str, int]:
    if values.size == 0:
        return {}
    uniq, counts = np.unique(values, return_counts=True)
    return {str(int(u)): int(c) for u, c in zip(uniq, counts)}


# ---------------------------------------------------------------------------
# Connected components / electrodes
# ---------------------------------------------------------------------------
def union_find_labels(n: int, i: Iterable[int], j: Iterable[int]) -> np.ndarray:
    parent = np.arange(n, dtype=int)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(i, j):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    roots = np.array([find(x) for x in range(n)], dtype=int)
    # Relabel roots densely for compact component statistics.
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def electrode_indices(
    centers: np.ndarray,
    box_length_m: float,
    *,
    axis: int = 0,
    slab_fraction: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    """Select particles in the low/high slab normal to ``axis``."""
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers 必须是 (N,3)")
    if not 0.02 <= slab_fraction < 0.5:
        raise ValueError("slab_fraction 应位于 [0.02, 0.5)")
    coord = centers[:, axis]
    L = float(box_length_m)
    a = np.flatnonzero(coord <= slab_fraction * L)
    b = np.flatnonzero(coord >= (1.0 - slab_fraction) * L)
    if a.size == 0:
        a = np.argsort(coord)[: max(1, int(0.03 * centers.shape[0]))]
    if b.size == 0:
        b = np.argsort(coord)[-max(1, int(0.03 * centers.shape[0])) :]
    return a.astype(int), b.astype(int)


# ---------------------------------------------------------------------------
# Generic graph transport
# ---------------------------------------------------------------------------
@dataclass
class NetworkSolution:
    percolating: bool
    total_flux: float
    effective_resistance: float
    effective_resistivity: float | None
    effective_conductivity: float | None
    n_active_nodes: int
    n_active_edges: int
    potential: np.ndarray | None
    warnings: list[str]


def solve_network(
    n_nodes: int,
    edges_i: np.ndarray,
    edges_j: np.ndarray,
    conductance: np.ndarray,
    electrodes_a: np.ndarray,
    electrodes_b: np.ndarray,
    *,
    length_m: float,
    area_m2: float,
    unit_potential: float = 1.0,
) -> NetworkSolution:
    """Solve a resistor network between two fixed-potential electrode sets.

    The returned ``effective_resistivity`` uses ``R * area / length``.  If the
    electrodes are not connected by the edge set, ``percolating`` is False and
    no effective transport value is reported.
    """
    warnings: list[str] = []
    i = np.asarray(edges_i, dtype=int)
    j = np.asarray(edges_j, dtype=int)
    g = np.asarray(conductance, dtype=float)
    keep = np.isfinite(g) & (g > 0) & (i >= 0) & (j >= 0) & (i < n_nodes) & (
        j < n_nodes
    )
    i, j, g = i[keep], j[keep], g[keep]
    if i.size == 0 or electrodes_a.size == 0 or electrodes_b.size == 0:
        return NetworkSolution(
            False, 0.0, float("inf"), None, None, 0, int(i.size), None,
            ["无有效导电边或电极集合为空"],
        )
    labels = union_find_labels(n_nodes, i, j)
    common = set(labels[electrodes_a]).intersection(set(labels[electrodes_b]))
    if not common:
        return NetworkSolution(
            False, 0.0, float("inf"), None, None, 0, int(i.size), None,
            ["两电极之间没有连通路径（未逾渗）"],
        )
    active = np.isin(labels, list(common))
    # Restrict all fixed electrode nodes to the percolating components.
    fixed_a = np.unique(electrodes_a[active[electrodes_a]])
    fixed_b = np.unique(electrodes_b[active[electrodes_b]])
    active_idx = np.flatnonzero(active)
    map_global_to_local = -np.ones(n_nodes, dtype=int)
    map_global_to_local[active_idx] = np.arange(active_idx.size)
    il = map_global_to_local[i]
    jl = map_global_to_local[j]
    keep2 = (il >= 0) & (jl >= 0)
    il, jl, g = il[keep2], jl[keep2], g[keep2]
    m = active_idx.size
    rows = np.concatenate([il, jl, il, jl])
    cols = np.concatenate([il, jl, jl, il])
    data = np.concatenate([g, g, -g, -g])
    Lmat = coo_matrix((data, (rows, cols)), shape=(m, m)).tocsr()
    v = np.full(m, np.nan)
    a_loc = np.unique(map_global_to_local[fixed_a])
    b_loc = np.unique(map_global_to_local[fixed_b])
    v[a_loc] = unit_potential
    v[b_loc] = 0.0
    fixed_mask = ~np.isnan(v)
    free = ~fixed_mask
    if np.any(free):
        Luu = Lmat[free][:, free].tocsc()
        Luf = Lmat[free][:, fixed_mask]
        rhs = -Luf @ v[fixed_mask]
        try:
            import warnings as _warnings

            with _warnings.catch_warnings():
                _warnings.simplefilter("error", MatrixRankWarning)
                sol = spsolve(Luu, rhs)
            if not np.all(np.isfinite(sol)):
                raise RuntimeError("线性求解返回非有限值")
            v[free] = sol
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"网络线性求解失败或奇异：{exc}")
            return NetworkSolution(
                False, 0.0, float("inf"), None, None, int(m), int(il.size), None,
                warnings,
            )
    # Joule power with unit voltage equals total current.
    dv = v[il] - v[jl]
    total_flux = float(np.sum(g * dv**2))
    if not np.isfinite(total_flux) or total_flux <= 0:
        return NetworkSolution(
            False, 0.0, float("inf"), None, None, int(m), int(il.size), v, 
            warnings + ["网络电流为零，无法定义有效输运系数"],
        )
    resistance = unit_potential / total_flux
    rho = resistance * area_m2 / length_m
    sigma = 1.0 / rho if rho > 0 else None
    return NetworkSolution(
        True,
        total_flux,
        float(resistance),
        float(rho),
        float(sigma) if sigma is not None else None,
        int(m),
        int(il.size),
        v,
        warnings,
    )


# ---------------------------------------------------------------------------
# Electrical network
# ---------------------------------------------------------------------------
def contact_conductive_probability(
    *,
    gap_m: float,
    overlap_m: float,
    insulation_thickness_m: float,
    defect_probability: float,
    damage_scale_m: float | None = None,
) -> float:
    """Phenomenological probability that a mechanical contact conducts.

    * An overlapping contact can damage the coating; the damage probability
      grows with ``overlap / damage_scale``.
    * A positive gap can only conduct through a defect/pinhole; the probability
      decays with ``gap / coating thickness``.

    This is a model assumption, not a measured law.  It is deliberately reported
    alongside the network result.
    """
    p0 = float(np.clip(defect_probability, 0.0, 1.0))
    t = max(float(insulation_thickness_m), 1e-12)
    scale = damage_scale_m or max(5.0 * t, 1e-9)
    if overlap_m > 0:
        p = 1.0 - (1.0 - p0) * math.exp(-overlap_m / scale)
    else:
        p = p0 * math.exp(-max(gap_m, 0.0) / t)
    return float(np.clip(p, 0.0, 0.999))


def build_electrical_edges(
    structure: Structure,
    insulation: InsulationParameters,
    *,
    rng: np.random.Generator,
    particle_resistivity_ohm_m: float | None,
    near_gap_m: float | None = None,
    transport_axis: int = 0,
) -> dict[str, Any]:
    """Sample conductive contacts and assign edge resistances."""
    c = structure.contacts
    if c.empty:
        return {
            "i": np.empty(0, int),
            "j": np.empty(0, int),
            "conductance": np.empty(0),
            "n_candidates": 0,
            "n_conductive": 0,
            "warnings": ["接触表为空，无法建立电网络"],
        }
    if particle_resistivity_ohm_m is None:
        raise ValueError("缺少颗粒电阻率，无法计算电网络")
    radii = structure.radii
    mean_d = float(np.mean(2.0 * radii))
    t_ins = insulation.thickness_mean_m or 1e-7
    p0 = insulation.defect_probability if insulation.defect_probability is not None else 0.0
    if near_gap_m is None:
        near_gap_m = max(5.0 * t_ins, 0.05 * mean_d)
    gap = c["surface_gap_m"].to_numpy(float)
    candidate = gap <= near_gap_m
    if not np.any(candidate):
        return {
            "i": np.empty(0, int),
            "j": np.empty(0, int),
            "conductance": np.empty(0),
            "n_candidates": 0,
            "n_conductive": 0,
            "warnings": ["没有位于绝缘层近场范围内的接触候选"],
        }
    ci = c.loc[candidate, "particle_i"].to_numpy(int)
    cj = c.loc[candidate, "particle_j"].to_numpy(int)
    cgap = gap[candidate]
    coverlap = c.loc[candidate, "overlap_m"].to_numpy(float)
    p = np.array(
        [
            contact_conductive_probability(
                gap_m=g,
                overlap_m=o,
                insulation_thickness_m=t_ins,
                defect_probability=p0,
            )
            for g, o in zip(cgap, coverlap)
        ]
    )
    sampled = rng.random(p.size) < p
    if not np.any(sampled):
        return {
            "i": np.empty(0, int),
            "j": np.empty(0, int),
            "conductance": np.empty(0),
            "n_candidates": int(p.size),
            "n_conductive": 0,
            "probabilities": p,
            "warnings": [
                "随机抽样未形成任何导电接触；在该缺陷率下电网络不逾渗是预期结果，"
                "并不表示数值求解失败。"
            ],
        }
    ii, jj = ci[sampled], cj[sampled]
    gg, oo = cgap[sampled], coverlap[sampled]
    # Constriction + particle spreading resistance.  For a closed defect, use a
    # small effective contact radius; for an overlapping contact use Hertz.
    r_i, r_j = radii[ii], radii[jj]
    r_star = r_i * r_j / np.maximum(r_i + r_j, 1e-30)
    a_h = np.sqrt(np.maximum(r_star * np.maximum(oo, 0.0), 0.0))
    a_c = np.maximum(a_h, 0.08 * np.minimum(r_i, r_j))
    rho = float(particle_resistivity_ohm_m)
    resistance = rho * (
        1.0 / (2.0 * r_i)
        + 1.0 / (2.0 * r_j)
        + 1.0 / (2.0 * a_c)
    )
    # Log-normal scatter represents unmodelled constriction geometry.  It is a
    # model assumption with median 1 and sigma_log = 0.6.
    resistance = resistance * np.exp(rng.normal(0.0, 0.6, size=resistance.size))
    conductance = 1.0 / np.maximum(resistance, 1e-30)
    # For transport, remove contacts wrapping across the open electrode axis.
    wraps = c.loc[candidate, f"wraps_{'xyz'[transport_axis]}"].to_numpy(bool)[sampled]
    keep = ~wraps
    return {
        "i": ii[keep],
        "j": jj[keep],
        "conductance": conductance[keep],
        "n_candidates": int(p.size),
        "n_conductive": int(np.sum(keep)),
        "n_removed_wrapped": int(np.sum(wraps)),
        "probabilities": p,
        "warnings": [],
    }


def electrical_network(
    structure: Structure,
    insulation: InsulationParameters,
    *,
    particle_resistivity_ohm_m: float | None,
    rng: np.random.Generator,
    near_gap_m: float | None = None,
    slab_fraction: float = 0.12,
) -> dict[str, Any]:
    """Estimate the effective electrical conductivity of the compact."""
    edges = build_electrical_edges(
        structure,
        insulation,
        rng=rng,
        particle_resistivity_ohm_m=particle_resistivity_ohm_m,
        near_gap_m=near_gap_m,
    )
    a, b = electrode_indices(
        structure.centers, structure.box_length_m, slab_fraction=slab_fraction
    )
    solution = solve_network(
        structure.n_particles,
        edges["i"],
        edges["j"],
        edges["conductance"],
        a,
        b,
        length_m=structure.box_length_m,
        area_m2=structure.box_length_m**2,
    )
    result = {
        "model_status": "conditional",
        "method": "sampled_contact_resistor_network",
        "percolating": solution.percolating,
        "effective_resistivity_ohm_m": solution.effective_resistivity,
        "effective_conductivity_S_m": solution.effective_conductivity,
        "effective_resistance_ohm": (
            solution.effective_resistance if np.isfinite(solution.effective_resistance) else None
        ),
        "n_active_nodes": solution.n_active_nodes,
        "n_active_edges": solution.n_active_edges,
        "n_candidate_contacts": edges["n_candidates"],
        "n_conductive_contacts": edges["n_conductive"],
        "n_removed_wrapped_contacts": edges.get("n_removed_wrapped", 0),
        "electrode_a_size": int(a.size),
        "electrode_b_size": int(b.size),
        "warnings": [
            *edges.get("warnings", []),
            *solution.warnings,
            "电接触概率为唯象模型；必须用电阻率/绝缘缺陷实验校准。",
            "x 方向周期性接触被移除以表示开放电极；y/z 仍为周期边界。",
        ],
        "assumptions": [
            "颗粒间接触电阻由收缩电阻与颗粒扩展电阻近似",
            "接触电阻对数正态散布（中位数 1，sigma_log=0.6）",
            "只有被随机缺陷/损伤模型选中的接触才导电",
        ],
    }
    if not solution.percolating:
        result["model_status"] = "needs_calibration"
    return result


# ---------------------------------------------------------------------------
# Magnetic network
# ---------------------------------------------------------------------------
def magnetic_network(
    structure: Structure,
    *,
    particle_relative_permeability: float | None,
    insulation_thickness_m: float | None,
    rng: np.random.Generator | None = None,
    near_gap_m: float | None = None,
    slab_fraction: float = 0.12,
    contact_radius_fraction: float = 0.30,
) -> dict[str, Any]:
    """DEM-informed magnetic circuit network.

    Each contact is represented by a lumped reluctance:

    ``R = R_body,i + R_body,j + t_gap / (mu0 * A_contact)``

    with an effective magnetic contact area ``pi (f r_min)^2``.  The factor
    ``f=0.30`` is a model assumption: magnetic flux spreads over a finite area
    around a mechanical contact.  It must be calibrated against permeability
    data.  A parallel pore path ``(1-phi)`` is added because the network only
    carries flux through the particle framework.

    Returns ``None`` values when the particle permeability is unknown or the
    network does not percolate.
    """
    if particle_relative_permeability is None or particle_relative_permeability <= 0:
        return {
            "model_status": "unavailable",
            "reason": "缺少颗粒本征相对磁导率，无法建立磁路网络",
            "effective_relative_permeability": None,
            "warnings": ["可在 material.initial_permeability 中补充本征磁导率或使用混合模型"],
        }
    c = structure.contacts
    if c.empty:
        return {
            "model_status": "unavailable",
            "reason": "接触表为空",
            "effective_relative_permeability": None,
            "warnings": [],
        }
    radii = structure.radii
    mean_d = float(np.mean(2.0 * radii))
    t_ins = insulation_thickness_m or 0.0
    if near_gap_m is None:
        near_gap_m = max(3.0 * t_ins, 0.02 * mean_d)
    gap = c["surface_gap_m"].to_numpy(float)
    candidate = gap <= near_gap_m
    ci = c.loc[candidate, "particle_i"].to_numpy(int)
    cj = c.loc[candidate, "particle_j"].to_numpy(int)
    cgap = np.maximum(gap[candidate], 0.0)
    r_i, r_j = radii[ci], radii[cj]
    r_min = np.minimum(r_i, r_j)
    r_star = r_i * r_j / np.maximum(r_i + r_j, 1e-30)
    overlap = c.loc[candidate, "overlap_m"].to_numpy(float)
    a_h = np.sqrt(np.maximum(r_star * np.maximum(overlap, 0.0), 0.0))
    a_eff = np.maximum(a_h, contact_radius_fraction * r_min)
    area = np.pi * a_eff**2
    mu_r = float(particle_relative_permeability)
    body_i = 2.0 / (MU0 * mu_r * math.pi * r_i)
    body_j = 2.0 / (MU0 * mu_r * math.pi * r_j)
    gap_thickness = cgap + t_ins
    gap_reluctance = gap_thickness / (MU0 * area)
    reluctance = body_i + body_j + gap_reluctance
    conductance = 1.0 / np.maximum(reluctance, 1e-30)
    wraps = c.loc[candidate, "wraps_x"].to_numpy(bool)
    keep = ~wraps
    a, b = electrode_indices(
        structure.centers, structure.box_length_m, slab_fraction=slab_fraction
    )
    solution = solve_network(
        structure.n_particles,
        ci[keep],
        cj[keep],
        conductance[keep],
        a,
        b,
        length_m=structure.box_length_m,
        area_m2=structure.box_length_m**2,
    )
    if not solution.percolating:
        return {
            "model_status": "needs_calibration",
            "effective_relative_permeability": None,
            "network_relative_permeability": None,
            "percolating": False,
            "n_active_edges": solution.n_active_edges,
            "n_candidate_contacts": int(candidate.sum()),
            "warnings": ["磁路网络不逾渗，不能给出有效磁导率"],
        }
    phi = float(
        structure.particles["volume_m3"].sum() / structure.box_length_m**3
    )
    # Flux through the particle framework / (A * dU) / mu0, plus the parallel
    # pore/matrix path (relative permeability 1).
    mu_network = solution.total_flux * structure.box_length_m / (
        structure.box_length_m**2 * MU0
    )
    mu_effective = mu_network + (1.0 - phi)
    return {
        "model_status": "conditional",
        "method": "dem_magnetic_reluctance_network",
        "percolating": True,
        "network_relative_permeability": float(mu_network),
        "effective_relative_permeability": float(mu_effective),
        "particle_relative_permeability": mu_r,
        "magnetic_volume_fraction": phi,
        "n_active_nodes": solution.n_active_nodes,
        "n_active_edges": solution.n_active_edges,
        "n_candidate_contacts": int(candidate.sum()),
        "n_removed_wrapped_contacts": int(np.sum(wraps)),
        "mean_gap_m": float(np.mean(cgap)) if cgap.size else 0.0,
        "mean_edge_reluctance_H_inv": float(np.mean(reluctance)) if reluctance.size else None,
        "electrode_a_size": int(a.size),
        "electrode_b_size": int(b.size),
        "warnings": [
            "磁路网络为线性、无局部饱和；高磁通下会高估有效磁导率。",
            "有效磁接触面积因子 f=0.30 为假设值，需要实验校准。",
            "x 方向周期接触被移除以表示开放磁极；y/z 保持周期。",
        ],
        "assumptions": [
            "颗粒内部磁化均匀，以磁阻集中参数表示",
            "接触间隙与绝缘层串联，忽略局部磁通拥挤的三维细节",
            "孔隙/非磁性相以 (1-phi) 并联路径计入",
        ],
    }


# ---------------------------------------------------------------------------
# RVE convergence
# ---------------------------------------------------------------------------
def rve_convergence(
    structure: Structure,
    *,
    fractions: Iterable[float] = (0.25, 0.35, 0.5, 0.75, 1.0),
    repeats: int = 5,
    rng: np.random.Generator | None = None,
    near_gap_m: float | None = None,
) -> dict[str, Any]:
    """Screen RVE size convergence by sub-box sampling.

    This is a geometric screening tool, not a rigorous statistical RVE proof.
    Particle volumes are assigned if the centre lies inside the sub-box; the
    resulting volume fraction can therefore exceed the global value for small
    boxes.  Compare the scatter and report the chosen RVE size explicitly.
    """
    rng = rng or np.random.default_rng(12345)
    L = structure.box_length_m
    centers = structure.centers
    volumes = (
        structure.particles["volume_m3"].to_numpy(float)
        if "volume_m3" in structure.particles
        else (4.0 / 3.0) * math.pi * structure.radii**3
    )
    gap = (
        structure.contacts["surface_gap_m"].to_numpy(float)
        if not structure.contacts.empty
        else np.empty(0)
    )
    ci = (
        structure.contacts["particle_i"].to_numpy(int)
        if not structure.contacts.empty
        else np.empty(0, int)
    )
    cj = (
        structure.contacts["particle_j"].to_numpy(int)
        if not structure.contacts.empty
        else np.empty(0, int)
    )
    if near_gap_m is None:
        near_gap_m = 0.02 * float(np.median(2.0 * structure.radii))
    keep_contact = gap <= near_gap_m
    ci, cj = ci[keep_contact], cj[keep_contact]
    results = []
    for frac in sorted(set(float(f) for f in fractions)):
        if not 0.1 <= frac <= 1.0:
            raise ValueError("RVE fraction 应位于 [0.1, 1.0]")
        Lsub = frac * L
        phis = []
        coords = []
        n_particles = []
        for _ in range(max(1, repeats)):
            origin = rng.uniform(0.0, L, size=3)
            rel = np.mod(centers - origin, L)
            mask = np.all(rel < Lsub, axis=1)
            idx = np.flatnonzero(mask)
            n_particles.append(int(idx.size))
            if idx.size == 0:
                continue
            phis.append(float(np.sum(volumes[idx]) / Lsub**3))
            selected = np.zeros(structure.n_particles, dtype=bool)
            selected[idx] = True
            both = selected[ci] & selected[cj]
            z = np.zeros(structure.n_particles, dtype=int)
            np.add.at(z, ci[both], 1)
            np.add.at(z, cj[both], 1)
            coords.append(float(np.mean(z[idx])) if idx.size else 0.0)
        results.append(
            {
                "fraction": frac,
                "subbox_length_m": Lsub,
                "n_particles_mean": float(np.mean(n_particles)) if n_particles else 0.0,
                "particle_volume_fraction_mean": float(np.mean(phis)) if phis else None,
                "particle_volume_fraction_std": float(np.std(phis, ddof=1)) if len(phis) > 1 else None,
                "coordination_mean": float(np.mean(coords)) if coords else None,
                "coordination_std": float(np.std(coords, ddof=1)) if len(coords) > 1 else None,
            }
        )
    global_phi = float(np.sum(volumes) / L**3)
    for row in results:
        phi = row["particle_volume_fraction_mean"]
        phi_std = row["particle_volume_fraction_std"]
        row["relative_phi_error"] = (
            float(phi / global_phi - 1.0) if phi is not None and global_phi > 0 else None
        )
        row["relative_phi_scatter"] = (
            float(phi_std / phi) if phi_std is not None and phi not in (None, 0) else None
        )
    recommended = None
    for row in results:
        if row["fraction"] >= 0.25 and row["relative_phi_scatter"] is not None:
            if row["relative_phi_scatter"] <= 0.05 and (
                row["relative_phi_error"] is None or abs(row["relative_phi_error"]) <= 0.03
            ):
                recommended = row["fraction"]
                break
    return {
        "model_status": "conditional",
        "global_particle_volume_fraction": global_phi,
        "near_gap_threshold_m": float(near_gap_m),
        "repeats": int(max(1, repeats)),
        "results": results,
        "recommended_fraction_from_screening": recommended,
        "recommended_rve_size_m": (
            recommended * L if recommended is not None else None
        ),
        "warnings": [
            "该 RVE 收敛性仅为几何筛选；磁性/介电性能的 RVE 收敛需单独验证。",
            "子盒按颗粒中心选入，未对边界截断体积做严格修正。",
        ],
    }
