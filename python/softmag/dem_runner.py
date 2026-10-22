"""Simplified DEM packing and compaction engine.

This module provides a **geometric prototype**, not a validated contact-dynamics
DEM.  It uses:

* random sequential adsorption for the initial configuration,
* a position-based overlap relaxation (mass-weighted corrections),
* isotropic box compression in a periodic cubic cell to a target particle
  volume fraction,
* Hertzian normal-force *post-processing* for contact-force bookkeeping.

Friction, restitution, rolling resistance, particle deformation and wall
friction are not resolved dynamically.  The output is suitable for building the
contact network and testing the multi-scale data flow, but the structure must be
validated against density, SEM/CT and coordination-number measurements before it
is used for quantitative predictions.

For production work use a validated engine (YADE, LIGGGHTS, ...) and import the
particles/contacts through :func:`load_external_structure`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .configs import DemConfig
from .io_utils import (
    ensure_dir,
    read_csv,
    read_json,
    software_versions,
    timestamp_utc,
    write_csv,
    write_json,
)
from .particle_generator import (
    ParticleSizeDistribution,
    distribution_from_config,
    size_summary,
)
from .schemas import PredictionConfig
from .units import MU0  # re-exported for convenience? noqa: F401  (kept for API stability)


@dataclass
class Structure:
    """Particle centres and contact table of one DEM realisation."""

    particles: pd.DataFrame
    contacts: pd.DataFrame
    box_length_m: float
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def centers(self) -> np.ndarray:
        return self.particles[["center_x_m", "center_y_m", "center_z_m"]].to_numpy(
            dtype=float
        )

    @property
    def radii(self) -> np.ndarray:
        return self.particles["radius_m"].to_numpy(dtype=float)

    @property
    def n_particles(self) -> int:
        return int(len(self.particles))

    def copy(self) -> "Structure":
        return Structure(
            particles=self.particles.copy(),
            contacts=self.contacts.copy(),
            box_length_m=float(self.box_length_m),
            summary=dict(self.summary),
        )


# ---------------------------------------------------------------------------
# Pair search
# ---------------------------------------------------------------------------
def find_pairs(
    centers: np.ndarray,
    radii: np.ndarray,
    box_length_m: float,
    cutoff_m: float = 0.0,
) -> dict[str, np.ndarray]:
    """Find periodic-image particle pairs with surface gap <= ``cutoff_m``.

    Returns arrays ``i, j, dvec, distance, gap``.  Neighbour lists are generated
    with a KD-tree using the minimum-image convention of a cubic periodic cell.
    """
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    n = centers.shape[0]
    empty = {
        "i": np.empty(0, dtype=int),
        "j": np.empty(0, dtype=int),
        "dvec": np.empty((0, 3), dtype=float),
        "distance": np.empty(0, dtype=float),
        "gap": np.empty(0, dtype=float),
    }
    if n < 2:
        return empty
    if box_length_m <= 0:
        raise ValueError("box_length_m 必须为正")
    rmax = float(radii.max())
    if box_length_m <= 2.0 * rmax + cutoff_m:
        # The periodic cell is too small for a well-defined contact search.
        # Fall back to an all-pairs search with minimum images.
        ii, jj = np.triu_indices(n, k=1)
    else:
        tree = cKDTree(centers, boxsize=float(box_length_m))
        pairs = tree.query_pairs(2.0 * rmax + cutoff_m, output_type="ndarray")
        if len(pairs) == 0:
            return empty
        ii, jj = pairs[:, 0], pairs[:, 1]
    dvec = centers[jj] - centers[ii]
    dvec -= box_length_m * np.round(dvec / box_length_m)
    distance = np.linalg.norm(dvec, axis=1)
    gap = distance - radii[ii] - radii[jj]
    keep = gap <= cutoff_m + 1e-15
    return {
        "i": ii[keep].astype(int),
        "j": jj[keep].astype(int),
        "dvec": dvec[keep],
        "distance": distance[keep],
        "gap": gap[keep],
    }


# ---------------------------------------------------------------------------
# Initial placement
# ---------------------------------------------------------------------------
def _try_place(
    radii: np.ndarray,
    box_length_m: float,
    rng: np.random.Generator,
    *,
    max_attempts_per_particle: int = 4000,
) -> np.ndarray | None:
    n = radii.size
    rmax = float(radii.max())
    if box_length_m <= 2.0 * rmax and n > 1:
        return None
    n_cells = max(1, int(box_length_m // (2.0 * rmax)))
    cell_size = box_length_m / n_cells
    grid: dict[tuple[int, int, int], list[int]] = {}
    centers = np.full((n, 3), np.nan)
    order = np.argsort(-radii)  # largest first: easier packing

    def cell_of(point: np.ndarray) -> tuple[int, int, int]:
        idx = np.floor(point / cell_size).astype(int) % n_cells
        return int(idx[0]), int(idx[1]), int(idx[2])

    for particle_id in order:
        r = float(radii[particle_id])
        placed = False
        for _ in range(max_attempts_per_particle):
            candidate = rng.uniform(0.0, box_length_m, size=3)
            ci, cj, ck = cell_of(candidate)
            offsets = {0} if n_cells == 1 else {-1, 0, 1}
            ok = True
            checked_cells: set[tuple[int, int, int]] = set()
            for di in offsets:
                for dj in offsets:
                    for dk in offsets:
                        key = ((ci + di) % n_cells, (cj + dj) % n_cells, (ck + dk) % n_cells)
                        if key in checked_cells:
                            continue
                        checked_cells.add(key)
                        for other in grid.get(key, ()):
                            delta = candidate - centers[other]
                            delta -= box_length_m * np.round(delta / box_length_m)
                            if np.dot(delta, delta) < (r + radii[other]) ** 2:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                centers[particle_id] = candidate
                grid.setdefault((ci, cj, ck), []).append(particle_id)
                placed = True
                break
        if not placed:
            return None
    return centers


def place_particles(
    radii: np.ndarray,
    box_length_m: float,
    rng: np.random.Generator,
    *,
    attempts: int = 5,
) -> tuple[np.ndarray, float]:
    """Place non-overlapping spheres, enlarging the box if necessary.

    The returned box length may exceed the requested value; the caller should
    report the actual initial volume fraction.
    """
    L = float(box_length_m)
    for attempt in range(attempts):
        centers = _try_place(radii, L, rng)
        if centers is not None:
            return centers, L
        L *= 1.06
    raise RuntimeError(
        "无法在周期盒子中完成无重叠初始放置。请降低 initial_particle_volume_fraction "
        "或减少 n_particles。"
    )


# ---------------------------------------------------------------------------
# Relaxation
# ---------------------------------------------------------------------------
def relax_overlaps(
    centers: np.ndarray,
    radii: np.ndarray,
    box_length_m: float,
    *,
    max_iterations: int = 600,
    relaxation_factor: float = 0.55,
    tolerance: float = 2.0e-3,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve overlaps with mass-weighted position corrections.

    ``tolerance`` is the maximum allowed overlap normalised by the sum of the two
    radii.  The algorithm does not introduce velocities; it is a static
    geometric relaxation suitable for structure generation only.
    """
    centers = np.asarray(centers, dtype=float).copy()
    radii = np.asarray(radii, dtype=float)
    L = float(box_length_m)
    n = centers.shape[0]
    if n == 0:
        return centers, {"converged": True, "iterations": 0, "max_overlap_rel": 0.0}
    rng = rng or np.random.default_rng(0)
    relax = float(relaxation_factor)
    best = float("inf")
    stall = 0
    history: list[float] = []
    converged = False
    max_rel = float("inf")
    rms_rel = float("inf")

    for iteration in range(1, max_iterations + 1):
        pairs = find_pairs(centers, radii, L, cutoff_m=0.0)
        i, j = pairs["i"], pairs["j"]
        if i.size == 0:
            converged = True
            max_rel = 0.0
            rms_rel = 0.0
            break
        gap = pairs["gap"]
        overlap = np.maximum(-gap, 0.0)
        active = overlap > 1e-15
        if not np.any(active):
            converged = True
            max_rel = 0.0
            rms_rel = 0.0
            break
        i, j = i[active], j[active]
        dvec = pairs["dvec"][active]
        distance = np.maximum(pairs["distance"][active], 1e-30)
        overlap = overlap[active]
        normal = dvec / distance[:, None]
        ri = radii[i]
        rj = radii[j]
        denom = ri**3 + rj**3
        move_i = relax * overlap * (rj**3 / denom)
        move_j = relax * overlap * (ri**3 / denom)
        disp = np.zeros_like(centers)
        contrib = -normal * move_i[:, None]
        contrib_j = normal * move_j[:, None]
        np.add.at(disp, i, contrib)
        np.add.at(disp, j, contrib_j)
        # Cap per-particle displacement (prevents tunnelling in dense regions).
        norms = np.linalg.norm(disp, axis=1)
        cap = 0.20 * np.maximum(radii, 1e-30)
        scale = np.ones_like(norms)
        too_big = norms > cap
        scale[too_big] = cap[too_big] / norms[too_big]
        centers += disp * scale[:, None]
        centers = np.mod(centers, L)

        rel = overlap / np.maximum(ri + rj, 1e-30)
        max_rel = float(np.max(rel))
        rms_rel = float(np.sqrt(np.mean(rel**2)))
        history.append(max_rel)
        if max_rel <= tolerance:
            converged = True
            break
        if max_rel < best * 0.995:
            best = max_rel
            stall = 0
        else:
            stall += 1
            if stall >= 30:
                relax *= 0.8
                stall = 0
                if relax < 0.05:
                    break

    pairs = find_pairs(centers, radii, L, cutoff_m=0.0)
    if pairs["i"].size:
        rel = np.maximum(-pairs["gap"], 0.0) / np.maximum(
            radii[pairs["i"]] + radii[pairs["j"]], 1e-30
        )
        max_rel = float(np.max(rel))
        rms_rel = float(np.sqrt(np.mean(rel**2)))
        converged = max_rel <= tolerance
    else:
        max_rel = rms_rel = 0.0
        converged = True
    return centers, {
        "converged": bool(converged),
        "iterations": iteration,
        "max_overlap_rel": max_rel,
        "rms_overlap_rel": rms_rel,
        "final_relaxation_factor": relax,
        "history": history[-50:],
    }


# ---------------------------------------------------------------------------
# Contact bookkeeping
# ---------------------------------------------------------------------------
def build_contacts(
    centers: np.ndarray,
    radii: np.ndarray,
    box_length_m: float,
    *,
    gap_tolerance_m: float,
    youngs_modulus_Pa: float,
    poisson_ratio: float,
) -> pd.DataFrame:
    """Build the contact table, including near contacts within a gap tolerance."""
    pairs = find_pairs(centers, radii, box_length_m, cutoff_m=gap_tolerance_m)
    i, j = pairs["i"], pairs["j"]
    if i.size == 0:
        return pd.DataFrame(
            columns=[
                "contact_id",
                "particle_i",
                "particle_j",
                "center_distance_m",
                "surface_gap_m",
                "overlap_m",
                "overlap_ratio",
                "normal_x",
                "normal_y",
                "normal_z",
                "contact_area_m2",
                "normal_force_N",
                "is_mechanical_contact",
                "wraps_x",
                "wraps_y",
                "wraps_z",
            ]
        )
    dvec = pairs["dvec"]
    distance = np.maximum(pairs["distance"], 1e-30)
    gap = pairs["gap"]
    overlap = np.maximum(-gap, 0.0)
    normal = dvec / distance[:, None]
    wraps = np.abs(np.round(dvec / box_length_m)) > 0
    # Hertzian normal force: F = 4/3 E* sqrt(R*) delta^(3/2)
    e_star = youngs_modulus_Pa / (2.0 * (1.0 - poisson_ratio**2))
    r_star = radii[i] * radii[j] / np.maximum(radii[i] + radii[j], 1e-30)
    force = np.where(
        overlap > 0,
        (4.0 / 3.0) * e_star * np.sqrt(r_star) * overlap**1.5,
        0.0,
    )
    contact_radius = np.where(overlap > 0, np.sqrt(np.maximum(r_star * overlap, 0.0)), 0.0)
    area = np.pi * contact_radius**2
    return pd.DataFrame(
        {
            "contact_id": np.arange(i.size, dtype=int),
            "particle_i": i,
            "particle_j": j,
            "center_distance_m": distance,
            "surface_gap_m": gap,
            "overlap_m": overlap,
            "overlap_ratio": overlap / np.maximum(radii[i] + radii[j], 1e-30),
            "normal_x": normal[:, 0],
            "normal_y": normal[:, 1],
            "normal_z": normal[:, 2],
            "contact_area_m2": area,
            "normal_force_N": force,
            "is_mechanical_contact": gap <= 1e-15,
            "wraps_x": wraps[:, 0],
            "wraps_y": wraps[:, 1],
            "wraps_z": wraps[:, 2],
        }
    )


def _coordination(
    n: int, contacts: pd.DataFrame, *, mechanical_only: bool = True
) -> np.ndarray:
    z = np.zeros(n, dtype=int)
    if contacts.empty:
        return z
    mask = (
        contacts["is_mechanical_contact"].to_numpy(dtype=bool)
        if mechanical_only
        else np.ones(len(contacts), dtype=bool)
    )
    ii = contacts.loc[mask, "particle_i"].to_numpy(dtype=int)
    jj = contacts.loc[mask, "particle_j"].to_numpy(dtype=int)
    np.add.at(z, ii, 1)
    np.add.at(z, jj, 1)
    return z


def _virial_pressure(contacts: pd.DataFrame, box_length_m: float) -> float:
    if contacts.empty:
        return 0.0
    mask = contacts["is_mechanical_contact"].to_numpy(dtype=bool)
    if not np.any(mask):
        return 0.0
    force = contacts.loc[mask, "normal_force_N"].to_numpy(dtype=float)
    distance = contacts.loc[mask, "center_distance_m"].to_numpy(dtype=float)
    volume = box_length_m**3
    return float(np.sum(force * distance) / (3.0 * volume))


# ---------------------------------------------------------------------------
# Packing workflow
# ---------------------------------------------------------------------------
def generate_packing(
    diameters_m: np.ndarray,
    dem: DemConfig,
    *,
    target_particle_volume_fraction: float,
    random_seed: int | None = None,
) -> Structure:
    """Generate and compact a periodic sphere packing."""
    diameters = np.asarray(diameters_m, dtype=float)
    if diameters.ndim != 1 or diameters.size < 1:
        raise ValueError("diameters_m 必须是一维非空数组")
    if np.any(diameters <= 0):
        raise ValueError("粒径必须为正")
    radii = 0.5 * diameters
    volumes = (4.0 / 3.0) * math.pi * radii**3
    solid_volume = float(volumes.sum())
    phi_target = float(target_particle_volume_fraction)
    if not 0.02 <= phi_target <= 0.72:
        raise ValueError("目标颗粒体积分数应位于 0.02 .. 0.72")
    phi_initial = min(dem.initial_particle_volume_fraction, phi_target * 0.9)
    L = (solid_volume / phi_initial) ** (1.0 / 3.0)
    seed = int(random_seed if random_seed is not None else dem.random_seed)
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    centers, L = place_particles(radii, L, rng)
    centers = np.mod(centers, L)
    centers, relax_info = relax_overlaps(
        centers,
        radii,
        L,
        max_iterations=dem.max_relax_iterations,
        relaxation_factor=dem.relaxation_factor,
        tolerance=dem.overlap_tolerance_rel,
        rng=rng,
    )
    initial_phi = solid_volume / L**3

    L_target = (solid_volume / phi_target) ** (1.0 / 3.0)
    best_centers = centers.copy()
    best_L = L
    best_info = relax_info
    compaction_steps = 0
    converged_to_target = initial_phi >= phi_target * 0.999
    warnings: list[str] = []

    # Compress directly to the target box and relax.  Incremental box scaling
    # is physically more gradual, but for a static geometric packing it mostly
    # multiplies the number of contact searches.  If the direct compression
    # jams, back off in small density steps until a feasible overlap is found.
    if not converged_to_target:
        max_attempts = max(1, dem.max_compaction_steps)
        for attempt in range(max_attempts):
            compaction_steps = attempt + 1
            L_trial = L_target * (1.0 + 0.005 * attempt)
            trial_centers = np.mod(centers * (L_trial / L), L_trial)
            trial_centers, trial_info = relax_overlaps(
                trial_centers,
                radii,
                L_trial,
                max_iterations=dem.max_relax_iterations,
                relaxation_factor=dem.relaxation_factor,
                tolerance=dem.overlap_tolerance_rel,
                rng=rng,
            )
            jam_limit = max(5.0 * dem.overlap_tolerance_rel, 1.0e-2)
            best_centers, best_L, best_info = (
                trial_centers.copy(),
                L_trial,
                trial_info,
            )
            if trial_info["max_overlap_rel"] <= jam_limit:
                converged_to_target = (
                    solid_volume / L_trial**3 >= phi_target * 0.995
                )
                break
            warnings.append(
                "直接压缩尝试未能达到重叠容差；已回退到更低密度重试。"
                f" 尝试 L={L_trial:.6e} m，最大相对重叠="
                f"{trial_info['max_overlap_rel']:.4g}。"
            )
        else:
            warnings.append(
                "所有压缩尝试均未消除残余重叠；导出的结构为最后一个尝试，"
                "不可用于定量接触力分析。"
            )

    centers, L = best_centers, best_L
    achieved_phi = solid_volume / L**3
    gap_tol = dem.contact_gap_tolerance_rel * float(np.mean(diameters))
    contacts = build_contacts(
        centers,
        radii,
        L,
        gap_tolerance_m=gap_tol,
        youngs_modulus_Pa=dem.youngs_modulus_Pa,
        poisson_ratio=dem.poisson_ratio,
    )
    coordination = _coordination(radii.size, contacts, mechanical_only=True)
    near_coordination = _coordination(radii.size, contacts, mechanical_only=False)
    pressure = _virial_pressure(contacts, L)
    n_mech = int(contacts["is_mechanical_contact"].sum()) if not contacts.empty else 0

    if not converged_to_target or achieved_phi < phi_target * 0.995:
        warnings.append(
            f"目标颗粒体积分数 {phi_target:.4f} 未完全达到，实际 {achieved_phi:.4f}。"
            "简化几何压实不能代表真实压制变形；请使用经过验证的 DEM 或放宽目标。"
        )
    if phi_target > 0.64:
        warnings.append(
            "目标体积分数高于单分散球体随机密堆极限（约 0.64）；"
            "多分散或塑性变形可能达到，但本简化模型不能保证。"
        )
    if not best_info.get("converged", False):
        warnings.append(
            "重叠松弛未达到容差；结构包含残余几何重叠，接触力估计不可靠。"
        )
    warnings.append(
        "简化 DEM：无摩擦/滚动阻力/真实接触动力学；接触力仅用于结构诊断。"
    )

    particles = pd.DataFrame(
        {
            "particle_id": np.arange(radii.size, dtype=int),
            "center_x_m": centers[:, 0],
            "center_y_m": centers[:, 1],
            "center_z_m": centers[:, 2],
            "radius_m": radii,
            "diameter_m": diameters,
            "volume_m3": volumes,
        }
    )
    summary = {
        "schema_version": "1.0",
        "created_utc": timestamp_utc(),
        "engine": "simplified_position_based",
        "engine_status": "prototype_not_validated_against_experiment",
        "boundary": dem.boundary,
        "random_seed": seed,
        "n_particles": int(radii.size),
        "box_length_m": float(L),
        "solid_volume_m3": solid_volume,
        "cell_volume_m3": float(L**3),
        "target_particle_volume_fraction": phi_target,
        "initial_particle_volume_fraction": float(initial_phi),
        "achieved_particle_volume_fraction": float(achieved_phi),
        "porosity_fraction": float(1.0 - achieved_phi),
        "target_reached": bool(converged_to_target and achieved_phi >= phi_target * 0.995),
        "compaction_steps": int(compaction_steps),
        "relaxation": best_info,
        "n_contacts": int(len(contacts)),
        "n_mechanical_contacts": n_mech,
        "n_near_contacts": int(len(contacts) - n_mech),
        "coordination_number_mean": float(coordination.mean()),
        "coordination_number_std": float(coordination.std(ddof=1)) if radii.size > 1 else 0.0,
        "coordination_number_max": int(coordination.max()) if radii.size else 0,
        "near_coordination_mean": float(near_coordination.mean()),
        "pressure_Pa_virial_proxy": pressure,
        "target_pressure_Pa": dem.target_pressure_Pa,
        "youngs_modulus_Pa": dem.youngs_modulus_Pa,
        "poisson_ratio": dem.poisson_ratio,
        "mean_diameter_m": float(np.mean(diameters)),
        "volume_weighted_mean_diameter_m": float(
            (np.sum(volumes * diameters) / np.sum(volumes))
        ),
        "warnings": warnings,
        "assumptions": [
            "刚性球体、周期边界、各向同性盒子压缩",
            "机械接触与电接触分离；接触表不默认导通",
            "接触力由 Hertz 法向公式事后计算，未参与松弛",
            "颗粒形状以体积等效球处理，形状信息仅用于损失/退磁修正",
        ],
        "software": software_versions(),
        "wall_time_s": float(time.perf_counter() - t0),
    }
    return Structure(particles=particles, contacts=contacts, box_length_m=L, summary=summary)


# ---------------------------------------------------------------------------
# External DEM import
# ---------------------------------------------------------------------------
def load_external_structure(
    particles_csv: str | Path,
    contacts_csv: str | Path | None = None,
    *,
    box_length_m: float | None = None,
    gap_tolerance_m: float | None = None,
) -> Structure:
    """Import a validated DEM structure from CSV.

    Required particle columns (aliases supported):
    ``particle_id``, ``center_x_m``, ``center_y_m``, ``center_z_m``,
    ``radius_m`` (or ``diameter_m``).

    Required contact columns when supplied: ``particle_i``, ``particle_j``,
    ``surface_gap_m``.  Optional: ``normal_force_N``, ``normal_x``, ``normal_y``,
    ``normal_z``, ``contact_area_m2``, ``is_mechanical_contact``.
    """
    df = read_csv(particles_csv)
    aliases = {
        "id": "particle_id",
        "x_m": "center_x_m",
        "y_m": "center_y_m",
        "z_m": "center_z_m",
        "diameter_m": "diameter_m",
        "radius": "radius_m",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})
    required = ["center_x_m", "center_y_m", "center_z_m"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"外部颗粒表缺少列：{missing}")
    if "radius_m" not in df.columns:
        if "diameter_m" not in df.columns:
            raise ValueError("外部颗粒表需要 radius_m 或 diameter_m")
        df["radius_m"] = df["diameter_m"] / 2.0
    if "diameter_m" not in df.columns:
        df["diameter_m"] = 2.0 * df["radius_m"]
    if "volume_m3" not in df.columns:
        df["volume_m3"] = (4.0 / 3.0) * math.pi * df["radius_m"] ** 3
    if "particle_id" not in df.columns:
        df["particle_id"] = np.arange(len(df), dtype=int)
    if box_length_m is None:
        raise ValueError("外部结构必须显式给出 box_length_m（来自 DEM 盒子或 RVE 尺寸）")
    centers = df[["center_x_m", "center_y_m", "center_z_m"]].to_numpy(float)
    radii = df["radius_m"].to_numpy(float)
    if contacts_csv:
        contacts = read_csv(contacts_csv)
    else:
        tol = gap_tolerance_m or 0.05 * float(np.mean(2 * radii))
        contacts = build_contacts(
            centers,
            radii,
            float(box_length_m),
            gap_tolerance_m=tol,
            youngs_modulus_Pa=2.0e11,
            poisson_ratio=0.3,
        )
    for col in ("normal_x", "normal_y", "normal_z"):
        if col not in contacts.columns:
            contacts[col] = np.nan
    if "is_mechanical_contact" not in contacts.columns:
        contacts["is_mechanical_contact"] = contacts["surface_gap_m"] <= 1e-15
    if "normal_force_N" not in contacts.columns:
        contacts["normal_force_N"] = 0.0
    summary = {
        "engine": "external_import",
        "engine_status": "external_structure_assumed_validated_by_user",
        "n_particles": int(len(df)),
        "n_contacts": int(len(contacts)),
        "box_length_m": float(box_length_m),
        "particle_volume_fraction": float(
            df["volume_m3"].sum() / float(box_length_m) ** 3
        ),
        "warnings": [
            "外部结构按原样使用；本项目未验证其接触模型、密度或周期性。"
        ],
        "software": software_versions(),
        "created_utc": timestamp_utc(),
    }
    return Structure(df, contacts, float(box_length_m), summary)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_structure(structure: Structure, outdir: str | Path, *, tag: str = "") -> dict[str, Path]:
    out = ensure_dir(outdir)
    suffix = f"_{tag}" if tag else ""
    paths = {
        "particles": write_csv(structure.particles, out / f"particles{suffix}.csv"),
        "contacts": write_csv(structure.contacts, out / f"contacts{suffix}.csv"),
        "summary": write_json(out / f"summary{suffix}.json", structure.summary),
        "arrays": out / f"structure{suffix}.npz",
    }
    np.savez_compressed(
        paths["arrays"],
        centers=structure.centers,
        radii=structure.radii,
        box_length_m=np.array([structure.box_length_m]),
    )
    return paths


def load_structure(dem_dir: str | Path) -> Structure:
    base = Path(dem_dir)
    particles_path = base / "particles.csv"
    contacts_path = base / "contacts.csv"
    summary_path = base / "summary.json"
    if not particles_path.is_file():
        raise FileNotFoundError(f"找不到 DEM 颗粒表：{particles_path}")
    particles = read_csv(particles_path)
    if contacts_path.is_file():
        contacts = read_csv(contacts_path)
    else:
        contacts = pd.DataFrame()
    summary = read_json(summary_path) if summary_path.is_file() else {}
    box = summary.get("box_length_m")
    if box is None:
        npz_path = base / "structure.npz"
        if npz_path.is_file():
            with np.load(npz_path) as data:
                box = float(np.asarray(data["box_length_m"]).reshape(-1)[0])
    if box is None:
        raise ValueError("DEM summary 缺少 box_length_m，无法恢复周期结构")
    return Structure(particles, contacts, float(box), summary)


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------
def run_dem(
    prediction: PredictionConfig,
    dem: DemConfig,
    outdir: str | Path,
    *,
    distribution: ParticleSizeDistribution | None = None,
) -> Structure:
    """Generate the DEM structure requested by the configuration."""
    if dem.engine == "external":
        structure = load_external_structure(
            dem.external_particles_csv,
            dem.external_contacts_csv,
            box_length_m=prediction.microstructure.rve_size_m,
            gap_tolerance_m=(
                dem.contact_gap_tolerance_rel
                * (prediction.particle.d50_m or 1e-5)
            ),
        )
        if dem.external_particles_csv is None:
            raise ValueError("external 模式需要 external_particles_csv")
        save_structure(structure, outdir)
        return structure

    dist = distribution or distribution_from_config(prediction.particle)
    target_phi, source, target_warnings = prediction.target_particle_volume_fraction()
    if dem.target_particle_volume_fraction is not None:
        target_phi = dem.target_particle_volume_fraction
        source = "dem.target_particle_volume_fraction"
        target_warnings = []
    if target_phi is None:
        raise ValueError(
            "无法确定 DEM 目标体积分数：请设置 dem.target_particle_volume_fraction、"
            "microstructure.magnetic_volume_fraction 或 compaction.target_relative_density"
        )
    rng = np.random.default_rng(dem.random_seed)
    diameters = dist.sample(dem.n_particles, rng, weighting="number")
    structure = generate_packing(
        diameters,
        dem,
        target_particle_volume_fraction=float(target_phi),
        random_seed=dem.random_seed,
    )
    structure.summary["target_source"] = source
    structure.summary["target_warnings"] = target_warnings
    structure.summary["particle_size_summary"] = size_summary(dist, diameters)
    structure.summary["particle_size_summary"]["sample"] = {
        "n": int(diameters.size),
        "mean_m": float(np.mean(diameters)),
        "volume_weighted_mean_m": float(
            (np.sum(diameters**3) / diameters.size) ** (1.0 / 3.0)
        ),
    }
    structure.summary["warnings"].extend(target_warnings)
    save_structure(structure, outdir)
    # Save the exact sampled diameters for reproducibility of this realisation.
    write_csv(
        pd.DataFrame(
            {
                "particle_id": np.arange(diameters.size),
                "diameter_m": diameters,
            }
        ),
        Path(outdir) / "sampled_diameters.csv",
    )
    write_json(
        Path(outdir) / "dem_input.json",
        {
            "dem": dem.model_dump(mode="json"),
            "target_particle_volume_fraction": target_phi,
            "target_source": source,
            "particle_size_summary": size_summary(dist, diameters),
        },
    )
    return structure
