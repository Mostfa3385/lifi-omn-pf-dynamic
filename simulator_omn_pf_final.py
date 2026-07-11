"""Paper-level dynamic OMN-PF simulator for dense LiFi attocells.

This version is designed for manuscript experiments rather than only smoke
testing. It preserves the static paper's protocol-level spirit while adding
time evolution, target-user interference, beam-pattern gains, first-order NLOS
reflections, baseline schedulers, and OMN-PF ablations.

The simulator is intentionally deterministic for a given seed and writes
plain Python dictionaries so experiment drivers can create CSV tables and
figures without depending on a notebook.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


# ----------------------------- System Parameters -----------------------------

ROOM_X = 16.0
ROOM_Y = 16.0
TX_HEIGHT = 3.0
RX_HEIGHT = 0.85
HF = TX_HEIGHT - RX_HEIGHT

R_MICRO = 3.3
DELTA = 2.0 / 3.0
K = 512
N_DATA = K // 2 - 1
N_CENTER = int(math.floor(DELTA * DELTA * N_DATA))
N_EDGE = (N_DATA - N_CENTER) // 3

PHI_HALF = math.radians(60.0)
LAMBERT_M = -math.log(2.0) / math.log(math.cos(PHI_HALF))
RESPONSIVITY = 0.4
OPTICAL_FILTER = 0.95
ETA_BASE = RESPONSIVITY * OPTICAL_FILTER
N_REFR = 1.5
A_PD = 5e-5
BANDWIDTH_HZ = 5e6
N0 = 1e-21
NOISE_SUB = (N0 * BANDWIDTH_HZ) / N_DATA
DEN_EPS = 1e-30
RATE_EPS = 1e-9

LED_EFFICIENCY = 0.3
PT_ELECTRICAL = 3.0
P_OPTICAL_MACRO = LED_EFFICIENCY * PT_ELECTRICAL
P_EDGE_BEAM = 0.7 * P_OPTICAL_MACRO / 7.0
P_CENTER_BEAM = P_OPTICAL_MACRO - 6.0 * P_EDGE_BEAM

RX_FOV_DEG = 50.0
EDGE_BEAM_SIGMA_DEG = 24.0
CENTER_BEAM_SIGMA_M = 3.0
SINR_THRESHOLDS_DB = (10.0, 20.0)
# Service-regularity threshold used for starvation analysis. A user-trial
# history is counted as starved if the user is not scheduled for this many
# consecutive frames or more.
SERVICE_STARVATION_GAP_FRAMES = 10
# Scheduler-side service-gap awareness. OMN-PF uses the current number of
# consecutive frames since each user was last scheduled to reduce prolonged
# neglect/starvation under opportunistic channel selection.
SERVICE_GAP_NORMALIZATION_FRAMES = 10.0

# Step-4/5 NLOS-stress and edge-reliability controls. The baseline profile leaves the Step-3 model
# unchanged; the stress profile increases first-order reflection relevance and
# weakens LOS links for tilted/edge users so the NLOS-awareness term can be
# tested under reflection-dependent links.
NLOS_PROFILE = "baseline"
NLOS_STRESS_SCALE = 3.0
LOS_STRESS_TILT_DECAY_DEG = 28.0
LOS_STRESS_EDGE_DECAY = 0.65


@dataclass(frozen=True)
class ResourceState:
    delta: float
    n_center: int
    n_edge: int


def make_resource_state(delta: float) -> ResourceState:
    n_center = int(math.floor(delta * delta * N_DATA))
    n_edge = max(1, (N_DATA - n_center) // 3)
    return ResourceState(delta=float(delta), n_center=n_center, n_edge=n_edge)


STATIC_RESOURCE_STATE = make_resource_state(DELTA)
ADAPTIVE_DELTA_CANDIDATES = (0.5, 2.0 / 3.0, 0.8)
# Common dynamic multi-user OFDMA admission controls. All frame-level
# schedulers use the same resource- and SINR-aware admission rule so scheduler
# comparisons are not confounded by a scheduler-specific hard user cap.
#
# A region may admit a variable number of users. The resource ceiling is
# derived from the current center/edge pool size and the minimum number of
# subcarriers required per admitted user. Additional users are accepted only
# when their predicted post-sharing SINR remains above the admission threshold
# and the region retains a conservative fraction of its single-user aggregate
# rate. OMN-PF-A differs only in the OFDMA/power allocation applied after this
# common admission step.
DYNAMIC_MIN_SUBCARRIERS_PER_USER = 16
DYNAMIC_ADMISSION_MIN_SINR_DB = 10.0
DYNAMIC_ADMISSION_MIN_SUM_RATE_RETENTION = 0.55
DYNAMIC_MAX_CENTER_USERS_PER_POOL = max(1, N_CENTER // DYNAMIC_MIN_SUBCARRIERS_PER_USER)
DYNAMIC_MAX_EDGE_USERS_PER_POOL = max(1, N_EDGE // DYNAMIC_MIN_SUBCARRIERS_PER_USER)
DYNAMIC_MAX_USERS_PER_POOL = max(
    DYNAMIC_MAX_CENTER_USERS_PER_POOL,
    DYNAMIC_MAX_EDGE_USERS_PER_POOL,
)
DYNAMIC_MAX_USERS_PER_REGION = (
    DYNAMIC_MAX_CENTER_USERS_PER_POOL + DYNAMIC_MAX_EDGE_USERS_PER_POOL
)
# Backward-compatible aliases used by older reports/scripts.
ADAPTIVE_MIN_SUBCARRIERS_PER_USER = DYNAMIC_MIN_SUBCARRIERS_PER_USER
ADAPTIVE_MAX_USERS_PER_POOL = DYNAMIC_MAX_USERS_PER_POOL
ADAPTIVE_MAX_USERS_PER_REGION = DYNAMIC_MAX_USERS_PER_REGION

@dataclass(frozen=True)
class Region:
    idx: int
    center: np.ndarray
    beam_az_deg: float
    edge_band: int


@dataclass
class User:
    uid: int
    pos: np.ndarray
    waypoint: np.ndarray
    speed: float
    theta_deg: float
    phi_deg: float
    avg_rate: float = 1.0
    last_region: int = -1
    outage_memory: float = 0.0
    service_gap_frames: int = 0


@dataclass(frozen=True)
class Patch:
    pos: np.ndarray
    normal: np.ndarray
    rho: float
    area: float


@dataclass(frozen=True)
class SchedulerConfig:
    name: str
    use_outage: bool = True
    use_edge: bool = True
    use_transition: bool = True
    use_nlos: bool = True
    use_fairness_history: bool = True
    use_service_gap: bool = True
    # Step 7 service-regularity tuning: service-gap awareness is used as an
    # explicit anti-starvation term. It increases priority for users that have
    # not been scheduled for several consecutive frames, complementing PF
    # history with a direct waiting-time signal.
    lambda_gap: float = 6.0
    # Step 6 reliability-tuned default: lower PF memory dominance and stronger edge priority,
    # with an explicit NLOS-fragility term retained. This version targets edge-user
    # service continuity rather than raw scheduled-link rate.
    pf_alpha: float = 0.20
    lambda_outage: float = 2.0
    lambda_edge: float = 14.0
    lambda_transition: float = 0.5
    lambda_nlos: float = 8.0


SCHEDULERS = {
    "full-load": SchedulerConfig("full-load"),
    "rr1": SchedulerConfig("rr1"),
    "max_sinr": SchedulerConfig("max_sinr"),
    "proportional_fair": SchedulerConfig("proportional_fair", pf_alpha=1.0),
    "omn_pf": SchedulerConfig("omn_pf"),
    "omn_pf_a": SchedulerConfig("omn_pf_a"),
    "omn_no_outage": SchedulerConfig("omn_no_outage", use_outage=False),
    "omn_no_edge": SchedulerConfig("omn_no_edge", use_edge=False),
    "omn_no_transition": SchedulerConfig("omn_no_transition", use_transition=False),
    "omn_no_nlos": SchedulerConfig("omn_no_nlos", use_nlos=False),
    "omn_no_fairness": SchedulerConfig("omn_no_fairness", use_fairness_history=False),
    "omn_no_gap": SchedulerConfig("omn_no_gap", use_service_gap=False),
}


def angle_diff_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def concentrator_gain(psi: float, psi_c: float) -> float:
    if 0.0 <= psi <= psi_c:
        return (N_REFR * N_REFR) / (math.sin(psi_c) ** 2 + DEN_EPS)
    return 0.0


def rx_normal(theta_deg: float, phi_deg: float) -> np.ndarray:
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)
    return np.array(
        [
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ],
        dtype=float,
    )


def build_regions() -> list[Region]:
    tx = np.array([ROOM_X / 2.0, ROOM_Y / 2.0], dtype=float)
    dist = 2.0 * R_MICRO * math.cos(math.pi / 6.0)
    regions = [Region(0, tx, 0.0, 0)]
    edge_bands = [0, 1, 2, 0, 1, 2]
    for rid, az_deg in enumerate(np.linspace(0.0, 360.0, 6, endpoint=False), start=1):
        az = math.radians(float(az_deg))
        center = tx + dist * np.array([math.cos(az), math.sin(az)])
        regions.append(Region(rid, center, float(az_deg), edge_bands[rid - 1]))
    return regions


def make_patches(wall_patches_per_wall: int = 2, floor_grid: int = 2) -> list[Patch]:
    patches: list[Patch] = []
    wall_area_x = (ROOM_X / wall_patches_per_wall) * (TX_HEIGHT - RX_HEIGHT)
    wall_area_y = (ROOM_Y / wall_patches_per_wall) * (TX_HEIGHT - RX_HEIGHT)
    z = 0.5 * (RX_HEIGHT + TX_HEIGHT)
    for i in range(wall_patches_per_wall):
        x = (i + 0.5) * ROOM_X / wall_patches_per_wall
        y = (i + 0.5) * ROOM_Y / wall_patches_per_wall
        patches.extend(
            [
                Patch(np.array([x, 0.0, z]), np.array([0.0, 1.0, 0.0]), 0.72, wall_area_x),
                Patch(np.array([x, ROOM_Y, z]), np.array([0.0, -1.0, 0.0]), 0.72, wall_area_x),
                Patch(np.array([0.0, y, z]), np.array([1.0, 0.0, 0.0]), 0.72, wall_area_y),
                Patch(np.array([ROOM_X, y, z]), np.array([-1.0, 0.0, 0.0]), 0.72, wall_area_y),
            ]
        )

    floor_area = (ROOM_X / floor_grid) * (ROOM_Y / floor_grid)
    for ix in range(floor_grid):
        for iy in range(floor_grid):
            patches.append(
                Patch(
                    np.array([(ix + 0.5) * ROOM_X / floor_grid, (iy + 0.5) * ROOM_Y / floor_grid, RX_HEIGHT]),
                    np.array([0.0, 0.0, 1.0]),
                    0.32,
                    floor_area,
                )
            )
    return patches


def clip_room(point: np.ndarray) -> np.ndarray:
    return np.array([np.clip(point[0], 0.2, ROOM_X - 0.2), np.clip(point[1], 0.2, ROOM_Y - 0.2)], dtype=float)


def sample_in_region(region: Region, rng: np.random.Generator) -> np.ndarray:
    r = R_MICRO * math.sqrt(float(rng.random()))
    th = 2.0 * math.pi * float(rng.random())
    return clip_room(region.center + r * np.array([math.cos(th), math.sin(th)]))


def random_waypoint(rng: np.random.Generator) -> np.ndarray:
    return np.array([rng.uniform(0.2, ROOM_X - 0.2), rng.uniform(0.2, ROOM_Y - 0.2)], dtype=float)


def initialize_users(users_per_macro: int, regions: list[Region], rng: np.random.Generator) -> list[User]:
    users: list[User] = []
    per_region = users_per_macro // len(regions)
    extras = users_per_macro % len(regions)
    uid = 0
    for region in regions:
        count = per_region + (1 if region.idx < extras else 0)
        for _ in range(count):
            users.append(
                User(
                    uid=uid,
                    pos=sample_in_region(region, rng),
                    waypoint=random_waypoint(rng),
                    speed=float(rng.uniform(0.2, 1.5)),
                    theta_deg=float(np.clip(rng.normal(15.0, 4.0), 0.0, 80.0)),
                    phi_deg=float(rng.uniform(0.0, 360.0)),
                )
            )
            uid += 1
    return users


def update_positions(users: list[User], dt: float, rng: np.random.Generator) -> None:
    for user in users:
        step = user.waypoint - user.pos
        dist = float(np.linalg.norm(step))
        if dist <= max(0.15, user.speed * dt):
            user.waypoint = random_waypoint(rng)
            user.speed = float(rng.uniform(0.2, 1.5))
            continue
        user.pos = clip_room(user.pos + (step / dist) * user.speed * dt)


def update_orientation(
    users: list[User],
    rng: np.random.Generator,
    rho: float = 0.85,
    theta_mean_deg: float = 15.0,
    tilt_sigma_deg: float = 5.0,
    azimuth_sigma_deg: float = 10.0,
) -> None:
    for user in users:
        theta = rho * user.theta_deg + (1.0 - rho) * theta_mean_deg + rng.normal(0.0, tilt_sigma_deg)
        user.theta_deg = float(np.clip(theta, 0.0, 80.0))
        user.phi_deg = float((user.phi_deg + rng.normal(0.0, azimuth_sigma_deg)) % 360.0)


def serving_region(user: User, regions: list[Region]) -> Region:
    return min(regions, key=lambda region: float(np.linalg.norm(user.pos - region.center)))


def is_edge_user(user: User, region: Region, resource_state: ResourceState = STATIC_RESOURCE_STATE) -> bool:
    return float(np.linalg.norm(user.pos - region.center)) > resource_state.delta * R_MICRO


def beam_pattern_gain(region: Region, target_xy: np.ndarray) -> float:
    tx = np.array([ROOM_X / 2.0, ROOM_Y / 2.0], dtype=float)
    vector = target_xy - tx
    d_h = float(np.linalg.norm(vector))
    if region.idx == 0:
        return float(math.exp(-0.5 * (d_h / CENTER_BEAM_SIGMA_M) ** 2))
    az = math.degrees(math.atan2(vector[1], vector[0]))
    if az < 0.0:
        az += 360.0
    diff = angle_diff_deg(az, region.beam_az_deg)
    return float(math.exp(-0.5 * (diff / EDGE_BEAM_SIGMA_DEG) ** 2))


def los_gain(region: Region, user: User, psi_c: float) -> float:
    tx_xyz = np.array([ROOM_X / 2.0, ROOM_Y / 2.0, TX_HEIGHT], dtype=float)
    rx_xyz = np.array([user.pos[0], user.pos[1], RX_HEIGHT], dtype=float)
    path = tx_xyz - rx_xyz
    d = float(np.linalg.norm(path))
    if d <= 0.0:
        return 0.0
    unit_rx_to_tx = path / d
    cos_phi = HF / d
    cos_psi = float(np.dot(unit_rx_to_tx, rx_normal(user.theta_deg, user.phi_deg)))
    if cos_phi <= 0.0 or cos_psi <= 0.0:
        return 0.0
    psi = math.acos(max(-1.0, min(1.0, cos_psi)))
    if psi > psi_c:
        return 0.0
    h = ((LAMBERT_M + 1.0) * A_PD / (2.0 * math.pi * d * d))
    h *= (cos_phi ** LAMBERT_M) * OPTICAL_FILTER * concentrator_gain(psi, psi_c) * cos_psi
    return max(0.0, h * beam_pattern_gain(region, user.pos))


_PATCH_CACHE = {}


def _get_patch_cache(patches: Iterable[Patch]) -> dict[str, np.ndarray]:
    """Vectorized patch cache for first-order NLOS calculations.

    The step-2 implementation computed each patch contribution in nested Python
    loops. This cache keeps the same first-order model but vectorizes the patch
    operations so medium-scale weight sweeps become practical.
    """
    patch_list = list(patches)
    key = id(patch_list) if isinstance(patches, list) else id(patches)
    # list(patches) above would change id for non-list iterables, but the simulator
    # passes a stable list. This guard keeps behavior safe for other iterables.
    if not isinstance(patches, list):
        key = tuple((float(p.pos[0]), float(p.pos[1]), float(p.pos[2]), float(p.rho), float(p.area)) for p in patch_list)
    if key in _PATCH_CACHE:
        return _PATCH_CACHE[key]

    tx_xyz = np.array([ROOM_X / 2.0, ROOM_Y / 2.0, TX_HEIGHT], dtype=float)
    pos = np.array([p.pos for p in patch_list], dtype=float)
    normal = np.array([p.normal for p in patch_list], dtype=float)
    rho = np.array([p.rho for p in patch_list], dtype=float)
    area = np.array([p.area for p in patch_list], dtype=float)
    tx_to_patch = pos - tx_xyz
    d1 = np.linalg.norm(tx_to_patch, axis=1)
    d1 = np.maximum(d1, DEN_EPS)
    u1 = tx_to_patch / d1[:, None]
    cos_phi = np.maximum(0.0, -u1[:, 2])
    cos_alpha = np.maximum(0.0, np.sum((-u1) * normal, axis=1))

    # Beam-pattern gain from each region toward every reflecting patch.
    regions = build_regions()
    beam_gain = np.zeros((len(regions), len(patch_list)), dtype=float)
    for region in regions:
        for qi, patch in enumerate(patch_list):
            beam_gain[region.idx, qi] = beam_pattern_gain(region, patch.pos[:2])

    cache = {
        "pos": pos,
        "normal": normal,
        "rho": rho,
        "area": area,
        "d1": d1,
        "u1": u1,
        "cos_phi": cos_phi,
        "cos_alpha": cos_alpha,
        "beam_gain": beam_gain,
    }
    _PATCH_CACHE[key] = cache
    return cache


def nlos_gain(region: Region, user: User, patches: Iterable[Patch], psi_c: float) -> float:
    cache = _get_patch_cache(patches)
    pos = cache["pos"]
    normal = cache["normal"]
    rho = cache["rho"]
    area = cache["area"]
    d1 = cache["d1"]
    cos_phi = cache["cos_phi"]
    cos_alpha = cache["cos_alpha"]
    beam_gain = cache["beam_gain"][region.idx]

    rx_xyz = np.array([user.pos[0], user.pos[1], RX_HEIGHT], dtype=float)
    n_rx = rx_normal(user.theta_deg, user.phi_deg)
    patch_to_rx = rx_xyz[None, :] - pos
    d2 = np.linalg.norm(patch_to_rx, axis=1)
    d2 = np.maximum(d2, DEN_EPS)
    u2 = patch_to_rx / d2[:, None]

    cos_beta = np.maximum(0.0, np.sum(u2 * normal, axis=1))
    cos_psi = np.maximum(0.0, np.sum((-u2) * n_rx[None, :], axis=1))
    valid = (cos_phi > 0.0) & (cos_alpha > 0.0) & (cos_beta > 0.0) & (cos_psi > 0.0)
    if not np.any(valid):
        return 0.0
    psi = np.arccos(np.clip(cos_psi, -1.0, 1.0))
    valid &= psi <= psi_c
    if not np.any(valid):
        return 0.0

    h = (LAMBERT_M + 1.0) * A_PD * rho * area
    h /= 2.0 * (math.pi ** 2) * (d1 ** 2) * (d2 ** 2)
    h *= (cos_phi ** LAMBERT_M) * cos_alpha * cos_beta * OPTICAL_FILTER
    h *= np.array([concentrator_gain(float(p), psi_c) for p in psi]) * cos_psi * beam_gain
    return float(np.sum(np.maximum(0.0, h[valid])))


def link_gain(region: Region, user: User, patches: list[Patch], include_nlos: bool, psi_c: float) -> float:
    h = los_gain(region, user, psi_c)
    if include_nlos:
        h += nlos_gain(region, user, patches, psi_c)
    return h


def region_power(region: Region) -> float:
    return P_CENTER_BEAM if region.idx == 0 else P_EDGE_BEAM


def pool_key(user: User, region: Region, resource_state: ResourceState = STATIC_RESOURCE_STATE) -> tuple[str, int]:
    if is_edge_user(user, region, resource_state):
        return ("edge", region.edge_band)
    return ("center", 0)


def pool_size(user: User, region: Region, resource_state: ResourceState = STATIC_RESOURCE_STATE) -> int:
    return resource_state.n_edge if is_edge_user(user, region, resource_state) else resource_state.n_center


def compatible_interference(
    target_user: User,
    target_region: Region,
    source_user: User,
    source_region: Region,
    resource_state: ResourceState = STATIC_RESOURCE_STATE,
) -> bool:
    target_pool = pool_key(target_user, target_region, resource_state)
    source_pool = pool_key(source_user, source_region, resource_state)
    return target_pool == source_pool


def local_pool_group(
    uid: int,
    assignments: dict[int, int],
    users: list[User],
    regions: list[Region],
    resource_state: ResourceState = STATIC_RESOURCE_STATE,
) -> list[int]:
    region = regions[assignments[uid]]
    key = pool_key(users[uid], region, resource_state)
    return [
        other_uid
        for other_uid, rid in assignments.items()
        if rid == region.idx and pool_key(users[other_uid], regions[rid], resource_state) == key
    ]


def local_region_group(uid: int, assignments: dict[int, int]) -> list[int]:
    """Users simultaneously served by the same beam/region.

    OMN-PF-A normalizes adaptive power over this group so the total emitted
    power of a region/beam remains equal to region_power(region).
    """
    rid = assignments[uid]
    return [other_uid for other_uid, other_rid in assignments.items() if other_rid == rid]


def _integer_share_counts(
    ordered_uids: list[int],
    pool_size_value: int,
    weights: np.ndarray | None = None,
) -> dict[int, int]:
    """Allocate an integer subcarrier count that sums exactly to the pool size.

    Every admitted user receives at least one subcarrier. Equal weights produce
    the fixed/equal OFDMA baseline; nonuniform weights produce OMN-PF-A's
    reliability-aware allocation. The order is deterministic for a fixed seed.
    """
    if not ordered_uids:
        return {}
    if pool_size_value < len(ordered_uids):
        # This should be prevented by admission control, but keep the helper
        # well-defined by serving the first pool_size users with one tone.
        return {uid: int(i < pool_size_value) for i, uid in enumerate(ordered_uids)}

    if weights is None:
        weights = np.ones(len(ordered_uids), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        weights = np.maximum(weights, DEN_EPS)

    raw = pool_size_value * weights / float(np.sum(weights))
    counts = np.floor(raw).astype(int)
    counts[counts == 0] = 1

    while int(np.sum(counts)) > pool_size_value:
        reducible = np.where(counts > 1)[0]
        if reducible.size == 0:
            break
        # Remove first from the most over-allocated user relative to its raw
        # target; deterministic uid order resolves ties.
        excess = counts[reducible] - raw[reducible]
        idx = int(reducible[np.argmax(excess)])
        counts[idx] -= 1

    while int(np.sum(counts)) < pool_size_value:
        residual = raw - counts
        idx = int(np.argmax(residual))
        counts[idx] += 1

    return {uid: int(counts[i]) for i, uid in enumerate(ordered_uids)}


def _ofdma_weight(
    uid: int,
    users: list[User],
    regions: list[Region],
    link_matrix: np.ndarray,
    nlos_matrix: np.ndarray | None,
    resource_state: ResourceState,
) -> float:
    """Bounded reliability weight used only by adaptive OFDMA/power."""
    user = users[uid]
    region = serving_region(user, regions)
    sinr_est = estimate_sinr(uid, region, users, regions, link_matrix, resource_state=resource_state)
    sinr_db = 10.0 * math.log10(sinr_est + DEN_EPS)
    outage = outage_risk_from_sinr_db(sinr_db)
    gap = min(1.0, user.service_gap_frames / SERVICE_GAP_NORMALIZATION_FRAMES)
    edge = 1.0 if is_edge_user(user, region, resource_state) else 0.0
    transition = transition_risk(user, region, regions)
    weight = 1.0 + 1.50 * outage + 1.25 * edge + 1.00 * gap + 0.35 * transition
    if nlos_matrix is not None:
        nlos_term = nlos_fragility(link_matrix[region.idx, uid], nlos_matrix[region.idx, uid])
        weight += 0.35 * nlos_term * outage
    # Avoid starving strong users or allowing one fragile user to consume the
    # whole pool. The adaptive rule changes shares, but only within a bounded
    # range around the equal-share baseline.
    return min(4.0, max(0.75, float(weight)))


def effective_subcarriers(
    uid: int,
    assignments: dict[int, int],
    users: list[User],
    regions: list[Region],
    link_matrix: np.ndarray,
    nlos_matrix: np.ndarray | None,
    resource_state: ResourceState = STATIC_RESOURCE_STATE,
    adaptive_ofdma: bool = False,
) -> int:
    """Return the user's orthogonal subcarrier share within its local pool."""
    region = regions[assignments[uid]]
    group = local_pool_group(uid, assignments, users, regions, resource_state)
    pool_size_value = pool_size(users[uid], region, resource_state)
    if adaptive_ofdma:
        weights = np.array(
            [_ofdma_weight(other_uid, users, regions, link_matrix, nlos_matrix, resource_state) for other_uid in group],
            dtype=float,
        )
    else:
        weights = None
    return _integer_share_counts(group, pool_size_value, weights).get(uid, 0)


def allocation_power(
    uid: int,
    assignments: dict[int, int],
    users: list[User],
    regions: list[Region],
    link_matrix: np.ndarray | None = None,
    nlos_matrix: np.ndarray | None = None,
    resource_state: ResourceState = STATIC_RESOURCE_STATE,
    adaptive_ofdma: bool = False,
    adaptive_power: bool = False,
) -> float:
    """Allocate total user power while preserving each active region budget.

    Fixed-resource schedulers use a uniform per-subcarrier power density, so a
    user's total power is proportional to its assigned subcarrier count.
    OMN-PF-A applies a bounded reliability/channel tilt and then renormalizes
    over the same region. In both cases, the user powers in an active region
    sum exactly to ``region_power(region)`` (up to floating-point error).
    """
    region = regions[assignments[uid]]
    region_group = local_region_group(uid, assignments)
    if link_matrix is None:
        return region_power(region) / max(1, len(region_group))

    n_sc = {
        other_uid: effective_subcarriers(
            other_uid,
            assignments,
            users,
            regions,
            link_matrix,
            nlos_matrix,
            resource_state=resource_state,
            adaptive_ofdma=adaptive_ofdma,
        )
        for other_uid in region_group
    }
    total_sc = max(1, sum(n_sc.values()))

    if not adaptive_power:
        return region_power(region) * n_sc[uid] / total_sc

    positive_gains = [
        max(DEN_EPS, float(link_matrix[region.idx, other_uid]))
        for other_uid in region_group
    ]
    h_ref = max(DEN_EPS, float(np.median(positive_gains)))
    raw_weights: dict[int, float] = {}
    for other_uid in region_group:
        user = users[other_uid]
        h = max(DEN_EPS, float(link_matrix[region.idx, other_uid]))
        sinr_est = estimate_sinr(
            other_uid,
            region,
            users,
            regions,
            link_matrix,
            resource_state=resource_state,
        )
        sinr_db = 10.0 * math.log10(sinr_est + DEN_EPS)
        outage = outage_risk_from_sinr_db(sinr_db)
        gap = min(1.0, user.service_gap_frames / SERVICE_GAP_NORMALIZATION_FRAMES)
        edge = 1.0 if is_edge_user(user, region, resource_state) else 0.0
        channel_tilt = min(1.50, max(0.75, math.sqrt(h_ref / h)))
        reliability_tilt = min(1.75, 1.0 + 0.35 * outage + 0.25 * gap + 0.25 * edge)
        # Multiplying by n_sc keeps the baseline approximately uniform in
        # power per subcarrier; the bounded tilts then favor fragile links.
        raw_weights[other_uid] = max(DEN_EPS, n_sc[other_uid] * channel_tilt * reliability_tilt)

    weight_sum = sum(raw_weights.values())
    return region_power(region) * raw_weights[uid] / max(DEN_EPS, weight_sum)


def local_pool_load(
    uid: int,
    assignments: dict[int, int],
    users: list[User],
    regions: list[Region],
    resource_state: ResourceState = STATIC_RESOURCE_STATE,
) -> int:
    return len(local_pool_group(uid, assignments, users, regions, resource_state))


def estimate_sinr(
    uid: int,
    region: Region,
    users: list[User],
    regions: list[Region],
    link_matrix: np.ndarray,
    resource_state: ResourceState = STATIC_RESOURCE_STATE,
) -> float:
    user = users[uid]
    # Scheduling proxy retains the established dynamic model's effective
    # per-link power convention. The post-allocation calculation below applies
    # the common multi-user split while keeping comparisons with the original
    # single-user dynamic results interpretable.
    p = region_power(region)
    signal = (ETA_BASE * link_matrix[region.idx, uid] * p) ** 2
    interference = 0.0
    target_pool = pool_key(user, region, resource_state)
    for source_region in regions:
        if source_region.idx == region.idx:
            continue
        if target_pool[0] == "center" or source_region.edge_band == region.edge_band:
            interference += (
                ETA_BASE * link_matrix[source_region.idx, uid] * region_power(source_region)
            ) ** 2
    return signal / (0.25 * interference + NOISE_SUB + DEN_EPS)


def sinr_and_rate(
    uid: int,
    assignments: dict[int, int],
    users: list[User],
    regions: list[Region],
    link_matrix: np.ndarray,
    nlos_matrix: np.ndarray | None = None,
    resource_state: ResourceState = STATIC_RESOURCE_STATE,
    adaptive_ofdma: bool = False,
    adaptive_power: bool = False,
) -> tuple[float, float]:
    """Compute per-subcarrier SINR and aggregate user spectral efficiency."""
    user = users[uid]
    region = regions[assignments[uid]]
    n_sc = effective_subcarriers(
        uid,
        assignments,
        users,
        regions,
        link_matrix,
        nlos_matrix,
        resource_state=resource_state,
        adaptive_ofdma=adaptive_ofdma,
    )
    if n_sc <= 0:
        return 0.0, 0.0

    p_total = allocation_power(
        uid,
        assignments,
        users,
        regions,
        link_matrix=link_matrix,
        nlos_matrix=nlos_matrix,
        resource_state=resource_state,
        adaptive_ofdma=adaptive_ofdma,
        adaptive_power=adaptive_power,
    )
    # ``p_total`` is the user's normalized effective power share under the
    # established dynamic model. OFDMA affects orthogonality and bandwidth;
    # it is not divided by n_sc a second time here.
    signal = (ETA_BASE * link_matrix[region.idx, uid] * p_total) ** 2

    interference = 0.0
    target_key = pool_key(user, region, resource_state)
    # OFDMA users in the same region are orthogonal and never interfere. For a
    # different region reusing the same pool, use the expected overlap of its
    # assigned subcarriers with the target user's subcarriers.
    source_region_ids = sorted(set(assignments.values()))
    for source_rid in source_region_ids:
        if source_rid == region.idx:
            continue
        source_region = regions[source_rid]
        source_uids = [
            other_uid
            for other_uid, rid in assignments.items()
            if rid == source_rid
            and pool_key(users[other_uid], source_region, resource_state) == target_key
        ]
        if not source_uids:
            continue
        source_pool_size = max(1, resource_state.n_center if target_key[0] == "center" else resource_state.n_edge)
        h_int = link_matrix[source_region.idx, uid]
        for source_uid in source_uids:
            source_n_sc = effective_subcarriers(
                source_uid,
                assignments,
                users,
                regions,
                link_matrix,
                nlos_matrix,
                resource_state=resource_state,
                adaptive_ofdma=adaptive_ofdma,
            )
            if source_n_sc <= 0:
                continue
            source_total_power = allocation_power(
                source_uid,
                assignments,
                users,
                regions,
                link_matrix=link_matrix,
                nlos_matrix=nlos_matrix,
                resource_state=resource_state,
                adaptive_ofdma=adaptive_ofdma,
                adaptive_power=adaptive_power,
            )
            overlap_fraction = min(1.0, source_n_sc / source_pool_size)
            interference += overlap_fraction * (ETA_BASE * h_int * source_total_power) ** 2

    sinr = signal / (interference + NOISE_SUB + DEN_EPS)
    return sinr, n_sc * math.log2(1.0 + sinr)

def region_candidates(users: list[User], regions: list[Region]) -> dict[int, list[int]]:
    grouped = {region.idx: [] for region in regions}
    for user in users:
        grouped[serving_region(user, regions).idx].append(user.uid)
    return grouped


def outage_risk_from_sinr_db(sinr_db: float, threshold_db: float = 10.0) -> float:
    return 1.0 / (1.0 + math.exp((sinr_db - threshold_db) / 2.5))


def transition_risk(user: User, region: Region, regions: list[Region]) -> float:
    distances = sorted(float(np.linalg.norm(user.pos - other.center)) for other in regions)
    if len(distances) < 2:
        return 0.0
    # A small nearest/second-nearest distance gap means the user is close to a
    # beam-region boundary and is more likely to transition soon.
    gap = max(0.0, distances[1] - distances[0])
    boundary_risk = math.exp(-gap / (0.30 * R_MICRO))
    recent_transition = 1.0 if user.last_region >= 0 and user.last_region != region.idx else 0.0
    return max(boundary_risk, recent_transition)


def nlos_fragility(total_gain: float, nlos_component: float) -> float:
    """Return how dependent the current link is on reflected/NLOS energy.

    N_u(t) = 1 - H_LOS/(H_LOS + H_NLOS + eps) = H_NLOS/(H_total + eps).
    The value is near zero for LOS-dominant links and increases when the
    selected user is more dependent on reflected paths.
    """
    if total_gain <= DEN_EPS or nlos_component <= 0.0:
        return 0.0
    return max(0.0, min(1.0, nlos_component / (total_gain + DEN_EPS)))


def select_adaptive_resource_state(
    users: list[User],
    regions: list[Region],
    link_matrix: np.ndarray,
    nlos_matrix: np.ndarray | None,
    previous_delta: float,
) -> ResourceState:
    """Experimental frame-level adaptive strict-FFR resource-state selector.

    This helper is retained for sensitivity/prototyping only. The manuscript
    variant `omn_pf_a` deliberately keeps delta fixed at 2/3 and adapts OFDMA
    sharing plus normalized power allocation, because the tested delta-adaptive
    heuristic tended to shrink protected edge pools.
    """
    best_state = STATIC_RESOURCE_STATE
    best_score = float("inf")
    delta_span = max(1e-9, max(ADAPTIVE_DELTA_CANDIDATES) - min(ADAPTIVE_DELTA_CANDIDATES))

    for delta_candidate in ADAPTIVE_DELTA_CANDIDATES:
        state = make_resource_state(delta_candidate)
        center_users = 0
        edge_users = 0
        outage_terms: list[float] = []
        gap_terms: list[float] = []

        for user in users:
            region = serving_region(user, regions)
            edge_indicator = 1.0 if is_edge_user(user, region, state) else 0.0
            if edge_indicator:
                edge_users += 1
            else:
                center_users += 1

            sinr_est = estimate_sinr(user.uid, region, users, regions, link_matrix, resource_state=state)
            sinr_db = 10.0 * math.log10(sinr_est + DEN_EPS)
            outage = outage_risk_from_sinr_db(sinr_db)
            if nlos_matrix is not None:
                nlos_term = nlos_fragility(link_matrix[region.idx, user.uid], nlos_matrix[region.idx, user.uid])
                outage = min(1.0, outage + 0.25 * nlos_term * outage)
            gap = min(1.0, user.service_gap_frames / SERVICE_GAP_NORMALIZATION_FRAMES)

            outage_terms.append(outage * (1.0 + 0.5 * edge_indicator))
            gap_terms.append(gap * (1.0 + 0.5 * edge_indicator))

        load_pressure = max(
            center_users / max(1, state.n_center),
            edge_users / max(1, 3 * state.n_edge),
        )
        outage_pressure = float(np.mean(outage_terms)) if outage_terms else 0.0
        gap_pressure = float(np.mean(gap_terms)) if gap_terms else 0.0
        switch_penalty = abs(state.delta - previous_delta) / delta_span

        score = (
            0.50 * load_pressure
            + 1.00 * outage_pressure
            + 0.50 * gap_pressure
            + 0.25 * switch_penalty
        )
        if score < best_score:
            best_score = score
            best_state = state

    return best_state


def _predict_common_admission_state(
    selected_uids: list[int],
    users: list[User],
    region: Region,
    regions: list[Region],
    link_matrix: np.ndarray,
    resource_state: ResourceState,
) -> tuple[dict[int, float], dict[int, float], dict[int, int], float]:
    """Predict equal-OFDMA/uniform-power performance for admission control.

    Admission is intentionally evaluated with the same fixed/equal resource
    model for every scheduler, including OMN-PF-A. This keeps user admission a
    common control and isolates OMN-PF-A's adaptive OFDMA/power layer to the
    post-admission allocation stage.

    The scheduling proxy ``estimate_sinr`` represents the user's serving link
    with the full regional power budget. The post-sharing prediction scales the
    signal term by the squared normalized user-power share, matching the
    simulator's established optical-power convention.
    """
    if not selected_uids:
        return {}, {}, {}, 0.0

    pool_groups: dict[tuple[str, int], list[int]] = {}
    for uid in selected_uids:
        key = pool_key(users[uid], region, resource_state)
        pool_groups.setdefault(key, []).append(uid)

    n_sc: dict[int, int] = {}
    for group_uids in pool_groups.values():
        pool_size_value = pool_size(users[group_uids[0]], region, resource_state)
        n_sc.update(_integer_share_counts(group_uids, pool_size_value, weights=None))

    total_sc = max(1, sum(n_sc.values()))
    sinr_by_uid: dict[int, float] = {}
    rate_by_uid: dict[int, float] = {}
    for uid in selected_uids:
        base_sinr = estimate_sinr(
            uid,
            region,
            users,
            regions,
            link_matrix,
            resource_state=resource_state,
        )
        power_fraction = n_sc[uid] / total_sc
        predicted_sinr = base_sinr * (power_fraction ** 2)
        sinr_by_uid[uid] = float(predicted_sinr)
        rate_by_uid[uid] = float(n_sc[uid] * math.log2(1.0 + predicted_sinr))

    return sinr_by_uid, rate_by_uid, n_sc, float(sum(rate_by_uid.values()))


def select_multi_user_set(
    ordered_uids: list[int],
    users: list[User],
    region: Region,
    regions: list[Region],
    link_matrix: np.ndarray,
    resource_state: ResourceState,
) -> list[int]:
    """Admit a variable resource- and SINR-aware multi-user OFDMA set.

    Scheduler-specific logic supplies ``ordered_uids``. This common admission
    layer then adds users one at a time without a fixed two-user cap. The
    maximum feasible count is derived from each center/edge pool's available
    subcarriers. Beyond the first user, a candidate is retained only when:

    * every admitted user receives at least the configured minimum tones;
    * every admitted link remains above the post-sharing SINR threshold; and
    * predicted regional sum rate remains above a conservative fraction of the
      first-user reference rate.

    The first ranked user is always admitted so an active region is never left
    idle solely because its channel is below the admission threshold.
    """
    selected: list[int] = []
    pool_counts: dict[tuple[str, int], int] = {}
    reference_sum_rate: float | None = None
    min_sinr_linear = 10.0 ** (DYNAMIC_ADMISSION_MIN_SINR_DB / 10.0)

    for uid in ordered_uids:
        key = pool_key(users[uid], region, resource_state)
        capacity_from_pool = max(
            1,
            pool_size(users[uid], region, resource_state)
            // DYNAMIC_MIN_SUBCARRIERS_PER_USER,
        )
        if pool_counts.get(key, 0) >= capacity_from_pool:
            continue

        tentative = selected + [uid]
        sinr_by_uid, _rate_by_uid, n_sc, sum_rate = _predict_common_admission_state(
            tentative,
            users,
            region,
            regions,
            link_matrix,
            resource_state,
        )

        if not selected:
            selected = tentative
            pool_counts[key] = 1
            reference_sum_rate = max(DEN_EPS, sum_rate)
            continue

        enough_subcarriers = all(
            n_sc.get(other_uid, 0) >= DYNAMIC_MIN_SUBCARRIERS_PER_USER
            for other_uid in tentative
        )
        sinr_safe = all(
            sinr_by_uid.get(other_uid, 0.0) >= min_sinr_linear
            for other_uid in tentative
        )
        rate_safe = sum_rate >= (
            DYNAMIC_ADMISSION_MIN_SUM_RATE_RETENTION
            * max(DEN_EPS, reference_sum_rate if reference_sum_rate is not None else sum_rate)
        )

        if enough_subcarriers and sinr_safe and rate_safe:
            selected = tentative
            pool_counts[key] = pool_counts.get(key, 0) + 1

    return selected

def schedule_users(
    scheduler_name: str,
    users: list[User],
    regions: list[Region],
    link_matrix: np.ndarray,
    frame: int,
    nlos_matrix: np.ndarray | None = None,
    resource_state: ResourceState = STATIC_RESOURCE_STATE,
) -> dict[int, int]:
    config = SCHEDULERS[scheduler_name]
    grouped = region_candidates(users, regions)

    if scheduler_name == "full-load":
        return {uid: serving_region(users[uid], regions).idx for uid in range(len(users))}

    assignments: dict[int, int] = {}
    for rid, candidates in grouped.items():
        if not candidates:
            continue
        region = regions[rid]

        if scheduler_name == "rr1":
            # Rotate the candidate order; the common resource/SINR-aware
            # admission layer determines how many users can be served.
            offset = frame % len(candidates)
            ordered_uids = candidates[offset:] + candidates[:offset]
        else:
            metrics: list[tuple[float, int]] = []
            for uid in candidates:
                sinr_est = estimate_sinr(
                    uid,
                    region,
                    users,
                    regions,
                    link_matrix,
                    resource_state=resource_state,
                )
                predicted_rate = pool_size(users[uid], region, resource_state) * math.log2(1.0 + sinr_est)
                sinr_db = 10.0 * math.log10(sinr_est + DEN_EPS)
                if scheduler_name == "max_sinr":
                    metric = sinr_est
                elif scheduler_name == "proportional_fair":
                    metric = predicted_rate / (users[uid].avg_rate + RATE_EPS)
                else:
                    denom = (
                        (users[uid].avg_rate + RATE_EPS) ** config.pf_alpha
                        if config.use_fairness_history
                        else 1.0
                    )
                    weight = 1.0
                    if config.use_outage:
                        weight += config.lambda_outage * max(
                            outage_risk_from_sinr_db(sinr_db),
                            users[uid].outage_memory,
                        )
                    if config.use_edge and is_edge_user(users[uid], region, resource_state):
                        weight += config.lambda_edge
                    if config.use_transition:
                        weight += config.lambda_transition * transition_risk(users[uid], region, regions)
                    if config.use_nlos and nlos_matrix is not None:
                        nlos_term = nlos_fragility(
                            link_matrix[region.idx, uid],
                            nlos_matrix[region.idx, uid],
                        )
                        nlos_assistance = nlos_term * outage_risk_from_sinr_db(sinr_db)
                        nlos_stability = (1.0 - nlos_term) * (1.0 - outage_risk_from_sinr_db(sinr_db))
                        weight += config.lambda_nlos * (
                            0.75 * nlos_assistance + 0.25 * nlos_stability
                        )
                    if config.use_service_gap:
                        gap_term = min(
                            1.0,
                            users[uid].service_gap_frames / SERVICE_GAP_NORMALIZATION_FRAMES,
                        )
                        weight += config.lambda_gap * gap_term
                    metric = (predicted_rate / denom) * weight
                metrics.append((float(metric), uid))
            ordered_uids = [uid for _metric, uid in sorted(metrics, reverse=True)]

        for chosen_uid in select_multi_user_set(
            ordered_uids,
            users,
            region,
            regions,
            link_matrix,
            resource_state,
        ):
            assignments[chosen_uid] = rid
    return assignments

def jain(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=float)
    denom = arr.size * float(np.sum(arr * arr))
    if denom <= 0.0:
        return 0.0
    return float((np.sum(arr) ** 2) / denom)


def outage_segments(flags: list[bool]) -> list[int]:
    segments: list[int] = []
    run = 0
    for flag in flags:
        if flag:
            run += 1
        elif run:
            segments.append(run)
            run = 0
    if run:
        segments.append(run)
    return segments


def compute_link_component_matrices(
    regions: list[Region],
    users: list[User],
    patches: list[Patch],
    include_nlos: bool,
    psi_c: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return total, LOS-only, and NLOS-only link-gain matrices.

    Keeping LOS and NLOS components separate allows the OMN-PF scheduler to
    include an explicit NLOS-fragility term while preserving the same total
    gain used by SINR/rate calculations.
    """
    los_mat = np.zeros((len(regions), len(users)), dtype=float)
    nlos_mat = np.zeros_like(los_mat)
    for region in regions:
        for user in users:
            los = los_gain(region, user, psi_c)
            nlos = nlos_gain(region, user, patches, psi_c) if include_nlos else 0.0
            if include_nlos and NLOS_PROFILE == "stress":
                # Stress setting: higher-reflection indoor surfaces and partial LOS
                # degradation for tilted and edge users. This is not used for the
                # baseline paper result; it isolates whether the NLOS-aware
                # scheduler component contributes under reflection-dependent links.
                tilt_factor = math.exp(-max(0.0, user.theta_deg) / LOS_STRESS_TILT_DECAY_DEG)
                edge_factor = LOS_STRESS_EDGE_DECAY if is_edge_user(user, region) else 1.0
                los = los * tilt_factor * edge_factor
                nlos = nlos * NLOS_STRESS_SCALE
            los_mat[region.idx, user.uid] = los
            nlos_mat[region.idx, user.uid] = nlos
    return los_mat + nlos_mat, los_mat, nlos_mat


def compute_link_matrix(regions: list[Region], users: list[User], patches: list[Patch], include_nlos: bool, psi_c: float) -> np.ndarray:
    total_mat, _los_mat, _nlos_mat = compute_link_component_matrices(regions, users, patches, include_nlos, psi_c)
    return total_mat


def simulate(
    trials: int = 50,
    frames: int = 50,
    users_per_macro: int = 42,
    scheduler: str = "omn_pf",
    include_mobility: bool = True,
    include_orientation: bool = True,
    include_nlos: bool = True,
    dt: float = 0.1,
    seed: int = 7,
    nlos_profile: str = "baseline",
) -> dict[str, float | str]:
    """Run the dynamic simulator and compute scheduler-level metrics.

    Step-2 metric correction:
    - scheduled_* metrics are computed only for actually scheduled links.
    - link_opportunity_* metrics are computed for every user at every frame by
      evaluating the user's potential serving link under the current active
      interference field.
    - service/outage/continuity metrics are computed for every user at every
      frame and count an unscheduled user as not served. These fields now match
      the all-user/all-frame interpretation used in the paper equations.
    - beam switching is tracked for every user at every frame, not only for
      scheduled users.
    Step-5 metric addition:
    - edge-user service continuity/outage is computed over all frames in which
      a user is classified as edge under the current serving region. This makes
      OMN-PF evaluation directly tied to the strict-FFR edge-protection goal.
    """
    if scheduler not in SCHEDULERS:
        raise ValueError(f"unknown scheduler {scheduler!r}")
    rng = np.random.default_rng(seed)
    regions = build_regions()
    patches = make_patches()
    psi_c = math.radians(RX_FOV_DEG)
    thresholds = {thr: 10.0 ** (thr / 10.0) for thr in SINR_THRESHOLDS_DB}

    scheduled_sinr: list[float] = []
    scheduled_rate: list[float] = []
    all_link_sinr: list[float] = []
    all_link_rate: list[float] = []
    service_rate: list[float] = []

    # Service-regularity metrics. These are evaluated from the actual
    # scheduled/not-scheduled history of each user and are useful for
    # distinguishing opportunistic Max-SINR selection from regular service
    # over time. They do not change the scheduling decision; they only measure
    # service continuity after the simulation.
    service_gap_lengths: list[int] = []
    service_longest_gaps: list[int] = []
    avg_inter_service_gaps: list[float] = []
    starvation_events: list[float] = []
    per_user_service_rates: list[float] = []
    edge_served_fractions: list[float] = []
    center_served_fractions: list[float] = []

    fairness_samples: list[float] = []
    link_continuity: dict[float, list[float]] = {thr: [] for thr in SINR_THRESHOLDS_DB}
    service_continuity: dict[float, list[float]] = {thr: [] for thr in SINR_THRESHOLDS_DB}
    scheduled_continuity: dict[float, list[float]] = {thr: [] for thr in SINR_THRESHOLDS_DB}
    edge_service_continuity: dict[float, list[float]] = {thr: [] for thr in SINR_THRESHOLDS_DB}
    center_service_continuity: dict[float, list[float]] = {thr: [] for thr in SINR_THRESHOLDS_DB}

    link_outage_duration_20: list[int] = []
    service_outage_duration_20: list[int] = []
    switch_rates: list[float] = []
    served_fraction: list[float] = []
    delta_used_samples: list[float] = []
    center_pool_samples: list[int] = []
    edge_pool_samples: list[int] = []
    delta_switch_events: list[float] = []
    scheduled_users_per_frame_samples: list[float] = []
    scheduled_users_per_region_samples: list[float] = []
    scheduled_users_per_active_region_samples: list[float] = []
    effective_subcarrier_samples: list[float] = []
    region_power_budget_error_samples: list[float] = []
    active_region_user_count_samples: list[int] = []

    start = time.time()
    for _trial in range(trials):
        users = initialize_users(users_per_macro, regions, rng)
        link_outage_flags = {thr: [[] for _ in users] for thr in SINR_THRESHOLDS_DB}
        service_outage_flags = {thr: [[] for _ in users] for thr in SINR_THRESHOLDS_DB}
        scheduled_outage_flags = {thr: [[] for _ in users] for thr in SINR_THRESHOLDS_DB}
        edge_service_outage_flags = {thr: [[] for _ in users] for thr in SINR_THRESHOLDS_DB}
        center_service_outage_flags = {thr: [[] for _ in users] for thr in SINR_THRESHOLDS_DB}
        region_history = [[] for _ in users]
        edge_frame_counts = np.zeros(len(users), dtype=float)
        center_frame_counts = np.zeros(len(users), dtype=float)
        served_counts = np.zeros(len(users), dtype=float)
        edge_served_counts = np.zeros(len(users), dtype=float)
        center_served_counts = np.zeros(len(users), dtype=float)
        service_rate_sums = np.zeros(len(users), dtype=float)
        served_history = [[] for _ in users]
        previous_delta = DELTA

        for frame in range(frames):
            if include_mobility and frame > 0:
                update_positions(users, dt, rng)
            if include_orientation and frame > 0:
                update_orientation(users, rng)

            current_region_ids = [serving_region(user, regions).idx for user in users]
            for uid, rid in enumerate(current_region_ids):
                region_history[uid].append(rid)

            link_matrix, _los_matrix, nlos_matrix = compute_link_component_matrices(regions, users, patches, include_nlos, psi_c)
            if scheduler == "omn_pf_a":
                # OMN-PF-A keeps the validated strict-FFR split fixed at
                # delta = 2/3 and adapts only the effective OFDMA sharing and
                # normalized power redistribution. Earlier experiments showed
                # that frame-level delta adaptation could shrink the protected
                # edge pools and reduce edge reliability/fairness. Delta
                # sensitivity should therefore be reported separately, while
                # this adaptive variant focuses on OFDMA/power control.
                resource_state = STATIC_RESOURCE_STATE
                adaptive_ofdma = True
                adaptive_power = True
            else:
                resource_state = STATIC_RESOURCE_STATE
                adaptive_ofdma = False
                adaptive_power = False

            delta_used_samples.append(resource_state.delta)
            center_pool_samples.append(resource_state.n_center)
            edge_pool_samples.append(resource_state.n_edge)
            delta_switch_events.append(float(abs(resource_state.delta - previous_delta) > 1e-9))
            previous_delta = resource_state.delta

            assignments = schedule_users(
                scheduler,
                users,
                regions,
                link_matrix,
                frame,
                nlos_matrix,
                resource_state=resource_state,
            )

            # Common multi-user OFDMA diagnostics. These verify that all
            # schedulers operate under the same simultaneous-user limits and
            # that adaptive power never increases a region's total budget.
            region_counts = [sum(1 for rid in assignments.values() if rid == region.idx) for region in regions]
            scheduled_users_per_frame_samples.append(float(len(assignments)))
            scheduled_users_per_region_samples.append(float(np.mean(region_counts)))
            active_counts = [count for count in region_counts if count > 0]
            active_region_user_count_samples.extend(active_counts)
            scheduled_users_per_active_region_samples.append(
                float(np.mean(active_counts)) if active_counts else 0.0
            )
            for uid in assignments:
                effective_subcarrier_samples.append(
                    float(
                        effective_subcarriers(
                            uid,
                            assignments,
                            users,
                            regions,
                            link_matrix,
                            nlos_matrix,
                            resource_state=resource_state,
                            adaptive_ofdma=adaptive_ofdma,
                        )
                    )
                )
            for rid in sorted(set(assignments.values())):
                active_uids = [uid for uid, assigned_rid in assignments.items() if assigned_rid == rid]
                allocated = sum(
                    allocation_power(
                        uid,
                        assignments,
                        users,
                        regions,
                        link_matrix=link_matrix,
                        nlos_matrix=nlos_matrix,
                        resource_state=resource_state,
                        adaptive_ofdma=adaptive_ofdma,
                        adaptive_power=adaptive_power,
                    )
                    for uid in active_uids
                )
                budget = region_power(regions[rid])
                region_power_budget_error_samples.append(abs(allocated - budget) / max(DEN_EPS, budget))

            frame_rates = np.zeros(len(users), dtype=float)
            scheduled_frame_sinr: dict[int, float] = {}
            scheduled_frame_rate: dict[int, float] = {}

            # Actual scheduled-link SINR/rate.
            for uid, rid in assignments.items():
                sinr, rate = sinr_and_rate(
                    uid,
                    assignments,
                    users,
                    regions,
                    link_matrix,
                    nlos_matrix=nlos_matrix,
                    resource_state=resource_state,
                    adaptive_ofdma=adaptive_ofdma,
                    adaptive_power=adaptive_power,
                )
                scheduled_frame_sinr[uid] = sinr
                scheduled_frame_rate[uid] = rate
                frame_rates[uid] = rate
                scheduled_sinr.append(sinr)
                scheduled_rate.append(rate)
                service_rate.append(rate)
                served_counts[uid] += 1.0
                service_rate_sums[uid] += rate
                for thr, lin in thresholds.items():
                    scheduled_outage_flags[thr][uid].append(sinr < lin)

            # Every-user link opportunity and every-user service metrics.
            for uid, rid in enumerate(current_region_ids):
                if uid in scheduled_frame_sinr:
                    link_sinr = scheduled_frame_sinr[uid]
                    link_rate = scheduled_frame_rate[uid]
                    served = True
                else:
                    potential_assignments = dict(assignments)
                    potential_assignments[uid] = rid
                    link_sinr, link_rate = sinr_and_rate(
                        uid,
                        potential_assignments,
                        users,
                        regions,
                        link_matrix,
                        nlos_matrix=nlos_matrix,
                        resource_state=resource_state,
                        adaptive_ofdma=adaptive_ofdma,
                        adaptive_power=adaptive_power,
                    )
                    served = False
                    service_rate.append(0.0)

                all_link_sinr.append(link_sinr)
                all_link_rate.append(link_rate)
                current_region = regions[rid]
                current_edge = is_edge_user(users[uid], current_region, resource_state)
                if current_edge:
                    edge_frame_counts[uid] += 1.0
                    if served:
                        edge_served_counts[uid] += 1.0
                else:
                    center_frame_counts[uid] += 1.0
                    if served:
                        center_served_counts[uid] += 1.0
                served_history[uid].append(served)

                for thr, lin in thresholds.items():
                    link_outage = link_sinr < lin
                    service_outage = (not served) or link_outage
                    link_outage_flags[thr][uid].append(link_outage)
                    service_outage_flags[thr][uid].append(service_outage)
                    if current_edge:
                        edge_service_outage_flags[thr][uid].append(service_outage)
                    else:
                        center_service_outage_flags[thr][uid].append(service_outage)

                # Outage memory is updated for all users, so unscheduled but
                # vulnerable users can later be prioritized by OMN-PF.
                users[uid].outage_memory = 0.8 * users[uid].outage_memory + 0.2 * float(link_sinr < thresholds[10.0])

            # PF history and service-gap memory are updated for every user every
            # frame. Unscheduled users get zero instantaneous service rate,
            # causing their PF average to decay. The explicit service-gap memory
            # is used by OMN-PF to reduce long waiting intervals.
            for uid, user in enumerate(users):
                user.avg_rate = 0.9 * user.avg_rate + 0.1 * frame_rates[uid]
                if uid in scheduled_frame_sinr:
                    user.service_gap_frames = 0
                else:
                    user.service_gap_frames += 1
                user.last_region = current_region_ids[uid]

            fairness_samples.append(jain(frame_rates))

        for uid, _user in enumerate(users):
            served_fraction.append(float(served_counts[uid] / max(1, frames)))
            per_user_service_rates.append(float(service_rate_sums[uid] / max(1, frames)))

            # Service-gap metrics are based only on whether a user was scheduled
            # in each frame. They are independent of SINR threshold, so they
            # directly capture scheduling regularity and starvation risk.
            not_served_flags = [not flag for flag in served_history[uid]]
            gaps = outage_segments(not_served_flags)
            service_gap_lengths.extend(gaps)
            longest_gap = max(gaps) if gaps else 0
            service_longest_gaps.append(longest_gap)
            starvation_events.append(float(longest_gap >= SERVICE_STARVATION_GAP_FRAMES))
            served_indices = [idx for idx, flag in enumerate(served_history[uid]) if flag]
            if len(served_indices) >= 2:
                avg_inter_service_gaps.append(float(np.mean(np.diff(served_indices))))
            else:
                # If a user is served fewer than twice, the inter-service gap is
                # treated as the full trial length, reflecting very irregular service.
                avg_inter_service_gaps.append(float(frames))

            if edge_frame_counts[uid] > 0:
                edge_served_fractions.append(float(edge_served_counts[uid] / edge_frame_counts[uid]))
            if center_frame_counts[uid] > 0:
                center_served_fractions.append(float(center_served_counts[uid] / center_frame_counts[uid]))

            for thr in SINR_THRESHOLDS_DB:
                link_flags = link_outage_flags[thr][uid]
                service_flags = service_outage_flags[thr][uid]
                sched_flags = scheduled_outage_flags[thr][uid]
                if link_flags:
                    link_continuity[thr].append(1.0 - float(np.mean(link_flags)))
                if service_flags:
                    service_continuity[thr].append(1.0 - float(np.mean(service_flags)))
                if sched_flags:
                    scheduled_continuity[thr].append(1.0 - float(np.mean(sched_flags)))
                edge_flags = edge_service_outage_flags[thr][uid]
                center_flags = center_service_outage_flags[thr][uid]
                if edge_flags:
                    edge_service_continuity[thr].append(1.0 - float(np.mean(edge_flags)))
                if center_flags:
                    center_service_continuity[thr].append(1.0 - float(np.mean(center_flags)))

            link_outage_duration_20.extend(outage_segments(link_outage_flags[20.0][uid]))
            service_outage_duration_20.extend(outage_segments(service_outage_flags[20.0][uid]))
            hist = region_history[uid]
            if len(hist) > 1:
                switch_rates.append(sum(1 for a, b in zip(hist, hist[1:]) if a != b) / float(len(hist) - 1))

    scheduled_sinr_arr = np.array(scheduled_sinr if scheduled_sinr else [0.0], dtype=float)
    scheduled_rate_arr = np.array(scheduled_rate if scheduled_rate else [0.0], dtype=float)
    all_link_sinr_arr = np.array(all_link_sinr if all_link_sinr else [0.0], dtype=float)
    all_link_rate_arr = np.array(all_link_rate if all_link_rate else [0.0], dtype=float)
    service_rate_arr = np.array(service_rate if service_rate else [0.0], dtype=float)
    user_service_rate_arr = np.array(per_user_service_rates if per_user_service_rates else [0.0], dtype=float)
    service_gap_arr = np.array(service_gap_lengths if service_gap_lengths else [0.0], dtype=float)
    longest_gap_arr = np.array(service_longest_gaps if service_longest_gaps else [0.0], dtype=float)
    inter_service_gap_arr = np.array(avg_inter_service_gaps if avg_inter_service_gaps else [float(frames)], dtype=float)
    starvation_arr = np.array(starvation_events if starvation_events else [0.0], dtype=float)
    edge_served_fraction_arr = np.array(edge_served_fractions if edge_served_fractions else [0.0], dtype=float)
    center_served_fraction_arr = np.array(center_served_fractions if center_served_fractions else [0.0], dtype=float)
    link_duration_arr = np.array(link_outage_duration_20 if link_outage_duration_20 else [0.0], dtype=float)
    service_duration_arr = np.array(service_outage_duration_20 if service_outage_duration_20 else [0.0], dtype=float)
    delta_arr = np.array(delta_used_samples if delta_used_samples else [DELTA], dtype=float)
    center_pool_arr = np.array(center_pool_samples if center_pool_samples else [N_CENTER], dtype=float)
    edge_pool_arr = np.array(edge_pool_samples if edge_pool_samples else [N_EDGE], dtype=float)
    delta_switch_arr = np.array(delta_switch_events if delta_switch_events else [0.0], dtype=float)
    scheduled_users_frame_arr = np.array(
        scheduled_users_per_frame_samples if scheduled_users_per_frame_samples else [0.0],
        dtype=float,
    )
    scheduled_users_region_arr = np.array(
        scheduled_users_per_region_samples if scheduled_users_per_region_samples else [0.0],
        dtype=float,
    )
    scheduled_users_active_region_arr = np.array(
        scheduled_users_per_active_region_samples if scheduled_users_per_active_region_samples else [0.0],
        dtype=float,
    )
    effective_subcarrier_arr = np.array(
        effective_subcarrier_samples if effective_subcarrier_samples else [0.0],
        dtype=float,
    )
    power_budget_error_arr = np.array(
        region_power_budget_error_samples if region_power_budget_error_samples else [0.0],
        dtype=float,
    )
    active_region_user_count_arr = np.array(
        active_region_user_count_samples if active_region_user_count_samples else [0],
        dtype=float,
    )

    def _delta_fraction(target: float) -> float:
        return float(np.mean(np.isclose(delta_arr, target, atol=1e-6)))

    return {
        "scheduler": scheduler,
        "trials": float(trials),
        "frames": float(frames),
        "users_per_macro": float(users_per_macro),
        "include_mobility": float(include_mobility),
        "include_orientation": float(include_orientation),
        "include_nlos": float(include_nlos),
        "nlos_profile": nlos_profile,
        "access_model": "dynamic_multi_user_ofdma",
        "resource_control": (
            "adaptive_ofdma_power_fixed_delta"
            if scheduler == "omn_pf_a"
            else "fixed_equal_ofdma_uniform_power"
        ),
        "mean_delta_used": float(np.mean(delta_arr)),
        "std_delta_used": float(np.std(delta_arr)),
        "delta_switch_rate": float(np.mean(delta_switch_arr)),
        "mean_center_pool_size": float(np.mean(center_pool_arr)),
        "mean_edge_pool_size": float(np.mean(edge_pool_arr)),
        "frac_delta_0_5": _delta_fraction(0.5),
        "frac_delta_0_6667": _delta_fraction(2.0 / 3.0),
        "frac_delta_0_8": _delta_fraction(0.8),
        "adaptive_delta_enabled": 0.0,
        "adaptive_ofdma_enabled": float(scheduler == "omn_pf_a"),
        "adaptive_power_enabled": float(scheduler == "omn_pf_a"),
        "multi_user_ofdma_enabled": 1.0,
        "admission_control": "variable_resource_sinr_aware",
        "admission_min_sinr_db": float(DYNAMIC_ADMISSION_MIN_SINR_DB),
        "admission_min_sum_rate_retention": float(DYNAMIC_ADMISSION_MIN_SUM_RATE_RETENTION),
        "max_users_per_region": float(DYNAMIC_MAX_USERS_PER_REGION),
        "max_users_per_pool": float(DYNAMIC_MAX_USERS_PER_POOL),
        "max_center_users_per_pool": float(DYNAMIC_MAX_CENTER_USERS_PER_POOL),
        "max_edge_users_per_pool": float(DYNAMIC_MAX_EDGE_USERS_PER_POOL),
        "min_subcarriers_per_user": float(DYNAMIC_MIN_SUBCARRIERS_PER_USER),
        "mean_scheduled_users_per_frame": float(np.mean(scheduled_users_frame_arr)),
        "mean_scheduled_users_per_region": float(np.mean(scheduled_users_region_arr)),
        "mean_scheduled_users_per_active_region": float(np.mean(scheduled_users_active_region_arr)),
        "max_scheduled_users_in_region_observed": float(np.max(active_region_user_count_arr)),
        "fraction_active_regions_serving_1_user": float(np.mean(active_region_user_count_arr == 1.0)),
        "fraction_active_regions_serving_2_users": float(np.mean(active_region_user_count_arr == 2.0)),
        "fraction_active_regions_serving_3plus_users": float(np.mean(active_region_user_count_arr >= 3.0)),
        "mean_effective_subcarriers_per_scheduled_user": float(np.mean(effective_subcarrier_arr)),
        "p5_effective_subcarriers_per_scheduled_user": float(np.percentile(effective_subcarrier_arr, 5.0)),
        "max_relative_region_power_budget_error": float(np.max(power_budget_error_arr)),
        # Backward-compatible adaptive fields retained for existing runners.
        "adaptive_max_users_per_region": float(DYNAMIC_MAX_USERS_PER_REGION),
        "adaptive_min_subcarriers_per_user": float(DYNAMIC_MIN_SUBCARRIERS_PER_USER),
        # Backward-compatible headline metrics remain scheduled-link metrics.
        "mean_sinr_db": float(10.0 * math.log10(float(np.mean(scheduled_sinr_arr)) + DEN_EPS)),
        "median_sinr_db": float(10.0 * math.log10(float(np.median(scheduled_sinr_arr)) + DEN_EPS)),
        "avg_rate_bpshz": float(np.mean(scheduled_rate_arr)),
        "p5_rate_bpshz": float(np.percentile(scheduled_rate_arr, 5.0)),
        # Corrected all-user/all-frame service metrics: unscheduled users count
        # as not served. These should be used for outage/continuity claims.
        "coverage_continuity_10db": float(np.mean(service_continuity[10.0])) if service_continuity[10.0] else 0.0,
        "coverage_continuity_20db": float(np.mean(service_continuity[20.0])) if service_continuity[20.0] else 0.0,
        "outage_probability_10db": float(1.0 - np.mean(service_continuity[10.0])) if service_continuity[10.0] else 1.0,
        "outage_probability_20db": float(1.0 - np.mean(service_continuity[20.0])) if service_continuity[20.0] else 1.0,
        "avg_outage_duration_frames": float(np.mean(service_duration_arr)),
        "p95_outage_duration_frames": float(np.percentile(service_duration_arr, 95.0)),
        # Additional diagnostic metrics for paper tables/appendices.
        "link_opportunity_mean_sinr_db": float(10.0 * math.log10(float(np.mean(all_link_sinr_arr)) + DEN_EPS)),
        "link_opportunity_median_sinr_db": float(10.0 * math.log10(float(np.median(all_link_sinr_arr)) + DEN_EPS)),
        "link_opportunity_avg_rate_bpshz": float(np.mean(all_link_rate_arr)),
        "link_opportunity_continuity_10db": float(np.mean(link_continuity[10.0])) if link_continuity[10.0] else 0.0,
        "link_opportunity_continuity_20db": float(np.mean(link_continuity[20.0])) if link_continuity[20.0] else 0.0,
        "link_opportunity_outage_probability_10db": float(1.0 - np.mean(link_continuity[10.0])) if link_continuity[10.0] else 1.0,
        "link_opportunity_outage_probability_20db": float(1.0 - np.mean(link_continuity[20.0])) if link_continuity[20.0] else 1.0,
        "link_opportunity_avg_outage_duration_20db": float(np.mean(link_duration_arr)),
        "scheduled_link_continuity_10db": float(np.mean(scheduled_continuity[10.0])) if scheduled_continuity[10.0] else 0.0,
        "scheduled_link_continuity_20db": float(np.mean(scheduled_continuity[20.0])) if scheduled_continuity[20.0] else 0.0,
        "service_avg_rate_bpshz_all_users": float(np.mean(service_rate_arr)),
        "beam_switching_rate": float(np.mean(switch_rates)) if switch_rates else 0.0,
        "fairness_over_time": float(np.mean(fairness_samples)) if fairness_samples else 0.0,
        "mean_served_fraction": float(np.mean(served_fraction)) if served_fraction else 0.0,
        # Service-regularity and starvation metrics. Lower is better for gap
        # and starvation metrics; higher is better for throughput and served ratios.
        "mean_user_throughput_bpshz": float(np.mean(user_service_rate_arr)),
        "p5_user_throughput_bpshz": float(np.percentile(user_service_rate_arr, 5.0)),
        "worst_user_throughput_bpshz": float(np.min(user_service_rate_arr)),
        "avg_time_between_services_frames": float(np.mean(inter_service_gap_arr)),
        "mean_service_gap_frames": float(np.mean(service_gap_arr)),
        "p95_service_gap_frames": float(np.percentile(service_gap_arr, 95.0)),
        "mean_longest_service_gap_frames": float(np.mean(longest_gap_arr)),
        "p95_longest_service_gap_frames": float(np.percentile(longest_gap_arr, 95.0)),
        "starvation_probability_gap10": float(np.mean(starvation_arr)),
        "edge_served_frame_ratio": float(np.mean(edge_served_fraction_arr)),
        "center_served_frame_ratio": float(np.mean(center_served_fraction_arr)),
        # Step-5 edge/center service metrics: use these to support claims
        # about reliability improvement where strict FFR matters most.
        "edge_coverage_continuity_10db": float(np.mean(edge_service_continuity[10.0])) if edge_service_continuity[10.0] else 0.0,
        "edge_coverage_continuity_20db": float(np.mean(edge_service_continuity[20.0])) if edge_service_continuity[20.0] else 0.0,
        "edge_outage_probability_10db": float(1.0 - np.mean(edge_service_continuity[10.0])) if edge_service_continuity[10.0] else 1.0,
        "edge_outage_probability_20db": float(1.0 - np.mean(edge_service_continuity[20.0])) if edge_service_continuity[20.0] else 1.0,
        "center_coverage_continuity_10db": float(np.mean(center_service_continuity[10.0])) if center_service_continuity[10.0] else 0.0,
        "center_coverage_continuity_20db": float(np.mean(center_service_continuity[20.0])) if center_service_continuity[20.0] else 0.0,
        "edge_frame_fraction": float(np.sum(edge_frame_counts) / max(1.0, np.sum(edge_frame_counts) + np.sum(center_frame_counts))),
        "time_s": float(time.time() - start),
    }


def experiment_cases() -> list[tuple[str, bool, bool, bool]]:
    return [
        ("static_los", False, False, False),
        ("mobility_only", True, False, False),
        ("orientation_only", False, True, False),
        ("nlos_only", False, False, True),
        ("mobility_orientation", True, True, False),
        ("combined_mobility_orientation_nlos", True, True, True),
    ]


def run_matrix(args: argparse.Namespace) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for case, mobility, orientation, nlos in experiment_cases():
        for scheduler in args.schedulers:
            row = simulate(
                trials=args.trials,
                frames=args.frames,
                users_per_macro=args.users_per_macro,
                scheduler=scheduler,
                include_mobility=mobility,
                include_orientation=orientation,
                include_nlos=nlos,
                seed=args.seed,
                nlos_profile=args.nlos_profile,
            )
            row["case"] = case
            rows.append(row)
            print(
                f"{case:35s} {scheduler:20s} "
                f"SINR={row['mean_sinr_db']:6.2f} dB "
                f"SE={row['avg_rate_bpshz']:7.2f} "
                f"C10={row['coverage_continuity_10db']:.3f} "
                f"J={row['fairness_over_time']:.3f}"
            )
    return rows


def run_ablation(args: argparse.Namespace) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for scheduler in ["proportional_fair", "omn_no_outage", "omn_no_edge", "omn_no_transition", "omn_no_nlos", "omn_no_fairness", "omn_no_gap", "omn_pf"]:
        row = simulate(
            trials=args.trials,
            frames=args.frames,
            users_per_macro=args.users_per_macro,
            scheduler=scheduler,
            include_mobility=True,
            include_orientation=True,
            include_nlos=True,
            seed=args.seed,
            nlos_profile=args.nlos_profile,
        )
        row["case"] = "ablation_combined"
        rows.append(row)
        print(
            f"{scheduler:20s} SE={row['avg_rate_bpshz']:7.2f} "
            f"C10={row['coverage_continuity_10db']:.3f} "
            f"Out10={row['outage_probability_10db']:.3f} "
            f"J={row['fairness_over_time']:.3f}"
        )
    return rows


def write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--users_per_macro", type=int, default=42)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mode", choices=["matrix", "ablation"], default="matrix")
    parser.add_argument("--nlos_profile", choices=["baseline", "stress"], default="baseline")
    parser.add_argument("--out", type=Path, default=Path("dynamic_extension/results_dynamic_v2/matrix.csv"))
    parser.add_argument(
        "--schedulers",
        nargs="+",
        default=["full-load", "rr1", "max_sinr", "proportional_fair", "omn_pf"],
    )
    args = parser.parse_args()
    rows = run_matrix(args) if args.mode == "matrix" else run_ablation(args)
    write_csv(rows, args.out)
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
