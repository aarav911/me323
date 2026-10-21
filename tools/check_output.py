#!/usr/bin/env python3
"""
Independent check of a generated helmet: is the lattice confined to the inside
of the helmet (step 1), and is the bottom open so it can be put on (step 2)?

    python tools/check_output.py final_helmet_auxetic.stl fitted_head.stl fitted_helmet.stl
    python tools/check_output.py final_helmet_auxetic.stl fitted_head.stl fitted_helmet.stl --png slices.png

Get fitted_head.stl / fitted_helmet.stl from `me323 --export-fit`: they are the
exact meshes the generator received, so nothing here re-derives the fit. The
head keeps the lid marker of data/head.stl ("ME323-HEAD v1 lid-faces <first>
<count>" in its STL header), which says which triangles only close the scan.

Needs numpy, scipy, open3d (and matplotlib for --png). Deliberately uses
Open3D's ray tracer rather than libigl, so it does not check the C++ output
with the same code that produced it.

Every solid point of the output is sorted into:
    skin     within reach of the helmet shell surface
    liner    within reach of the head surface (the head-side liner)
    lattice  inside the helmet cavity (mapped outward from the head centre)
    OUTSIDE  none of the above: material outside the helmet.
Tolerances scale with the head size, matching the generator's own thicknesses
(skin <= 2.25 voxels, liner <= clearance + 2 voxels, voxel ~ head height / 120).

Step 1 (lattice only inside the helmet): OUTSIDE below 0.1% of all material.
Step 2 (bottom open, can be put on), each below 0.1% of all material:
    a. liner or lattice in directions where the head was not scanned (seen
       from the head centre, only its lid is there);
    b. liner in directions the helmet does not cover (seen from the head
       centre; a vent counts as covered, and under one the liner may stand
       above the bridged surface, as it must not be holed there);
    c. for the best straight lift-off direction (tilts -45..45 and sideways
       -10..10 degrees, 5-degree steps), material the head passes through as
       the helmet lifts off. Shell within the head clearance is a fit problem
       (the stock helmet sits too close), so it is reported apart.
"""

import argparse
import os
import sys

import numpy as np
import open3d as o3d
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy import ndimage

STL_TRIANGLE = np.dtype([("normal", "<f4", 3), ("corners", "<f4", (3, 3)), ("attribute", "<u2")])
LIMIT = 0.001


def scene(mesh):
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return s


def tensor(a):
    return o3d.core.Tensor(np.ascontiguousarray(a, dtype=np.float32))


def inside(s, P):
    """Inside test by majority over 3 rays. Open3D's one-ray parity (also
    behind compute_signed_distance) flips for points whose ray grazes an edge,
    e.g. points exactly in the plane of the head's extreme vertex."""
    return s.compute_occupancy(tensor(P), nsamples=3).numpy() > 0.5


def signed_distance(s, P):
    d = s.compute_distance(tensor(P)).numpy()
    return np.where(inside(s, P), -d, d)


def read_binary_stl(path):
    """Header bytes and the triangles in file order, (n, 3, 3). Read by hand:
    the lid marker refers to file order, which a mesh loader may not keep."""
    raw = open(path, "rb").read()
    n = int.from_bytes(raw[80:84], "little") if len(raw) >= 84 else -1
    if len(raw) != 84 + 50 * n:
        sys.exit(f"{path} is not a binary STL")
    return raw[:80], np.frombuffer(raw, STL_TRIANGLE, n, 84)["corners"].astype(float)


def head_lid(header, tri, path):
    """Per-triangle lid mask from the marker, or None for a complete head.
    Same checks as the generator: a fan around one point nothing else touches."""
    text = header.split(b"\0")[0].decode("ascii", "replace")
    if not text.startswith("ME323-HEAD"):
        return None
    parts = text.split()
    if len(parts) != 5 or parts[1] != "v1" or parts[2] != "lid-faces":
        sys.exit(f"{path}: unrecognised marker {text!r}")
    first, count = int(parts[3]), int(parts[4])
    if count < 3 or first < 0 or first + count > len(tri):
        sys.exit(f"{path}: marker names triangles past the end of the mesh")
    lid = tri[first:first + count]
    shared = [c for c in lid[0] if (lid == c).all(axis=2).any(axis=1).all()]
    if len(shared) != 1:
        sys.exit(f"{path}: the marked triangles are not a fan around one point")
    mask = np.zeros(len(tri), bool)
    mask[first:first + count] = True
    if (tri[~mask] == shared[0]).all(axis=2).any():
        sys.exit(f"{path}: an unmarked triangle touches the lid's apex")
    return mask


def soup_mesh(tri):
    """Triangle i of the mesh is triangle i of the file (Open3D ray hits report
    it as primitive id i)."""
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(tri.reshape(-1, 3)),
        o3d.utility.Vector3iVector(np.arange(3 * len(tri)).reshape(-1, 3)))


def direction_map(res_deg=0.5):
    """Cell centres of the direction map: polar angle from +Y, azimuth atan2(z, x)."""
    nt = int(round(180 / res_deg)); nph = 2 * nt; step = np.pi / nt
    th = (np.arange(nt) + 0.5) * step; ph = (np.arange(nph) + 0.5) * step
    TT, PP = np.meshgrid(th, ph, indexing="ij")
    D = np.stack([np.sin(TT) * np.cos(PP), np.cos(TT), np.sin(TT) * np.sin(PP)], -1).reshape(-1, 3)
    return nt, nph, step, TT, D


def first_hits(s, centre, D):
    return s.cast_rays(tensor(np.hstack([np.tile(centre, (len(D), 1)), D])))


def cavity_map(helmet_scene, centre, res_deg=0.5, max_vent=0.05, return_vents=False):
    """Independent re-implementation of HelmetCavity: radius map, vents filled
    (and, if asked, which cells were vents)."""
    nt, nph, step, TT, D = direction_map(res_deg)
    R = first_hits(helmet_scene, centre, D)["t_hit"].numpy().astype(float).reshape(nt, nph)
    R[~np.isfinite(R)] = np.nan

    def wrap(i, j):
        if i < 0:
            i, j = -i - 1, j + nph // 2
        elif i >= nt:
            i, j = 2 * nt - i - 1, j + nph // 2
        return i, j % nph

    miss = np.isnan(R)
    lab, nlab = ndimage.label(miss, structure=np.ones((3, 3)))
    # Azimuth wraps, and each pole row joins itself half a turn round. Merge
    # labels with union-find: relabelling pair by pair can point a later pair at
    # a label that was already merged away, splitting a hole that should be one.
    parent = list(range(nlab + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = [(lab[i, 0], lab[i, -1]) for i in range(nt)]
    pairs += [(lab[i, 0], lab[i + d, -1]) for i in range(nt) for d in (-1, 1) if 0 <= i + d < nt]
    pairs += [(lab[0, j], lab[0, (j + nph // 2 + d) % nph]) for j in range(nph) for d in (-1, 0, 1)]
    pairs += [(lab[-1, j], lab[-1, (j + nph // 2 + d) % nph]) for j in range(nph) for d in (-1, 0, 1)]
    for a, b in pairs:
        if a and b:
            parent[find(a)] = find(b)
    lab = np.vectorize(find)(lab) * (lab > 0)
    w = np.sin(TT)
    vent = np.zeros_like(miss)
    for k in np.unique(lab[lab > 0]):
        if w[lab == k].sum() / w.sum() <= max_vent:
            vent |= lab == k
    if vent.any():
        cells = np.argwhere(vent)
        idx = -np.ones((nt, nph), int)
        idx[vent] = np.arange(len(cells))
        rows, cols, vals, b = [], [], [], np.zeros(len(cells))
        for k, (i, j) in enumerate(cells):
            deg = 0.0
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ii, jj = wrap(i + di, j + dj)
                if vent[ii, jj]:
                    rows.append(k); cols.append(idx[ii, jj]); vals.append(-1.0); deg += 1
                elif not np.isnan(R[ii, jj]):
                    b[k] += R[ii, jj]; deg += 1
            rows.append(k); cols.append(k); vals.append(deg)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(len(cells),) * 2).tocsc()
        R[vent] = spl.spsolve(A, b)
    return (R, step, vent) if return_vents else (R, step)


def unscanned_map(head_scene, lid, centre, res_deg=0.5):
    """True where the ray from the centre leaves the head through its lid."""
    nt, nph, _, _, D = direction_map(res_deg)
    hit = first_hits(head_scene, centre, D)
    if not np.isfinite(hit["t_hit"].numpy()).all():
        sys.exit("a ray from the head centre misses the head: the head mesh has a hole")
    return lid[hit["primitive_ids"].numpy()].reshape(nt, nph)


def bilinear(shape, step, centre, P):
    """Distance from the centre, the four surrounding map cells and their weights."""
    nt, nph = shape
    d = P - centre
    r = np.linalg.norm(d, axis=1)
    u = d / np.maximum(r, 1e-300)[:, None]
    theta = np.arccos(np.clip(u[:, 1], -1, 1))
    phi = np.mod(np.arctan2(u[:, 2], u[:, 0]), 2 * np.pi)
    ti, pj = theta / step - 0.5, phi / step - 0.5
    i0, j0 = np.floor(ti).astype(int), np.floor(pj).astype(int)
    fi, fj = ti - i0, pj - j0

    def cell(i, j):
        i, j = i.copy(), j.copy()
        n = i < 0; i[n] = -i[n] - 1; j[n] += nph // 2
        s = i >= nt; i[s] = 2 * nt - i[s] - 1; j[s] += nph // 2
        return i, j % nph

    corners = [cell(i0, j0), cell(i0 + 1, j0), cell(i0, j0 + 1), cell(i0 + 1, j0 + 1)]
    weights = [(1 - fi) * (1 - fj), fi * (1 - fj), (1 - fi) * fj, fi * fj]
    return r, corners, weights


def lift_off_direction(tilt, side):
    t, q = np.radians(tilt), np.radians(side)
    d = np.array([np.sin(q) * np.cos(t), np.cos(q) * np.cos(t), np.sin(t)])
    return d / np.linalg.norm(d)


def describe(tilt, side):
    if tilt == 0 and side == 0:
        return "straight up"
    parts = []
    if tilt:
        parts.append(f"{abs(tilt)} deg toward " + ("-Z (back)" if tilt < 0 else "+Z (face)"))
    if side:
        parts.append(f"{abs(side)} deg toward " + ("-X" if side < 0 else "+X"))
    return "up, tilted " + " and ".join(parts)


def surface_spacing(v, fine_spacing):
    """The grid the generator takes the surface from: the coarse grid, or with
    --fine-spacing the coarse spacing / k, k the smallest integer giving a
    spacing <= the request (fine_grid.cpp make_fine_grid)."""
    if not fine_spacing or fine_spacing <= 0:
        return v
    return v / max(1.0, np.ceil(v / fine_spacing * (1.0 - 1e-12)))


def generator_reservation(Vh, Vm, shell="wall", skin_cells=0.5, fine_spacing=0.0):
    """lattice_generator.cpp's grid spacing for this head and helmet, and what it
    reserves between the head and the lattice: clearance (1 voxel) + liner
    (max(2 voxels, 0.5% of head height)) + the shell's inward part (band: half
    the skin, max(2.25 voxels, 1%)/2; wall: the skin, skin_cells cells of the
    surface grid, which is the fine grid with --fine-spacing).
    The grid coarsens as the helmet grows (4M-sample cap), so this is per fit.
    Returns (coarse spacing, reserved, surface spacing)."""
    H = np.ptp(Vh[:, 1])
    cmin, cmax = np.minimum(Vh.min(0), Vm.min(0)), np.maximum(Vh.max(0), Vm.max(0))
    v = H / 120.0

    def count(v):
        pad = 8.0 * v
        return np.prod(np.ceil((cmax - cmin + 2 * pad) / v) + 1)

    while count(v) > 4_000_000:
        v *= np.cbrt(count(v) / 4_000_000) * 1.01
    vs = surface_spacing(v, fine_spacing)
    inward = max(2.25 * v, 0.01 * H) / 2 if shell == "band" else skin_cells * vs
    return v, v + max(2.0 * v, 0.005 * H) + inward, vs


def zoom_fit(stock_vertices, Vh, gap=0.005):
    """main.cpp's original zoom fit of the stock helmet (head unscaled, unmoved)."""
    hmin, hmax = Vh.min(0), Vh.max(0)
    mmin, mmax = stock_vertices.min(0), stock_vertices.max(0)
    mc, hc = (mmin + mmax) / 2, (hmin + hmax) / 2
    s = (hmax[1] - hmin[1] + 2 * gap) / (mmax[1] - mmin[1])
    V = (stock_vertices - mc) * s
    V[:, 0] += hc[0]
    V[:, 2] += hc[2]
    V[:, 1] += hmax[1] + gap - (mmax[1] - mc[1]) * s
    return V


def front_edge(covered, el, az):
    """Lowest covered elevation per azimuth column over the front (azimuth 60..120 deg)."""
    out = np.full(covered.shape[1], np.nan)
    for j in np.where((az[0] >= 60) & (az[0] < 120))[0]:
        rows = np.where(covered[:, j])[0]
        if len(rows):
            out[j] = el[rows.max(), j]
    return out


def step3(T, lid, centre, helm, S_helm, R, U, step, args, in_way_skin):
    """Room for lattice along the head's outward normal, over a protection zone,
    and the step-3 decision gate (agreed 2026-10-15, see DECISIONS.md)."""
    Vh = T.reshape(-1, 3)
    Vm = np.asarray(helm.vertices)
    voxel, reserved, _ = generator_reservation(Vh, Vm, args.shell, args.shell_skin, args.fine_spacing)
    nt, nph, _, TT, D = direction_map(0.5)
    w = np.sin(TT)
    el = 90 - np.degrees(TT)
    az = (np.degrees(np.arctan2(D[:, 2], D[:, 0])).reshape(nt, nph) + 360) % 360
    S_head = scene(soup_mesh(T))
    rh = first_hits(S_head, centre, D)["t_hit"].numpy().reshape(nt, nph).astype(float)
    if args.zone_helmet:
        stock = o3d.io.read_triangle_mesh(args.zone_helmet)
        Rz, _ = cavity_map(scene(o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(zoom_fit(np.asarray(stock.vertices), Vh)), stock.triangles)), centre)
        zone = np.isfinite(Rz) & ~U
        edge0 = front_edge(np.isfinite(Rz), el, az)
    else:
        zone = np.isfinite(R) & ~U
        edge0 = None
    covered = np.isfinite(R)

    # samples on the scanned head surface, area-uniform (seed differs from the study's)
    Ts = T[~lid] if lid is not None else T
    A = 0.5 * np.linalg.norm(np.cross(Ts[:, 1] - Ts[:, 0], Ts[:, 2] - Ts[:, 0]), axis=1)
    rng = np.random.default_rng(7)
    n = 40000
    k = rng.choice(len(Ts), n, p=A / A.sum())
    a, b = rng.random(n), rng.random(n)
    f = a + b > 1
    a[f], b[f] = 1 - a[f], 1 - b[f]
    tri = Ts[k]
    P = tri[:, 0] + a[:, None] * (tri[:, 1] - tri[:, 0]) + b[:, None] * (tri[:, 2] - tri[:, 0])
    N = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    N /= np.linalg.norm(N, axis=1)[:, None]
    if (np.einsum("ij,ij->i", N, P - centre) > 0).mean() < 0.5:
        N = -N
    u = (P - centre) / np.linalg.norm(P - centre, axis=1)[:, None]
    ci = np.clip((np.arccos(np.clip(u[:, 1], -1, 1)) / step).astype(int), 0, nt - 1)
    cj = (np.mod(np.arctan2(u[:, 2], u[:, 0]), 2 * np.pi) / step).astype(int) % nph
    in_zone = zone[ci, cj]

    # which way the shell's inner face is wound, from the first hits seen from the centre
    Tm = Vm[np.asarray(helm.triangles)]
    Nm = np.cross(Tm[:, 1] - Tm[:, 0], Tm[:, 2] - Tm[:, 0])
    hit = first_hits(S_helm, centre, D)
    ok = np.isfinite(hit["t_hit"].numpy())
    facing = np.einsum("ij,ij->i", Nm[hit["primitive_ids"].numpy()[ok]], D[ok]) > 0
    shape = max(facing.mean(), 1 - facing.mean())
    inner_sign = 1.0 if facing.mean() >= 0.5 else -1.0

    h = S_helm.cast_rays(tensor(np.hstack([P + 1e-5 * N, N])))
    tn, pn = h["t_hit"].numpy().astype(float), h["primitive_ids"].numpy()
    valid = np.isfinite(tn)
    valid[valid] &= np.sign(np.einsum("ij,ij->i", Nm[pn[valid]], N[valid])) == inner_sign
    central = R - rh - reserved
    room = np.where(valid, tn - reserved, central[ci, cj])[in_zone]
    uncovered = np.isnan(room).mean()
    room = room[np.isfinite(room)]
    cover = (w * (zone & covered)).sum() / (w * zone).sum()
    edge = front_edge(covered, el, az)
    front = np.nan
    if edge0 is not None:
        both = np.isfinite(edge) & np.isfinite(edge0)
        front = (edge[both] - edge0[both]).min() if both.any() else np.nan
    ext = np.ptp(Vm, 0)
    q = {x: np.percentile(room, x) for x in (5, 50, 95)}
    mm = 1000.0 if np.ptp(Vh[:, 1]) < 2.0 else 1.0
    unit = "mm" if mm == 1000.0 else "units"
    elz = el[ci, cj][in_zone][np.isfinite(np.where(valid, tn, central[ci, cj])[in_zone])]
    regions = [np.median(room[sel]) * mm if sel.any() else np.nan
               for sel in (elz > 50, (elz > 20) & (elz <= 50), elz <= 20)]

    print(f"step 3 (room for lattice; {'protection zone from --zone-helmet' if args.zone_helmet else 'zone = what this helmet covers'}):")
    print(f"   generator grid {voxel*mm:.2f} {unit}, reserved for air + liner + half skin {reserved*mm:.2f} {unit}; "
          f"measured along the head's normal ({100*valid[in_zone].mean():.0f}% of points; the rest along the ray from the centre); "
          f"shell mode {args.shell}")
    print(f"   room: min {room.min()*mm:.2f}, p5 {q[5]*mm:.2f}, median {q[50]*mm:.2f}, p95 {q[95]*mm:.2f}, max {room.max()*mm:.2f} {unit}; "
          f"top / middle / low medians {regions[0]:.1f} / {regions[1]:.1f} / {regions[2]:.1f}")
    print(f"   helmet {ext[0]*mm:.1f} x {ext[2]*mm:.1f} {unit} (W x L); zone covered {100*cover:.2f}% "
          f"({100*uncovered:.2f}% of zone points uncovered); front edge vs today "
          + (f"{front:+.1f} deg" if np.isfinite(front) else "n/a"))
    checks = {
        f"room >= {args.room_target*mm:g} {unit} everywhere in the zone": room.min() >= args.room_target,
        "p95/p5 room <= 1.5": q[5] > 0 and q[95] / q[5] <= 1.5,
        f"size <= {args.max_size[0]*mm:g} x {args.max_size[1]*mm:g} {unit}": ext[0] <= args.max_size[0] and ext[2] <= args.max_size[1],
        "zone covered >= 99%": cover >= 0.99 and uncovered <= 0.01,
        "front edge not lower than today": not np.isfinite(front) or front >= -0.5,
        "shape: inner face first >= 99%": shape >= 0.99,
        "shell not in the way (best lift-off)": not in_way_skin,
    }
    for name, ok in checks.items():
        print(f"   {'PASS' if ok else 'FAIL'}  {name}")
    verdict = "PASS" if all(checks.values()) else "FAIL"
    failed = [name for name, ok in checks.items() if not ok]
    note = ""
    if failed == ["zone covered >= 99%"] and cover >= 0.985:
        note = (f" -- borderline: only coverage misses, by {100*(0.99-cover):.2f} points"
                " (the gate is ours; DECISIONS.md 2026-10-16 accepts such a fit)")
    print(f"step 3 gate: {verdict}{note}")
    return verdict


def exact_volume(mesh):
    """Enclosed volume of a closed mesh: sum of signed tetrahedra to the origin.
    (Not Open3D's get_volume(), which first runs a very slow self-intersection
    test.) Positive when the triangles face outward."""
    V = np.asarray(mesh.vertices)
    T = V[np.asarray(mesh.triangles)]
    return float(np.einsum("ij,ij->i", T[:, 0], np.cross(T[:, 1], T[:, 2])).sum() / 6.0)


def overhang_report(mesh, S_head, S_helm, centre, skin_tol, liner_tol, angle, H):
    """Downward-facing area an FDM print would have to bridge or support, for
    the two ways the walls can grow (along the helmet's up axis): crown-up
    (rim on the bed, build direction +Y) and crown-down (crown on the bed,
    build -Y). A face overhangs if it leans more than `angle` past vertical,
    i.e. its outward normal points down by more than `angle` below the
    horizontal (n . down > sin angle). Faces within 1 mm of the bed are not
    counted (they sit on it). Split by
    surface (centroid nearest to the shell: skin; to the head: liner; else
    lattice), and for skin and liner by side (normal toward the head centre or
    away), so the overhangs inside the part (the roof closing the lattice) show
    apart from those outside it (supportable). Up to 1M triangles, sampled by
    area, are classified."""
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    T = V[F]
    N = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    area = 0.5 * np.linalg.norm(N, axis=1)
    if exact_volume(mesh) < 0:
        N = -N
    good = area > 0
    rng = np.random.default_rng(11)
    idx = np.where(good)[0]
    if len(idx) > 1_000_000:
        idx = np.sort(rng.choice(idx, 1_000_000, replace=False, p=area[idx] / area[idx].sum()))
        weight = np.full(len(idx), area[good].sum() / len(idx))
    else:
        weight = area[idx]
    n = N[idx] / np.linalg.norm(N[idx], axis=1)[:, None]
    c = T[idx].mean(axis=1)
    d_skin = S_helm.compute_distance(tensor(c)).numpy()
    d_head = np.abs(signed_distance(S_head, c))
    toward = np.einsum("ij,ij->i", n, centre - c) > 0          # normal points toward the head centre
    is_skin = d_skin <= skin_tol
    is_liner = ~is_skin & (d_head <= liner_tol)
    classes = [("shell, outer face", is_skin & ~toward), ("shell, inner face (roof over the lattice)", is_skin & toward),
               ("liner, head side", is_liner & toward), ("liner, lattice side (roof)", is_liner & ~toward),
               ("lattice walls", ~is_skin & ~is_liner)]
    cos_limit = np.cos(np.radians(90.0 - angle))
    unit = "cm^2" if H < 2.0 else "units^2"
    to_unit = 1e4 if H < 2.0 else 1.0
    print(f"FDM overhangs (faces pointing down by more than {angle:g} deg from vertical walls; area, {unit}):")
    for name, up in (("crown-up (rim on the bed)", np.array([0.0, 1.0, 0.0])), ("crown-down (crown on the bed)", np.array([0.0, -1.0, 0.0]))):
        heights = V @ up
        bed = heights.min() + (0.001 if H < 2.0 else 1.0)
        down = (n @ up < -cos_limit) & (c @ up > bed)
        parts = ", ".join(f"{label} {(weight[down & m]).sum() * to_unit:.0f}" for label, m in classes)
        print(f"   {name}: {(weight[down]).sum() * to_unit:.0f} in all ({parts})")


def shell_checks(S_out, S_helm, centre, R, vents, H, v, vs, args):
    """Is the shell there and where it should be? Over the directions the helmet
    covers, away from vents and edges (one map cell in from them):
      continuity  a ray from far outside, aimed at the head centre, must first
                  meet the output near the stock helmet's outer face: a hole in
                  the shell would let it reach the lattice or liner much deeper.
                  (Rays from the centre would hit liner and lattice first, so
                  they cannot test the shell.)
      offset      how far the output's outer surface stands off the stock outer
                  face: ~ the skin (wall) or half the band thickness (band).
      wall        the middle of the stock wall must be solid in the output."""
    nt, nph, step, TT, D = direction_map(0.5)
    w = np.sin(TT)
    good = np.isfinite(R) & ~vents
    for di in (-1, 0, 1):                          # erode by one cell (azimuth wraps)
        for dj in (-1, 0, 1):
            shifted = np.roll(np.roll(good, di, 0), dj, 1)
            if di == -1:
                shifted[-1, :] = False
            elif di == 1:
                shifted[0, :] = False
            good &= shifted
    d = D[good.ravel()]
    far = 10.0 * H
    origin = centre + far * d
    t_helm = S_helm.cast_rays(tensor(np.hstack([origin, -d])))["t_hit"].numpy().astype(float)
    t_out = S_out.cast_rays(tensor(np.hstack([origin, -d])))["t_hit"].numpy().astype(float)
    ok = np.isfinite(t_helm)
    standoff = t_helm[ok] - t_out[ok]                # > 0: output stands outside the stock face
    # v: the generator's coarse grid spacing for this fit; vs: the surface grid's
    expected = (max(2.25 * v, 0.01 * H) / 2) if args.shell == "band" else args.shell_skin * vs
    deep = ~np.isfinite(standoff) | (standoff < -1.0 * v)   # met nothing, or met it well inside the stock face
    wd = w[good][ok]
    # middle of the stock wall along rays from the centre
    t1 = S_helm.cast_rays(tensor(np.hstack([np.tile(centre, (len(d), 1)), d])))["t_hit"].numpy().astype(float)
    k = np.isfinite(t1)
    t2 = S_helm.cast_rays(tensor(np.hstack([centre + (t1[k] + 1e-5)[:, None] * d[k], d[k]])))["t_hit"].numpy().astype(float)
    m = np.isfinite(t2)
    mid = centre + (t1[k][m] + t2[m] / 2)[:, None] * d[k][m]
    solid_mid = inside(S_out, mid)
    mm = 1000.0 if H < 2.0 else 1.0
    unit = "mm" if mm == 1000.0 else "units"
    print(f"shell ({args.shell}; {100*(w*good).sum()/w.sum():.1f}% of directions tested, vents and edges excluded):")
    print(f"   continuity: {100*(wd*deep).sum()/wd.sum():.3f}% of directions reach past the shell (holes)")
    print(f"   outer surface stands off the stock outer face by median {np.nanmedian(standoff)*mm:.2f} {unit} "
          f"(p5 {np.nanpercentile(standoff,5)*mm:.2f}, p95 {np.nanpercentile(standoff,95)*mm:.2f}; expected ~{expected*mm:.2f})")
    print(f"   middle of the stock wall solid: {100*solid_mid.mean():.3f}% of {len(mid)} points")
    ok_shell = (wd * deep).sum() / wd.sum() < 0.0005 and solid_mid.mean() >= 0.999
    print(f"step 3b (shell continuous, wall solid): {'PASS' if ok_shell else 'FAIL'}")
    return "PASS" if ok_shell else "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("fitted_head")
    ap.add_argument("fitted_helmet")
    ap.add_argument("--png", default=None, help="write side/front slice pictures here")
    ap.add_argument("--step3", action="store_true",
                    help="also report the room for lattice and the step-3 gate (counts in the exit status)")
    ap.add_argument("--zone-helmet", default=None,
                    help="the stock helmet (data/helmet.STL): defines the protection zone as what its "
                         "default zoom fit (Gap 5) covers, and the front edge to compare with")
    ap.add_argument("--room-target", type=float, default=0.020, help="step-3 room target, model units (0.02 = 20 mm)")
    ap.add_argument("--max-size", type=float, nargs=2, default=(0.245, 0.280), metavar=("W", "L"),
                    help="step-3 size limit, model units")
    ap.add_argument("--shell", choices=("wall", "band"), default="wall",
                    help="the generator's --shell (default wall), for the step-3 allowance and the shell checks")
    ap.add_argument("--shell-skin", type=float, default=0.5, help="the generator's --shell-skin, grid cells")
    ap.add_argument("--fine-spacing", type=float, default=0.0,
                    help="the generator's --fine-spacing (0 = off), for the wall skin in the allowance and shell checks")
    ap.add_argument("--overhangs", action="store_true",
                    help="report FDM overhang area for crown-up and crown-down printing")
    ap.add_argument("--overhang-angle", type=float, default=45.0,
                    help="overhang limit, degrees from vertical (default 45)")
    ap.add_argument("--density", type=float, default=1.0, help="g/cm^3 for the mass estimate (printed nylon ~0.95-1.0)")
    args = ap.parse_args()

    out = o3d.io.read_triangle_mesh(args.output)
    helm = o3d.io.read_triangle_mesh(args.fitted_helmet)
    for m in (out, helm):
        m.remove_duplicated_vertices()
    header, Th = read_binary_stl(args.fitted_head)
    lid = head_lid(header, Th, args.fitted_head)
    head = soup_mesh(Th)
    Vh = Th.reshape(-1, 3)
    centre = (Vh.min(0) + Vh.max(0)) / 2               # the generator's reference point
    H = np.ptp(Vh[:, 1])
    v = H / 120.0                                       # nominal generator voxel
    skin_tol, liner_tol, cav_tol = 3.0 * v, 5.0 * v, 1.0 * v

    S_out, S_head, S_helm = scene(out), scene(head), scene(helm)
    R, step, vents = cavity_map(S_helm, centre, return_vents=True)
    if lid is not None:
        if not inside(S_head, centre[None])[0]:
            sys.exit("the head centre is not inside the head")
        U = unscanned_map(S_head, lid, centre)

    Vo = np.asarray(out.vertices)
    g = 1.25 * v
    axes = [np.arange(Vo.min(0)[i] - g, Vo.max(0)[i] + g, g) for i in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    solid = inside(S_out, G)
    P = G[solid]
    d_skin = S_helm.compute_distance(tensor(P)).numpy()
    sd_head = signed_distance(S_head, P)
    d_head = np.abs(sd_head)
    r, corners, weights = bilinear(R.shape, step, centre, P)
    rin = sum(w * R[c] for c, w in zip(corners, weights))   # NaN where any corner is open
    in_cavity = np.isfinite(rin) & (r < rin + cav_tol)
    # Unscanned / uncovered only where all four surrounding cells are: the
    # generator stops at the first, so its own boundary layer never counts.
    unscanned = (np.logical_and.reduce([U[c] for c in corners]) if lid is not None
                 else np.zeros(len(P), bool))
    uncovered = np.logical_and.reduce([np.isnan(R[c]) for c in corners])

    is_skin = d_skin <= skin_tol
    is_liner = ~is_skin & (d_head <= liner_tol)
    is_lattice = ~is_skin & ~is_liner & in_cavity
    is_outside = ~is_skin & ~is_liner & ~in_cavity
    touching = is_skin & (sd_head < v)                 # shell inside the head clearance
    cell = g ** 3
    vol = lambda m: m.sum() * cell

    # Lift-off: rays from every judged point along each candidate direction.
    judged = np.where(~touching)[0]
    results = []
    for tilt in range(-45, 46, 5):
        for side in range(-10, 11, 5):
            d = lift_off_direction(tilt, side)
            t = S_head.cast_rays(tensor(np.hstack([P[judged], np.tile(d, (len(judged), 1))])))["t_hit"].numpy()
            blocked = np.zeros(len(P), bool)
            blocked[judged[np.isfinite(t)]] = True
            results.append((blocked.sum(), np.degrees(np.arccos(d[1])), tilt, side, blocked))
    best = min(results, key=lambda x: (x[0], x[1]))
    up = next(x for x in results if x[2] == 0 and x[3] == 0)
    in_way = best[4]

    labels, counts, _ = out.cluster_connected_triangles()
    counts = np.asarray(counts)
    tiny = int((counts < 0.001 * counts.max()).sum())

    unit = "L" if H < 2.0 else "model units^3"          # metres in, litres out
    k = 1e3 if H < 2.0 else 1.0
    total = vol(solid) if solid.any() else 1.0
    pct = lambda m: 100 * vol(m) / total
    print(f"output: {len(out.triangles):,} triangles, {len(counts)} connected pieces "
          f"({tiny} tiny fragments; largest piece {100 * counts.max() / counts.sum():.1f}% of triangles)")
    print(f"solid material, sampled every {g*1000 if H < 2 else g:.2f}{' mm' if H < 2 else ''}:")
    print(f"   helmet skin              {vol(is_skin)*k:8.3f} {unit}")
    print(f"   head-side liner          {vol(is_liner)*k:8.3f} {unit}")
    print(f"   lattice inside helmet    {vol(is_lattice)*k:8.3f} {unit}")
    print(f"   OUTSIDE the helmet       {vol(is_outside)*k:8.3f} {unit}")
    if H < 2.0:
        print(f"   total {vol(solid)*1e6:.0f} cm^3 -> about {vol(solid)*1e3*args.density:.2f} kg at {args.density:g} g/cm^3 "
              f"(skin {vol(is_skin)*1e6:.0f}, liner {vol(is_liner)*1e6:.0f}, lattice {vol(is_lattice)*1e6:.0f} cm^3)")
    volume = exact_volume(out)
    if H < 2.0:
        print(f"   exact mesh volume {abs(volume)*1e6:.0f} cm^3 -> {abs(volume)*1e3*args.density:.2f} kg at "
              f"{args.density:g} g/cm^3 (the split above is sampled; this total is exact)")
    else:
        print(f"   exact mesh volume {abs(volume):.4g} {unit}")
    if args.overhangs:
        overhang_report(out, S_head, S_helm, centre, skin_tol, liner_tol, args.overhang_angle, H)
    under_vent = np.logical_and.reduce([vents[c] for c in corners])
    print(f"   (liner under the vents   {vol(is_liner & under_vent)*k:8.3f} {unit}; vents are bridged, not open)")

    frac1 = vol(is_outside) / total
    v1 = "PASS" if frac1 < LIMIT else "FAIL"
    print(f"step 1 (lattice only inside the helmet): {v1} "
          f"-- {100*frac1:.3f}% of all material lies outside the helmet")

    a = ~is_skin & unscanned
    b = is_liner & uncovered
    ok = [vol(a) / total < LIMIT, vol(b) / total < LIMIT, vol(in_way) / total < LIMIT]
    v2 = "PASS" if all(ok) else "FAIL"
    print(f"step 2 (bottom open, can be put on): {v2}")
    if lid is None:
        print("   a. head has no lid marker: treated as complete, nothing unscanned")
    else:
        print(f"   a. liner/lattice where the head was not scanned:  {pct(a):6.3f}%  {'ok' if ok[0] else 'FAIL'}"
              f"   (shell there: {pct(is_skin & unscanned):.2f}%, allowed)")
    print(f"   b. liner where the helmet does not cover:         {pct(b):6.3f}%  {'ok' if ok[1] else 'FAIL'}")
    print(f"   c. in the head's way, best lift-off ({describe(best[2], best[3])}): "
          f"{pct(in_way):6.3f}%  {'ok' if ok[2] else 'FAIL'}"
          f"   (of it shell {pct(in_way & is_skin):.3f}%; straight up: {pct(up[4]):.2f}%)")
    if touching.any():
        amount = f"{vol(touching)*1e6:.2f} cm^3" if H < 2 else f"{vol(touching):.3g} {unit}"
        print(f"   note: shell within the head clearance (fit too tight there, not judged): {amount}")

    v_generator, _, v_surface = generator_reservation(Th.reshape(-1, 3), np.asarray(helm.vertices),
                                                      args.shell, args.shell_skin, args.fine_spacing)
    v3b = shell_checks(S_out, S_helm, centre, R, vents, H, v_generator, v_surface, args)
    v3 = None
    if args.step3:
        v3 = step3(Th, lid, centre, helm, S_helm, R, U if lid is not None else np.zeros_like(np.isfinite(R)),
                   step, args, bool((in_way & is_skin).any()))

    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        import matplotlib.patches as mpatches
        cls = np.zeros(len(G), int)
        where = np.where(solid)[0]
        cls[where[is_skin]] = 1
        cls[where[is_liner]] = 2
        cls[where[is_lattice]] = 3
        cls[where[is_outside]] = 4
        cls[where[a]] = 5
        cls[where[in_way]] = 6
        C = cls.reshape([len(x) for x in axes])
        cols = ["#ffffff", "#24426e", "#c3b8a8", "#f08c28", "#d7191c", "#7b3294", "#e7298a"]
        fig, axs = plt.subplots(1, 2, figsize=(14, 6.6))
        ix = np.argmin(np.abs(axes[0] - centre[0]))
        iz = np.argmin(np.abs(axes[2] - centre[2]))
        for ax, img, ext, title in (
                (axs[0], C[ix, :, :], (axes[2], axes[1]), "SIDE: front-to-back slice (face on the right)"),
                (axs[1], C[:, :, iz].T, (axes[0], axes[1]), "FRONT: side-to-side slice")):
            ax.imshow(img, origin="lower", cmap=ListedColormap(cols), vmin=0, vmax=6,
                      interpolation="nearest",
                      extent=[ext[0][0] * 1000, ext[0][-1] * 1000, ext[1][0] * 1000, ext[1][-1] * 1000])
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("mm")
            ax.set_aspect("equal")
        fig.legend(handles=[mpatches.Patch(color=cols[1], label="helmet skin"),
                            mpatches.Patch(color=cols[2], label="head-side liner"),
                            mpatches.Patch(color=cols[3], label="lattice inside the helmet"),
                            mpatches.Patch(color=cols[4], label="material OUTSIDE the helmet"),
                            mpatches.Patch(color=cols[5], label="liner/lattice where head not scanned"),
                            mpatches.Patch(color=cols[6], label="in the head's way lifting off")],
                   loc="lower center", ncol=3, frameon=False)
        fig.suptitle(f"{os.path.basename(args.output)}: step 1 {v1}, step 2 {v2}", fontsize=12)
        plt.tight_layout(rect=(0, 0.09, 1, 0.96))
        plt.savefig(args.png, dpi=110)
        print(f"slices written to {args.png}")
    return 0 if v1 == "PASS" and v2 == "PASS" and v3b == "PASS" and v3 in (None, "PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
