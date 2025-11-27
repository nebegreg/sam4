from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

def depth_to_normals(depth: np.ndarray, intrinsics: Optional[np.ndarray] = None) -> np.ndarray:
    d = depth.astype("float32")
    H, W = d.shape
    dzdx = np.zeros_like(d)
    dzdy = np.zeros_like(d)
    dzdx[:, 1:-1] = (d[:, 2:] - d[:, :-2]) * 0.5
    dzdy[1:-1, :] = (d[2:, :] - d[:-2, :]) * 0.5

    if intrinsics is not None:
        fx = float(intrinsics[0, 0])
        fy = float(intrinsics[1, 1])
        nx = -dzdx * fx
        ny = -dzdy * fy
        nz = 1.0
    else:
        nx = -dzdx
        ny = -dzdy
        nz = 1.0
    n = np.stack([nx, ny, np.full_like(d, nz)], axis=-1)
    n /= (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
    return n.astype("float32")

def normals_to_rgb8(normals: np.ndarray) -> np.ndarray:
    n = np.clip(normals, -1.0, 1.0)
    rgb = (n * 0.5 + 0.5) * 255.0
    return np.clip(rgb, 0, 255).astype("uint8")

def depth_to_point_cloud(depth: np.ndarray, intrinsics: np.ndarray, extrinsics: Optional[np.ndarray] = None, rgb: Optional[np.ndarray] = None, stride: int = 4) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    H, W = depth.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    zs = depth[0:H:stride, 0:W:stride]
    mask = zs > 0
    xs = xs[mask].astype("float32")
    ys = ys[mask].astype("float32")
    zs = zs[mask].astype("float32")

    fx = float(intrinsics[0,0]); fy = float(intrinsics[1,1])
    cx = float(intrinsics[0,2]); cy = float(intrinsics[1,2])

    Xc = (xs - cx) * zs / fx
    Yc = (ys - cy) * zs / fy
    Zc = zs
    pts_cam = np.stack([Xc, Yc, Zc], axis=-1)

    if extrinsics is not None:
        R = extrinsics[:, :3]
        t = extrinsics[:, 3]
        pts_world = (R.T @ (pts_cam.T - t.reshape(3,1))).T
    else:
        pts_world = pts_cam

    cols = None
    if rgb is not None:
        rgb_ds = rgb[0:H:stride, 0:W:stride].reshape(-1,3)
        cols = rgb_ds[mask.reshape(-1)].astype("uint8")
    return pts_world.astype("float32"), cols

def save_ply(path: Path, points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(points.shape[0])
    has_col = colors is not None
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_col:
        header += ["property uchar red","property uchar green","property uchar blue"]
    header += ["end_header"]

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        if has_col:
            c = colors.astype("uint8")
            for (x,y,z),(r,g,b) in zip(points, c):
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
        else:
            for (x,y,z) in points:
                f.write(f"{x} {y} {z}\n")
