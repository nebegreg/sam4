from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import cv2

class MaskCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, frame_idx: int, obj_id: int) -> Path:
        return self.root / f"obj_{obj_id:03d}" / f"alpha_{frame_idx:05d}.png"

    def write_alpha(self, frame_idx: int, obj_id: int, alpha_u8: np.ndarray) -> None:
        p = self._path(frame_idx, obj_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), alpha_u8.astype(np.uint8))

    def read_alpha(self, frame_idx: int, obj_id: int) -> Optional[np.ndarray]:
        p = self._path(frame_idx, obj_id)
        if not p.exists():
            return None
        a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if a is None:
            return None
        if a.ndim == 3:
            a = a[..., 0]
        return a.astype(np.uint8)

    def clear(self):
        if self.root.exists():
            for p in self.root.glob("obj_*"):
                if p.is_dir():
                    for f in p.glob("alpha_*.png"):
                        f.unlink(missing_ok=True)

class DepthCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _depth_path(self, frame_idx: int) -> Path:
        return self.root / f"depth_{frame_idx:05d}.npy"

    def write_depth(self, frame_idx: int, depth_f32) -> None:
        import numpy as np
        p = self._depth_path(frame_idx)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), np.asarray(depth_f32, dtype=np.float32))

    def read_depth(self, frame_idx: int):
        import numpy as np
        p = self._depth_path(frame_idx)
        if not p.exists():
            return None
        return np.load(str(p))

    def write_camera_npz(self, extrinsics, intrinsics) -> Path:
        import numpy as np
        out = self.root / "camera_da3.npz"
        np.savez_compressed(str(out), extrinsics=np.asarray(extrinsics, np.float32), intrinsics=np.asarray(intrinsics, np.float32))
        return out

    def read_camera_npz(self):
        import numpy as np
        p = self.root / "camera_da3.npz"
        if not p.exists():
            return None
        data = np.load(str(p))
        return {"extrinsics": data["extrinsics"], "intrinsics": data["intrinsics"]}

    def clear(self):
        for p in self.root.glob("depth_*.npy"):
            p.unlink(missing_ok=True)
        (self.root / "camera_da3.npz").unlink(missing_ok=True)
