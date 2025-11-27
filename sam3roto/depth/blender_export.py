from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional

def write_blender_export_script(out_dir: Path, camera_npz: Path, pointcloud_ply: Optional[Path] = None) -> Path:
    """Génère un script Blender *standalone* qui:
    - charge extrinsics/intrinsics depuis camera_da3.npz
    - crée une caméra animée
    - (optionnel) importe un PLY (point cloud)
    - export FBX ou Alembic via argument CLI --out_format
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    script = out_dir / "export_da3_to_fbx_or_abc.py"

    # Note: Blender Python (bpy) only available inside Blender.
    script.write_text(f"""import argparse, os
import numpy as np

import bpy
from mathutils import Matrix

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def load_camera_npz(path):
    data = np.load(path)
    extr = data['extrinsics']  # [N,3,4] world->cam (OpenCV/Colmap)
    intr = data['intrinsics']  # [N,3,3]
    return extr, intr

def opencv_w2c_to_blender_cam_matrix(w2c_3x4):
    # w2c: p_c = R p_w + t. Convert to c2w then to Blender matrix.
    R = w2c_3x4[:, :3]
    t = w2c_3x4[:, 3]
    c2w_R = R.T
    c2w_t = -c2w_R @ t.reshape(3,1)
    M = np.eye(4, dtype=np.float32)
    M[:3,:3] = c2w_R
    M[:3, 3] = c2w_t[:,0]

    # OpenCV camera coords: x right, y down, z forward.
    # Blender camera: x right, y up, z backward (camera looks -Z).
    # A common conversion:
    cv2_to_blender = np.array([
        [1, 0,  0, 0],
        [0,-1,  0, 0],
        [0, 0, -1, 0],
        [0, 0,  0, 1],
    ], dtype=np.float32)

    Mb = M @ cv2_to_blender
    return Matrix(Mb.tolist())

def create_camera(name="DA3_Camera"):
    cam_data = bpy.data.cameras.new(name)
    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    return cam_obj

def animate_camera(cam_obj, extrinsics, intrinsics, fps=25):
    scene = bpy.context.scene
    scene.render.fps = int(fps)

    # Use first intrinsics for focal; optional: per-frame intrinsics
    K0 = intrinsics[0]
    fx, fy, cx, cy = float(K0[0,0]), float(K0[1,1]), float(K0[0,2]), float(K0[1,2])

    # We can't know sensor size from DA3; we approximate with 36mm width and compute focal length in mm:
    # focal_mm = fx * sensor_width / image_width
    # We'll set lens later once we know image size (we assume principal point is near center).
    cam = cam_obj.data
    cam.sensor_width = 36.0

    for i, w2c in enumerate(extrinsics):
        frame = i + 1
        cam_obj.matrix_world = opencv_w2c_to_blender_cam_matrix(w2c)
        cam_obj.keyframe_insert(data_path="location", frame=frame)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=frame)

def import_pointcloud_ply(path):
    # Blender doesn't import PLY pointcloud as points by default (it imports as mesh).
    # We'll import as mesh; it's fine for export reference.
    bpy.ops.import_mesh.ply(filepath=path)

def export_scene(out_path, out_format):
    if out_format == "fbx":
        bpy.ops.export_scene.fbx(filepath=out_path, use_selection=False, add_leaf_bones=False)
    elif out_format == "abc":
        bpy.ops.wm.alembic_export(filepath=out_path, selected=False)
    else:
        raise ValueError("out_format must be fbx or abc")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_npz", required=True)
    parser.add_argument("--pointcloud_ply", default="")
    parser.add_argument("--out_format", default="abc", choices=["abc","fbx"])
    parser.add_argument("--out_path", default="da3_export.abc")
    parser.add_argument("--fps", default=25, type=float)
    args, _ = parser.parse_known_args()

    clear_scene()
    extr, intr = load_camera_npz(args.camera_npz)
    cam_obj = create_camera()
    animate_camera(cam_obj, extr, intr, fps=args.fps)

    if args.pointcloud_ply and os.path.exists(args.pointcloud_ply):
        import_pointcloud_ply(args.pointcloud_ply)

    export_scene(args.out_path, args.out_format)
    print("Export written:", args.out_path)

if __name__ == "__main__":
    main()
""", encoding="utf-8")

    return script
