from __future__ import annotations
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

from .backend.sam3_backend import SAM3Backend
from .depth.da3_backend import DepthAnything3Backend
from .depth.geometry import depth_to_normals, normals_to_rgb8, depth_to_point_cloud, save_ply
from .depth.blender_export import write_blender_export_script
from .io.media import load_video, load_image_sequence
from .io.cache import MaskCache, DepthCache
from .io.project import ProjectState, ObjectSpec, PromptBundle, save_project, load_project
from .io.export import save_png_gray, save_png_rgba, try_export_prores4444_from_png_sequence
from .post.matte import fill_small_holes, remove_small_dots, grow_shrink, border_fix, feather_alpha, alpha_from_trimap, temporal_smooth
from .post.despill import estimate_bg_color, despill_green_average, despill_blue_average, despill_physical, luminance_restore
from .post.pixelspread import pixel_spread_rgb
from .post.composite import premultiply
from .post.flowblur import edge_motion_blur_alpha
from .post.advanced_matting import refine_alpha_for_hair, multi_scale_refinement, edge_aware_smoothing
from .post.matting_presets import get_preset, list_presets, PRESETS
from .ui.viewer import Viewer
from .ui.widgets import LabeledSlider

def pil_to_rgb_u8(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"), dtype=np.uint8)

def qpix_from_rgb(rgb_u8: np.ndarray) -> QtGui.QPixmap:
    h, w = rgb_u8.shape[:2]
    qimg = QtGui.QImage(rgb_u8.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888).copy()
    return QtGui.QPixmap.fromImage(qimg)

def overlay_rgba_from_alpha(alpha_u8: np.ndarray, color: Tuple[int,int,int]) -> QtGui.QImage:
    h,w = alpha_u8.shape[:2]
    rgba = np.zeros((h,w,4), np.uint8)
    rgba[...,0] = color[0]
    rgba[...,1] = color[1]
    rgba[...,2] = color[2]
    rgba[...,3] = alpha_u8
    qimg = QtGui.QImage(rgba.data, w, h, 4*w, QtGui.QImage.Format.Format_RGBA8888).copy()
    return qimg

def blend_overlays(alphas: List[Tuple[np.ndarray, Tuple[int,int,int]]]) -> QtGui.QImage:
    if not alphas:
        return QtGui.QImage()
    h,w = alphas[0][0].shape[:2]
    out = np.zeros((h,w,4), np.float32)
    for a_u8, col in alphas:
        a = (a_u8.astype(np.float32)/255.0)[...,None]
        c = (np.array(col, np.float32)/255.0).reshape(1,1,3)
        src = np.concatenate([np.tile(c,(h,w,1)), a], axis=-1)
        src_p = src.copy(); src_p[...,:3] *= src_p[...,3:4]
        dst_p = out.copy(); dst_p[...,:3] *= dst_p[...,3:4]
        out_p = src_p + dst_p*(1.0 - src_p[...,3:4])
        out = out_p
    # unpremultiply for display
    a = np.clip(out[...,3:4], 1e-6, 1.0)
    rgb = np.clip(out[...,:3]/a, 0, 1)
    rgba = np.concatenate([rgb, out[...,3:4]], axis=-1)
    rgba8 = np.clip(rgba*255.0,0,255).astype(np.uint8)
    return QtGui.QImage(rgba8.data, w, h, 4*w, QtGui.QImage.Format.Format_RGBA8888).copy()

class Worker(QtCore.QObject):
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
    @QtCore.Slot()
    def run(self):
        try:
            self.finished.emit(self.fn(*self.args, **self.kwargs))
        except Exception:
            self.failed.emit(traceback.format_exc())

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM3 + DA3 Roto Ultimate PRO v0.4")
        self.resize(1640, 980)

        self.sam3 = SAM3Backend()
        self.da3 = DepthAnything3Backend(Path.cwd() / ".sam3roto_cache" / "da3_tmp")

        self.state = ProjectState()
        self.frames: List[Image.Image] = []
        self.fps: float = 25.0
        self.frame_idx: int = 0
        self.active_obj: int = 1

        self.mask_cache = MaskCache(Path.cwd() / ".sam3roto_cache" / "masks")
        self.depth_cache = DepthCache(Path.cwd() / ".sam3roto_cache" / "depth")

        self._build_ui()
        self._ensure_obj(1)
        self._refresh_obj_list(select=1)

    # ---------------- UI ----------------
    def _build_ui(self):
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        main = QtWidgets.QHBoxLayout(cw)

        self.viewer = Viewer()
        main.addWidget(self.viewer, 4)

        right = QtWidgets.QWidget()
        main.addWidget(right, 2)
        r = QtWidgets.QVBoxLayout(right)
        r.setContentsMargins(10,10,10,10)

        src_row = QtWidgets.QHBoxLayout()
        self.btn_video = QtWidgets.QPushButton("📼 Import vidéo")
        self.btn_seq = QtWidgets.QPushButton("🖼️ Import suite")
        src_row.addWidget(self.btn_video)
        src_row.addWidget(self.btn_seq)
        r.addLayout(src_row)

        self.le_sam3 = QtWidgets.QLineEdit(self.state.model_path)
        self.le_sam3.setPlaceholderText("SAM3 model id (facebook/sam3) ou chemin local")
        r.addWidget(self.le_sam3)
        self.btn_sam3_load = QtWidgets.QPushButton("⚙️ Charger SAM3")
        r.addWidget(self.btn_sam3_load)

        r.addWidget(QtWidgets.QLabel("Objets :"))
        self.obj_list = QtWidgets.QListWidget()
        self.obj_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        r.addWidget(self.obj_list, 1)

        obj_row = QtWidgets.QHBoxLayout()
        self.btn_add_obj = QtWidgets.QPushButton("＋")
        self.btn_del_obj = QtWidgets.QPushButton("－")
        self.btn_clear_prompts = QtWidgets.QPushButton("Clear prompts(frame,obj)")
        obj_row.addWidget(self.btn_add_obj)
        obj_row.addWidget(self.btn_del_obj)
        obj_row.addWidget(self.btn_clear_prompts)
        r.addLayout(obj_row)

        self.tabs = QtWidgets.QTabWidget()
        r.addWidget(self.tabs, 4)

        # --- Seg/Track tab ---
        t1 = QtWidgets.QWidget(); l1 = QtWidgets.QVBoxLayout(t1)

        self.mode = QtWidgets.QComboBox()
        self.mode.addItems([
            "Concept (PCS) image",
            "Interactive (PVS) image",
            "Concept (PCS) video (track all instances)",
            "Interactive (PVS) video (keyframes)",
        ])
        l1.addWidget(self.mode)

        self.le_text = QtWidgets.QLineEdit(self.state.concept_text)
        self.le_text.setPlaceholderText("Concept text (ex: person, hard hat, red dress…)")
        l1.addWidget(self.le_text)

        tool_row = QtWidgets.QHBoxLayout()
        self.tool = QtWidgets.QComboBox(); self.tool.addItems(["Point","+/-", "Box"])
        self.sign = QtWidgets.QComboBox(); self.sign.addItems(["+", "-"])
        tool_row.addWidget(QtWidgets.QLabel("Tool"))
        tool_row.addWidget(self.tool, 1)
        tool_row.addWidget(QtWidgets.QLabel("Sign"))
        tool_row.addWidget(self.sign, 0)
        l1.addLayout(tool_row)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_segment = QtWidgets.QPushButton("▶ Segment frame")
        self.btn_track = QtWidgets.QPushButton("🧷 Track (video)")
        btn_row.addWidget(self.btn_segment)
        btn_row.addWidget(self.btn_track)
        l1.addLayout(btn_row)

        self.lbl_status = QtWidgets.QLabel("Prêt.")
        self.lbl_status.setWordWrap(True)
        l1.addWidget(self.lbl_status)
        self.tabs.addTab(t1, "Seg/Track")

        # --- Matte tab ---
        t2 = QtWidgets.QWidget(); l2 = QtWidgets.QVBoxLayout(t2)

        # Presets
        preset_group = QtWidgets.QGroupBox("Matting Presets")
        preset_layout = QtWidgets.QVBoxLayout(preset_group)

        preset_row = QtWidgets.QHBoxLayout()
        preset_row.addWidget(QtWidgets.QLabel("Preset:"))
        self.cb_preset = QtWidgets.QComboBox()
        self.cb_preset.addItem("Custom", "custom")
        for preset_key, preset in PRESETS.items():
            self.cb_preset.addItem(preset.name, preset_key)
        preset_row.addWidget(self.cb_preset, 1)
        preset_layout.addLayout(preset_row)

        self.lbl_preset_desc = QtWidgets.QLabel("")
        self.lbl_preset_desc.setWordWrap(True)
        self.lbl_preset_desc.setStyleSheet("color: #888; font-size: 10px;")
        preset_layout.addWidget(self.lbl_preset_desc)

        l2.addWidget(preset_group)

        # Advanced matting options
        adv_group = QtWidgets.QGroupBox("Advanced Matting")
        adv_layout = QtWidgets.QVBoxLayout(adv_group)

        self.chk_advanced_matting = QtWidgets.QCheckBox("Enable advanced matting (for hair/fur)")
        self.chk_advanced_matting.setChecked(True)
        adv_layout.addWidget(self.chk_advanced_matting)

        self.cb_advanced_mode = QtWidgets.QComboBox()
        self.cb_advanced_mode.addItems(["Guided Filter", "Trimap Refinement", "Both (Best Quality)"])
        adv_layout.addWidget(self.cb_advanced_mode)

        self.chk_multi_scale = QtWidgets.QCheckBox("Multi-scale refinement (slower, better quality)")
        adv_layout.addWidget(self.chk_multi_scale)

        l2.addWidget(adv_group)

        # Standard matte controls
        self.sl_grow = LabeledSlider("Grow/Shrink", -20, 20, 0, " px")
        self.sl_holes = LabeledSlider("Fill holes area", 0, 8000, 300, "")
        self.sl_dots = LabeledSlider("Remove dots area", 0, 8000, 200, "")
        self.sl_border = LabeledSlider("Border fix", 0, 25, 2, " px")
        self.sl_feather = LabeledSlider("Feather", 0, 40, 4, " px")
        self.sl_trimap = LabeledSlider("Trimap band", 1, 60, 14, " px")
        self.sl_temporal = LabeledSlider("Temporal smooth", 0, 100, 60, " %")
        for w in [self.sl_grow, self.sl_holes, self.sl_dots, self.sl_border, self.sl_feather, self.sl_trimap, self.sl_temporal]:
            l2.addWidget(w)
        self.chk_trimap = QtWidgets.QCheckBox("Raffiner alpha (trimap distance)")
        self.chk_trimap.setChecked(True)
        l2.addWidget(self.chk_trimap)

        self.chk_flowblur = QtWidgets.QCheckBox("Edge motion blur (optical flow)")
        self.chk_flowblur.setChecked(False)
        l2.addWidget(self.chk_flowblur)
        self.sl_flow = LabeledSlider("Flow blur strength", 0, 100, 35, " %")
        l2.addWidget(self.sl_flow)

        self.btn_preview = QtWidgets.QPushButton("👁️ Preview overlays (visible objs)")
        l2.addWidget(self.btn_preview)
        self.tabs.addTab(t2, "Matte")

        # --- RGB tab ---
        t3 = QtWidgets.QWidget(); l3 = QtWidgets.QVBoxLayout(t3)
        self.chk_despill = QtWidgets.QCheckBox("Despill")
        self.chk_despill.setChecked(True)
        l3.addWidget(self.chk_despill)

        self.cb_despill = QtWidgets.QComboBox()
        self.cb_despill.addItems(["Green average", "Blue average", "Physical (auto BG)"])
        l3.addWidget(self.cb_despill)
        self.sl_despill = LabeledSlider("Despill strength", 0, 100, 85, " %")
        l3.addWidget(self.sl_despill)
        self.chk_luma = QtWidgets.QCheckBox("Luminance restore")
        self.chk_luma.setChecked(True)
        l3.addWidget(self.chk_luma)

        self.chk_spread = QtWidgets.QCheckBox("Edge extend / Pixel spread")
        self.chk_spread.setChecked(True)
        l3.addWidget(self.chk_spread)
        self.sl_spread = LabeledSlider("Spread radius", 0, 40, 10, " px")
        l3.addWidget(self.sl_spread)

        self.chk_premult = QtWidgets.QCheckBox("Exporter premultiplied (sinon straight)")
        self.chk_premult.setChecked(False)
        l3.addWidget(self.chk_premult)
        self.tabs.addTab(t3, "RGB / Comp")

        # --- Depth tab (DA3) ---
        t4 = QtWidgets.QWidget(); l4 = QtWidgets.QVBoxLayout(t4)
        self.le_da3 = QtWidgets.QLineEdit(self.state.da3_model_id)
        l4.addWidget(QtWidgets.QLabel("DA3 model id:"))
        l4.addWidget(self.le_da3)

        row = QtWidgets.QHBoxLayout()
        self.btn_da3_load = QtWidgets.QPushButton("⚙️ Charger DA3")
        self.btn_da3_run = QtWidgets.QPushButton("🌊 Depth+Camera (all frames)")
        row.addWidget(self.btn_da3_load)
        row.addWidget(self.btn_da3_run)
        l4.addLayout(row)

        self.btn_da3_prev_depth = QtWidgets.QPushButton("👁️ Preview depth (false color)")
        self.btn_da3_prev_norm = QtWidgets.QPushButton("👁️ Preview normals")
        l4.addWidget(self.btn_da3_prev_depth)
        l4.addWidget(self.btn_da3_prev_norm)

        self.le_da3_out = QtWidgets.QLineEdit(str(Path.cwd() / "exports" / "depth_anything3"))
        l4.addWidget(QtWidgets.QLabel("Export folder (DA3):"))
        l4.addWidget(self.le_da3_out)

        self.btn_da3_exp_depth = QtWidgets.QPushButton("Export depth PNG16 seq")
        self.btn_da3_exp_norm = QtWidgets.QPushButton("Export normals PNG seq")
        self.btn_da3_exp_cam = QtWidgets.QPushButton("Export camera_npz")
        self.btn_da3_exp_ply = QtWidgets.QPushButton("Export global pointcloud PLY")
        self.btn_da3_blender = QtWidgets.QPushButton("Generate Blender export script (FBX/Alembic)")
        for b in [self.btn_da3_exp_depth, self.btn_da3_exp_norm, self.btn_da3_exp_cam, self.btn_da3_exp_ply, self.btn_da3_blender]:
            l4.addWidget(b)

        l4.addStretch(1)
        self.tabs.addTab(t4, "Depth / Camera (DA3)")

        # --- Export tab ---
        t5 = QtWidgets.QWidget(); l5 = QtWidgets.QVBoxLayout(t5)
        self.le_export = QtWidgets.QLineEdit(str(Path.cwd() / "exports"))
        l5.addWidget(QtWidgets.QLabel("Export folder (SAM3):"))
        l5.addWidget(self.le_export)
        self.btn_exp_alpha = QtWidgets.QPushButton("Export alpha PNG (obj actif)")
        self.btn_exp_rgba = QtWidgets.QPushButton("Export RGBA PNG (obj actif)")
        self.btn_exp_all_alpha = QtWidgets.QPushButton("Export alpha ALL objs")
        self.btn_exp_all_rgba = QtWidgets.QPushButton("Export RGBA ALL objs")
        self.btn_exp_prores = QtWidgets.QPushButton("Export ProRes4444 MOV (obj actif, ffmpeg)")
        for b in [self.btn_exp_alpha, self.btn_exp_rgba, self.btn_exp_all_alpha, self.btn_exp_all_rgba, self.btn_exp_prores]:
            l5.addWidget(b)
        self.tabs.addTab(t5, "Export")

        # timeline
        bottom = QtWidgets.QHBoxLayout()
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.lbl_frame = QtWidgets.QLabel("Frame: -/-")
        bottom.addWidget(self.slider, 1)
        bottom.addWidget(self.lbl_frame, 0)
        r.addLayout(bottom)

        # menu
        menu = self.menuBar().addMenu("File")
        act_save = menu.addAction("Save project…")
        act_load = menu.addAction("Load project…")

        # connections
        self.btn_video.clicked.connect(self.on_load_video)
        self.btn_seq.clicked.connect(self.on_load_seq)
        self.btn_sam3_load.clicked.connect(self.on_load_sam3)

        self.btn_add_obj.clicked.connect(self.on_add_obj)
        self.btn_del_obj.clicked.connect(self.on_del_obj)
        self.btn_clear_prompts.clicked.connect(self.on_clear_prompts)
        self.obj_list.itemChanged.connect(self.on_obj_visibility_changed)
        self.obj_list.itemSelectionChanged.connect(self.on_obj_selected)

        self.tool.currentTextChanged.connect(self.on_tool_changed)
        self.sign.currentTextChanged.connect(self.on_sign_changed)
        self.viewer.pointAdded.connect(self.on_point_added)
        self.viewer.boxAdded.connect(self.on_box_added)

        self.btn_segment.clicked.connect(self.on_segment_frame)
        self.btn_track.clicked.connect(self.on_track_video)
        self.btn_preview.clicked.connect(self.on_preview_overlay)

        self.btn_da3_load.clicked.connect(self.on_da3_load)
        self.btn_da3_run.clicked.connect(self.on_da3_run_all)
        self.btn_da3_prev_depth.clicked.connect(self.on_da3_preview_depth)
        self.btn_da3_prev_norm.clicked.connect(self.on_da3_preview_normals)
        self.btn_da3_exp_depth.clicked.connect(self.on_da3_export_depth)
        self.btn_da3_exp_norm.clicked.connect(self.on_da3_export_normals)
        self.btn_da3_exp_cam.clicked.connect(self.on_da3_export_camera)
        self.btn_da3_exp_ply.clicked.connect(self.on_da3_export_ply)
        self.btn_da3_blender.clicked.connect(self.on_da3_generate_blender)

        self.btn_exp_alpha.clicked.connect(self.on_export_alpha)
        self.btn_exp_rgba.clicked.connect(self.on_export_rgba)
        self.btn_exp_all_alpha.clicked.connect(self.on_export_all_alpha)
        self.btn_exp_all_rgba.clicked.connect(self.on_export_all_rgba)
        self.btn_exp_prores.clicked.connect(self.on_export_prores)

        self.slider.valueChanged.connect(self.on_seek)
        act_save.triggered.connect(self.on_save_project)
        act_load.triggered.connect(self.on_load_project)

        self.cb_preset.currentIndexChanged.connect(self.on_preset_changed)

        self.on_tool_changed(self.tool.currentText())
        self.on_sign_changed(self.sign.currentText())

        # shortcuts
        QtGui.QShortcut(QtGui.QKeySequence("["), self, activated=self.prev_frame)
        QtGui.QShortcut(QtGui.QKeySequence("]"), self, activated=self.next_frame)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Enter"), self, activated=self.on_segment_frame)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+T"), self, activated=self.on_track_video)

    def _set_status(self, s: str):
        self.lbl_status.setText(s)

    def on_preset_changed(self, index: int):
        """Charge un preset et met à jour tous les sliders"""
        preset_key = self.cb_preset.currentData()
        if preset_key == "custom" or preset_key is None:
            self.lbl_preset_desc.setText("Paramètres personnalisés")
            return

        preset = get_preset(preset_key)
        self.lbl_preset_desc.setText(preset.description)

        # Update sliders
        self.sl_grow.setValue(preset.grow_shrink)
        self.sl_holes.setValue(preset.fill_holes)
        self.sl_dots.setValue(preset.remove_dots)
        self.sl_border.setValue(preset.border_fix)
        self.sl_feather.setValue(int(preset.feather))
        self.sl_trimap.setValue(preset.trimap_band)
        self.sl_temporal.setValue(int(preset.temporal_smooth * 100))

        # Update checkboxes
        self.chk_trimap.setChecked(preset.use_trimap)
        self.chk_advanced_matting.setChecked(preset.use_advanced_matting)

        # Update advanced options
        mode_map = {"guided": 0, "trimap": 1, "both": 2}
        self.cb_advanced_mode.setCurrentIndex(mode_map.get(preset.advanced_mode, 0))
        self.chk_multi_scale.setChecked(preset.multi_scale)

        # Update RGB settings
        self.cb_despill.setCurrentIndex(preset.despill_mode)
        self.sl_despill.setValue(int(preset.despill_strength * 100))
        self.chk_luma.setChecked(preset.use_luminance_restore)
        self.sl_spread.setValue(int(preset.pixel_spread))

        self._set_status(f"✅ Preset chargé: {preset.name}")

    def prev_frame(self):
        if self.frames:
            self.slider.setValue(max(0, self.frame_idx-1))

    def next_frame(self):
        if self.frames:
            self.slider.setValue(min(len(self.frames)-1, self.frame_idx+1))

    # -------- objects / prompts --------
    def _ensure_obj(self, obj_id: int):
        if obj_id not in self.state.objects:
            palette = [(0,255,0),(255,0,0),(0,128,255),(255,0,255),(255,128,0),(0,255,200),(200,255,0)]
            self.state.objects[obj_id] = ObjectSpec(obj_id=obj_id, name=f"Obj {obj_id}", color=palette[(obj_id-1)%len(palette)], visible=True)

    def _bundle(self, frame_idx: int, obj_id: int) -> PromptBundle:
        self.state.prompts.setdefault(frame_idx, {})
        if obj_id not in self.state.prompts[frame_idx]:
            self.state.prompts[frame_idx][obj_id] = PromptBundle()
        return self.state.prompts[frame_idx][obj_id]

    def _refresh_obj_list(self, select: Optional[int] = None):
        self.obj_list.blockSignals(True)
        self.obj_list.clear()
        for oid in sorted(self.state.objects.keys()):
            o = self.state.objects[oid]
            it = QtWidgets.QListWidgetItem(f"{oid}: {o.name}")
            it.setData(QtCore.Qt.ItemDataRole.UserRole, oid)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsSelectable)
            it.setCheckState(QtCore.Qt.CheckState.Checked if o.visible else QtCore.Qt.CheckState.Unchecked)
            it.setForeground(QtGui.QColor(*o.color))
            self.obj_list.addItem(it)
        if select is None:
            select = self.active_obj
        for i in range(self.obj_list.count()):
            if int(self.obj_list.item(i).data(QtCore.Qt.ItemDataRole.UserRole)) == int(select):
                self.obj_list.setCurrentRow(i)
                break
        self.obj_list.blockSignals(False)

    def on_add_obj(self):
        oid = 1
        while oid in self.state.objects:
            oid += 1
        self._ensure_obj(oid)
        self.active_obj = oid
        self._refresh_obj_list(select=oid)
        self._set_status(f"Objet ajouté: {oid}")

    def on_del_obj(self):
        if len(self.state.objects) <= 1:
            return
        if self.active_obj in self.state.objects:
            del self.state.objects[self.active_obj]
        for f in list(self.state.prompts.keys()):
            self.state.prompts[f].pop(self.active_obj, None)
            if not self.state.prompts[f]:
                del self.state.prompts[f]
        self.active_obj = sorted(self.state.objects.keys())[0]
        self._refresh_obj_list(select=self.active_obj)
        self.on_preview_overlay()

    def on_clear_prompts(self):
        if self.frame_idx in self.state.prompts and self.active_obj in self.state.prompts[self.frame_idx]:
            self.state.prompts[self.frame_idx][self.active_obj] = PromptBundle()
            self._set_status("Prompts effacés (frame,obj).")
        self.on_preview_overlay()

    def on_obj_visibility_changed(self, item: QtWidgets.QListWidgetItem):
        oid = int(item.data(QtCore.Qt.ItemDataRole.UserRole))
        self._ensure_obj(oid)
        self.state.objects[oid].visible = item.checkState() == QtCore.Qt.CheckState.Checked
        self.on_preview_overlay()

    def on_obj_selected(self):
        items = self.obj_list.selectedItems()
        if items:
            self.active_obj = int(items[0].data(QtCore.Qt.ItemDataRole.UserRole))
            self.on_preview_overlay()

    def on_tool_changed(self, t: str):
        if t.lower().startswith("box"):
            self.viewer.set_mode("box")
        else:
            self.viewer.set_mode("point")

    def on_sign_changed(self, s: str):
        lbl = 1 if s == "+" else 0
        self.viewer.set_point_label(lbl)
        self.viewer.set_box_label(lbl)

    def on_point_added(self, x: int, y: int, label: int):
        self._ensure_obj(self.active_obj)
        self._bundle(self.frame_idx, self.active_obj).points.append((x,y,label))
        self._set_status(f"Point @ frame {self.frame_idx} obj {self.active_obj} ({'+' if label else '-'})")

    def on_box_added(self, x1: int, y1: int, x2: int, y2: int, label: int):
        self._ensure_obj(self.active_obj)
        self._bundle(self.frame_idx, self.active_obj).boxes.append((x1,y1,x2,y2,label))
        self._set_status(f"Box @ frame {self.frame_idx} obj {self.active_obj} ({'+' if label else '-'})")

    # -------- loading media/model --------
    def on_load_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choisir vidéo", "", "Video (*.mov *.mp4 *.mkv *.avi)")
        if not path:
            return
        self._set_status("Chargement vidéo…")
        self._run_thread(lambda: load_video(path), tag="LOAD_MEDIA")

    def on_load_seq(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Choisir dossier suite")
        if not folder:
            return
        self._set_status("Chargement suite…")
        self._run_thread(lambda: load_image_sequence(folder, fps=self.fps), tag="LOAD_MEDIA")

    def _apply_media(self, media):
        self.state.source_path = media.path
        self.frames = media.frames
        self.fps = media.fps
        self.state.fps = float(self.fps)
        self.frame_idx = 0
        self.slider.setEnabled(True)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, len(self.frames)-1))
        self.slider.setValue(0)
        self.mask_cache.clear()
        self.depth_cache.clear()
        self._show_frame(0)
        self._set_status(f"Source chargée: {media.path} ({len(self.frames)} frames @ {self.fps:.2f}fps)")

    def on_load_sam3(self):
        mid = self.le_sam3.text().strip() or "facebook/sam3"
        self.state.model_path = mid
        self._set_status("Chargement SAM3…")
        self._run_thread(lambda: self.sam3.load(mid), tag="SAM3_LOADED")

    # -------- timeline --------
    def on_seek(self, v: int):
        self._show_frame(int(v))

    def _show_frame(self, idx: int):
        if not self.frames:
            return
        self.frame_idx = int(np.clip(idx,0,len(self.frames)-1))
        self.lbl_frame.setText(f"Frame: {self.frame_idx+1}/{len(self.frames)}")
        rgb = pil_to_rgb_u8(self.frames[self.frame_idx])
        self.viewer.set_image(qpix_from_rgb(rgb))
        self.on_preview_overlay()

    # -------- matte + rgb processing --------
    def _post_alpha(self, mask_u8: np.ndarray, prev_alpha: Optional[np.ndarray], prev_rgb: Optional[np.ndarray], cur_rgb: Optional[np.ndarray]) -> np.ndarray:
        m = mask_u8.astype(np.uint8)
        m = fill_small_holes(m, self.sl_holes.value())
        m = remove_small_dots(m, self.sl_dots.value())
        m = grow_shrink(m, self.sl_grow.value())
        m = border_fix(m, self.sl_border.value())

        if self.chk_trimap.isChecked():
            a = alpha_from_trimap(m, band=self.sl_trimap.value())
        else:
            a = feather_alpha(m, radius=float(self.sl_feather.value()))

        # Advanced matting pour les détails fins (cheveux, fur)
        if self.chk_advanced_matting.isChecked() and cur_rgb is not None:
            mode_idx = self.cb_advanced_mode.currentIndex()
            mode_map = {0: "guided", 1: "trimap", 2: "both"}
            mode = mode_map.get(mode_idx, "guided")

            if self.chk_multi_scale.isChecked():
                # Multi-scale pour capturer différents niveaux de détails
                a = multi_scale_refinement(cur_rgb, a, scales=[1.0, 0.5, 0.25])
            else:
                # Raffinement simple
                a = refine_alpha_for_hair(cur_rgb, a, mode=mode, radius=8, eps=1e-5)

        if self.chk_flowblur.isChecked() and prev_rgb is not None and cur_rgb is not None and prev_alpha is not None:
            a = edge_motion_blur_alpha(prev_rgb, cur_rgb, a, strength=self.sl_flow.value()/100.0, samples=6)

        if prev_alpha is not None:
            a = temporal_smooth(prev_alpha, a, strength=self.sl_temporal.value()/100.0)

        return a

    def _process_rgb(self, rgb_u8: np.ndarray, alpha_u8: np.ndarray) -> np.ndarray:
        out = rgb_u8
        if self.chk_despill.isChecked():
            strength = self.sl_despill.value()/100.0
            mode = self.cb_despill.currentIndex()
            if mode == 0:
                out = despill_green_average(out, strength=strength)
            elif mode == 1:
                out = despill_blue_average(out, strength=strength)
            else:
                bg = estimate_bg_color(out, alpha_u8)
                phys = despill_physical(out, alpha_u8, bg_rgb01=bg, edge_only=True)
                out = (out.astype(np.float32)*(1.0-strength) + phys.astype(np.float32)*strength).astype(np.uint8)
            if self.chk_luma.isChecked():
                out = luminance_restore(rgb_u8, out, amount=1.0)
        if self.chk_spread.isChecked():
            out = pixel_spread_rgb(out, alpha_u8, radius=float(self.sl_spread.value()))
        return out

    # -------- preview overlays --------
    def on_preview_overlay(self):
        if not self.frames:
            return
        overlays = []
        for oid, spec in self.state.objects.items():
            if not spec.visible:
                continue
            a = self.mask_cache.read_alpha(self.frame_idx, oid)
            if a is None:
                continue
            overlays.append((a, spec.color))
        qimg = blend_overlays(overlays)
        if not qimg.isNull():
            self.viewer.set_overlay(qimg, opacity=0.55)
        else:
            self.viewer.clear_overlay()

    # -------- SAM3 operations --------
    def on_segment_frame(self):
        if not self.frames:
            return
        if not self.sam3.is_ready():
            QtWidgets.QMessageBox.warning(self, "SAM3", "Charge SAM3 d'abord.")
            return
        mode = self.mode.currentIndex()
        if mode == 0:
            text = self.le_text.text().strip() or self.state.concept_text
            self.state.concept_text = text
            self._set_status("PCS image…")
            self._run_thread(lambda: self._job_pcs_image(text), tag="SEG_DONE")
        elif mode == 1:
            bun = self._bundle(self.frame_idx, self.active_obj)
            if not bun.points and not bun.boxes:
                QtWidgets.QMessageBox.warning(self, "PVS", "Ajoute des points (+/-) ou une box.")
                return
            self._set_status("PVS image…")
            self._run_thread(lambda: self._job_pvs_image(bun.points, bun.boxes), tag="SEG_DONE")
        else:
            QtWidgets.QMessageBox.information(self, "Mode", "Utilise le bouton Track pour les modes vidéo.")

    def _job_pcs_image(self, text: str):
        img = self.frames[self.frame_idx]
        masks_by_id = self.sam3.segment_concept_image(img, text=text)
        # create objects per returned id
        prev_alpha_by_id = {}
        for oid, m in masks_by_id.items():
            self._ensure_obj(int(oid))
            prev_a = self.mask_cache.read_alpha(self.frame_idx-1, int(oid)) if self.frame_idx>0 else None
            prev_rgb = pil_to_rgb_u8(self.frames[self.frame_idx-1]) if self.frame_idx>0 else None
            cur_rgb = pil_to_rgb_u8(img)
            a = self._post_alpha(m, prev_a, prev_rgb, cur_rgb)
            self.mask_cache.write_alpha(self.frame_idx, int(oid), a)
            prev_alpha_by_id[int(oid)] = a
        return True

    def _job_pvs_image(self, points, boxes):
        img = self.frames[self.frame_idx]
        m = self.sam3.segment_interactive_image(img, points=points, boxes=boxes, multimask=False)
        prev_a = self.mask_cache.read_alpha(self.frame_idx-1, self.active_obj) if self.frame_idx>0 else None
        prev_rgb = pil_to_rgb_u8(self.frames[self.frame_idx-1]) if self.frame_idx>0 else None
        cur_rgb = pil_to_rgb_u8(img)
        a = self._post_alpha(m, prev_a, prev_rgb, cur_rgb)
        self.mask_cache.write_alpha(self.frame_idx, self.active_obj, a)
        return True

    def on_track_video(self):
        if not self.frames:
            return
        if not self.sam3.is_ready():
            QtWidgets.QMessageBox.warning(self, "SAM3", "Charge SAM3 d'abord.")
            return
        mode = self.mode.currentIndex()
        if mode == 2:
            text = self.le_text.text().strip() or self.state.concept_text
            self.state.concept_text = text
            self._set_status("PCS video tracking…")
            self._run_thread(lambda: self._job_pcs_video([text]), tag="TRACK_DONE")
        elif mode == 3:
            # prompts: frame -> obj -> points (+ use box center as extra point)
            prompts: Dict[int, Dict[int, List[Tuple[int,int,int]]]] = {}
            for f, objs in self.state.prompts.items():
                for oid, bun in objs.items():
                    pts = list(bun.points)
                    for (x1,y1,x2,y2,label) in bun.boxes:
                        pts.append((int((x1+x2)/2), int((y1+y2)/2), int(label)))
                    if pts:
                        prompts.setdefault(int(f), {})[int(oid)] = pts
            if not prompts:
                QtWidgets.QMessageBox.warning(self, "Prompts", "Ajoute des keyframes (points/box) d'abord.")
                return
            self._set_status("PVS video tracking…")
            self._run_thread(lambda: self._job_pvs_video(prompts), tag="TRACK_DONE")
        else:
            QtWidgets.QMessageBox.information(self, "Mode", "Choisis un mode vidéo dans la liste.")

    def _job_pcs_video(self, texts: List[str]):
        prev_alpha = {}  # obj -> alpha
        prev_rgb_u8 = None
        for fm in self.sam3.track_concept_video(self.frames, texts=texts):
            cur_rgb_u8 = pil_to_rgb_u8(self.frames[fm.frame_idx])
            for oid, m in fm.masks_by_id.items():
                self._ensure_obj(int(oid))
                a_prev = prev_alpha.get(int(oid))
                a = self._post_alpha(m, a_prev, prev_rgb_u8, cur_rgb_u8)
                self.mask_cache.write_alpha(fm.frame_idx, int(oid), a)
                prev_alpha[int(oid)] = a
            prev_rgb_u8 = cur_rgb_u8
        return True

    def _job_pvs_video(self, prompts: Dict[int, Dict[int, List[Tuple[int,int,int]]]]):
        prev_alpha = {}
        prev_rgb_u8 = None
        for fm in self.sam3.track_interactive_video(self.frames, prompts=prompts):
            cur_rgb_u8 = pil_to_rgb_u8(self.frames[fm.frame_idx])
            for oid, m in fm.masks_by_id.items():
                self._ensure_obj(int(oid))
                a_prev = prev_alpha.get(int(oid))
                a = self._post_alpha(m, a_prev, prev_rgb_u8, cur_rgb_u8)
                self.mask_cache.write_alpha(fm.frame_idx, int(oid), a)
                prev_alpha[int(oid)] = a
            prev_rgb_u8 = cur_rgb_u8
        return True

    # -------- DA3 operations --------
    def on_da3_load(self):
        mid = self.le_da3.text().strip() or "depth-anything/DA3-BASE"
        self.state.da3_model_id = mid
        self._set_status("Chargement DA3…")
        self._run_thread(lambda: self.da3.load(mid), tag="DA3_LOADED")

    def on_da3_run_all(self):
        if not self.frames:
            return
        if not self.da3.is_ready():
            QtWidgets.QMessageBox.warning(self, "DA3", "Charge DA3 d'abord.")
            return
        self._set_status("DA3 infer all frames…")
        self._run_thread(lambda: self._job_da3_infer_all(), tag="DA3_DONE")

    def _job_da3_infer_all(self):
        pred = self.da3.infer(self.frames, indices=list(range(len(self.frames))))
        for j, fi in enumerate(pred.indices):
            self.depth_cache.write_depth(fi, pred.depth[j])
        if pred.extrinsics is not None and pred.intrinsics is not None:
            self.depth_cache.write_camera_npz(pred.extrinsics, pred.intrinsics)
        return True

    def _da3_falsecolor(self, depth: np.ndarray) -> np.ndarray:
        d = depth.astype(np.float32)
        valid = d[d>0]
        vmin = float(np.percentile(valid, 2.0)) if valid.size else float(d.min())
        vmax = float(np.percentile(valid, 98.0)) if valid.size else float(d.max())
        if vmax <= vmin:
            vmax = vmin + 1.0
        x = np.clip((d - vmin)/(vmax-vmin), 0.0, 1.0)
        u8 = (x*255.0).astype(np.uint8)
        col = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
        return cv2.cvtColor(col, cv2.COLOR_BGR2RGB)

    def on_da3_preview_depth(self):
        d = self.depth_cache.read_depth(self.frame_idx)
        if d is None:
            QtWidgets.QMessageBox.information(self,"DA3","Aucune depth en cache. Lance 'Depth+Camera'.")
            return
        rgb = self._da3_falsecolor(d)
        qimg = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3*rgb.shape[1], QtGui.QImage.Format.Format_RGB888).copy()
        self.viewer.set_overlay(qimg, opacity=0.7)

    def on_da3_preview_normals(self):
        d = self.depth_cache.read_depth(self.frame_idx)
        cam = self.depth_cache.read_camera_npz()
        if d is None or cam is None:
            QtWidgets.QMessageBox.information(self,"DA3","Depth et/ou camera manque. Lance 'Depth+Camera'.")
            return
        K = cam["intrinsics"][self.frame_idx]
        n = depth_to_normals(d, intrinsics=K)
        rgb = normals_to_rgb8(n)
        qimg = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3*rgb.shape[1], QtGui.QImage.Format.Format_RGB888).copy()
        self.viewer.set_overlay(qimg, opacity=0.7)

    def on_da3_export_depth(self):
        out = Path(self.le_da3_out.text().strip() or "exports/depth_anything3") / "depth_png16"
        self._set_status("Export depth PNG16…")
        self._run_thread(lambda: self._job_da3_export_depth(out), tag="DA3_EXPORT")

    def _job_da3_export_depth(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        # global normalization across sequence for coherent z-depth
        depths = []
        for i in range(len(self.frames)):
            d = self.depth_cache.read_depth(i)
            if d is not None:
                depths.append(d[d>0])
        if depths:
            allv = np.concatenate(depths)
            vmin = float(np.percentile(allv, 0.5))
            vmax = float(np.percentile(allv, 99.5))
            if vmax <= vmin:
                vmax = vmin + 1.0
        else:
            vmin, vmax = 0.0, 1.0

        for i in range(len(self.frames)):
            d = self.depth_cache.read_depth(i)
            if d is None:
                continue
            x = np.clip((d - vmin)/(vmax-vmin), 0.0, 1.0)
            d16 = (x*65535.0).astype(np.uint16)
            cv2.imwrite(str(out_dir / f"depth_{i:05d}.png"), d16)
        return True

    def on_da3_export_normals(self):
        out = Path(self.le_da3_out.text().strip() or "exports/depth_anything3") / "normals_png"
        self._set_status("Export normals…")
        self._run_thread(lambda: self._job_da3_export_normals(out), tag="DA3_EXPORT")

    def _job_da3_export_normals(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        cam = self.depth_cache.read_camera_npz()
        if cam is None:
            return False
        extr = cam["extrinsics"]; intr = cam["intrinsics"]
        for i in range(len(self.frames)):
            d = self.depth_cache.read_depth(i)
            if d is None:
                continue
            n = depth_to_normals(d, intrinsics=intr[i])
            rgb = normals_to_rgb8(n)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"normals_{i:05d}.png"), bgr)
        return True

    def on_da3_export_camera(self):
        out = Path(self.le_da3_out.text().strip() or "exports/depth_anything3")
        self._set_status("Export camera_npz…")
        self._run_thread(lambda: self._job_da3_export_camera(out), tag="DA3_EXPORT")

    def _job_da3_export_camera(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        cam = self.depth_cache.read_camera_npz()
        if cam is None:
            return False
        # already stored in cache; copy to exports
        npz_path = out_dir / "camera_da3.npz"
        np.savez_compressed(str(npz_path), extrinsics=cam["extrinsics"], intrinsics=cam["intrinsics"])
        return True

    def on_da3_export_ply(self):
        out = Path(self.le_da3_out.text().strip() or "exports/depth_anything3")
        ply = out / "pointcloud_da3_global.ply"
        self._set_status("Export pointcloud PLY…")
        self._run_thread(lambda: self._job_da3_export_ply(ply), tag="DA3_EXPORT")

    def _job_da3_export_ply(self, ply_path: Path):
        cam = self.depth_cache.read_camera_npz()
        if cam is None:
            return False
        extr = cam["extrinsics"]; intr = cam["intrinsics"]
        pts_all = []
        cols_all = []
        stride = 4
        for i in range(len(self.frames)):
            d = self.depth_cache.read_depth(i)
            if d is None:
                continue
            rgb = pil_to_rgb_u8(self.frames[i])
            pts, cols = depth_to_point_cloud(d, intr[i], extrinsics=extr[i], rgb=rgb, stride=stride)
            if pts.size == 0:
                continue
            pts_all.append(pts)
            if cols is not None:
                cols_all.append(cols)
        if not pts_all:
            return False
        P = np.concatenate(pts_all, axis=0)
        C = np.concatenate(cols_all, axis=0) if cols_all else None
        save_ply(ply_path, P, C)
        return True

    def on_da3_generate_blender(self):
        out = Path(self.le_da3_out.text().strip() or "exports/depth_anything3") / "blender_export"
        cam_cache = self.depth_cache.read_camera_npz()
        if cam_cache is None:
            QtWidgets.QMessageBox.warning(self,"Blender export","Aucune camera_npz. Lance 'Depth+Camera' puis export caméra.")
            return
        out.mkdir(parents=True, exist_ok=True)
        npz = out / "camera_da3.npz"
        np.savez_compressed(str(npz), extrinsics=cam_cache["extrinsics"], intrinsics=cam_cache["intrinsics"])
        ply = Path(self.le_da3_out.text().strip() or "exports/depth_anything3") / "pointcloud_da3_global.ply"
        script = write_blender_export_script(out, camera_npz=npz, pointcloud_ply=ply if ply.exists() else None)
        QtWidgets.QMessageBox.information(self, "Blender export", f"Script généré:\n{script}\n\nEx:\nblender -b -P {script} -- --out_format abc")

    # -------- Exports SAM3 --------
    def _iter_alpha(self, obj_id: int):
        h,w = (self.frames[0].height, self.frames[0].width) if self.frames else (0,0)
        for i in range(len(self.frames)):
            a = self.mask_cache.read_alpha(i, obj_id)
            if a is None:
                a = np.zeros((h,w), np.uint8)
            yield i, a

    def on_export_alpha(self):
        if not self.frames:
            return
        out = Path(self.le_export.text().strip() or "exports") / f"alpha_obj{self.active_obj:03d}"
        self._set_status("Export alpha…")
        self._run_thread(lambda: self._job_export_alpha(out, self.active_obj), tag="EXPORT_DONE")

    def _job_export_alpha(self, out_dir: Path, obj_id: int):
        for i,a in self._iter_alpha(obj_id):
            save_png_gray(out_dir / f"alpha_{i:05d}.png", a)
        return True

    def on_export_rgba(self):
        if not self.frames:
            return
        out = Path(self.le_export.text().strip() or "exports") / f"rgba_obj{self.active_obj:03d}"
        self._set_status("Export RGBA…")
        self._run_thread(lambda: self._job_export_rgba(out, self.active_obj), tag="EXPORT_DONE")

    def _job_export_rgba(self, out_dir: Path, obj_id: int):
        for i,a in self._iter_alpha(obj_id):
            rgb = pil_to_rgb_u8(self.frames[i])
            rgb2 = self._process_rgb(rgb, a)
            if self.chk_premult.isChecked():
                rgb2 = premultiply(rgb2, a)
            rgba = np.dstack([rgb2, a]).astype(np.uint8)
            save_png_rgba(out_dir / f"rgba_{i:05d}.png", rgba)
        return True

    def on_export_all_alpha(self):
        if not self.frames:
            return
        base = Path(self.le_export.text().strip() or "exports") / "alpha_all"
        self._set_status("Export alpha ALL…")
        self._run_thread(lambda: self._job_export_all_alpha(base), tag="EXPORT_DONE")

    def _job_export_all_alpha(self, base: Path):
        for oid in sorted(self.state.objects.keys()):
            out = base / f"obj{oid:03d}"
            for i,a in self._iter_alpha(oid):
                save_png_gray(out / f"alpha_{i:05d}.png", a)
        return True

    def on_export_all_rgba(self):
        if not self.frames:
            return
        base = Path(self.le_export.text().strip() or "exports") / "rgba_all"
        self._set_status("Export RGBA ALL…")
        self._run_thread(lambda: self._job_export_all_rgba(base), tag="EXPORT_DONE")

    def _job_export_all_rgba(self, base: Path):
        for oid in sorted(self.state.objects.keys()):
            out = base / f"obj{oid:03d}"
            for i,a in self._iter_alpha(oid):
                rgb = pil_to_rgb_u8(self.frames[i])
                rgb2 = self._process_rgb(rgb, a)
                if self.chk_premult.isChecked():
                    rgb2 = premultiply(rgb2, a)
                rgba = np.dstack([rgb2, a]).astype(np.uint8)
                save_png_rgba(out / f"rgba_{i:05d}.png", rgba)
        return True

    def on_export_prores(self):
        if not self.frames:
            return
        base = Path(self.le_export.text().strip() or "exports")
        rgba_dir = base / f"rgba_obj{self.active_obj:03d}"
        mov = base / f"obj{self.active_obj:03d}_prores4444.mov"
        self._set_status("Export ProRes4444…")
        self._run_thread(lambda: self._job_export_prores(rgba_dir, mov), tag="EXPORT_DONE")

    def _job_export_prores(self, rgba_dir: Path, mov: Path):
        rgba_dir.mkdir(parents=True, exist_ok=True)
        # ensure frames exist
        for i,a in self._iter_alpha(self.active_obj):
            out = rgba_dir / f"rgba_{i:05d}.png"
            if out.exists():
                continue
            rgb = pil_to_rgb_u8(self.frames[i])
            rgb2 = self._process_rgb(rgb, a)
            if self.chk_premult.isChecked():
                rgb2 = premultiply(rgb2, a)
            rgba = np.dstack([rgb2, a]).astype(np.uint8)
            save_png_rgba(out, rgba)
        ok = try_export_prores4444_from_png_sequence(str(rgba_dir / "rgba_%05d.png"), mov, fps=self.fps)
        return ok

    # -------- project IO --------
    def on_save_project(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save project", "", "SAM3Roto (*.sam3roto.json)")
        if not path:
            return
        self.state.model_path = self.le_sam3.text().strip() or self.state.model_path
        self.state.concept_text = self.le_text.text().strip() or self.state.concept_text
        self.state.da3_model_id = self.le_da3.text().strip() or self.state.da3_model_id
        save_project(Path(path), self.state)
        self._set_status(f"Projet sauvé: {path}")

    def on_load_project(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load project", "", "SAM3Roto (*.sam3roto.json)")
        if not path:
            return
        self.state = load_project(Path(path))
        self.le_sam3.setText(self.state.model_path)
        self.le_text.setText(self.state.concept_text)
        self.le_da3.setText(self.state.da3_model_id)
        self.active_obj = sorted(self.state.objects.keys())[0]
        self._refresh_obj_list(select=self.active_obj)
        self._set_status(f"Projet chargé: {path}")
        self.on_preview_overlay()

    # -------- threading --------
    def _run_thread(self, fn, tag: str):
        th = QtCore.QThread(self)
        wk = Worker(fn)
        wk.moveToThread(th)
        th.started.connect(wk.run)

        def done(res):
            th.quit(); th.wait()
            if tag == "LOAD_MEDIA":
                self._apply_media(res)
            elif tag == "SAM3_LOADED":
                self._set_status("✅ SAM3 chargé.")
            elif tag == "SEG_DONE":
                self._set_status("✅ Segmentation OK (cache).")
                self._refresh_obj_list(select=self.active_obj)
                self.on_preview_overlay()
            elif tag == "TRACK_DONE":
                self._set_status("✅ Tracking OK (cache).")
                self._refresh_obj_list(select=self.active_obj)
                self.on_preview_overlay()
            elif tag == "DA3_LOADED":
                self._set_status("✅ DA3 chargé.")
            elif tag == "DA3_DONE":
                self._set_status("✅ DA3 terminé (depth+camera en cache).")
            elif tag == "DA3_EXPORT":
                self._set_status("✅ Export DA3 terminé.")
            elif tag == "EXPORT_DONE":
                self._set_status("✅ Export terminé.")
            else:
                self._set_status("✅ Terminé.")
            wk.deleteLater(); th.deleteLater()

        def fail(err):
            th.quit(); th.wait()
            QtWidgets.QMessageBox.critical(self, "Erreur", err)
            self._set_status("❌ Erreur.")
            wk.deleteLater(); th.deleteLater()

        wk.finished.connect(done)
        wk.failed.connect(fail)
        th.start()

def main():
    app = QtWidgets.QApplication([])
    app.setApplicationName("SAM3+DA3 Roto Ultimate")
    w = MainWindow()
    w.show()
    app.exec()
