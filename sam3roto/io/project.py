from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

@dataclass
class PromptBundle:
    # points: (x, y, label) label 1 = +, 0 = -
    points: List[Tuple[int,int,int]] = field(default_factory=list)
    # boxes: (x1,y1,x2,y2,label) label 1 = positive box
    boxes: List[Tuple[int,int,int,int,int]] = field(default_factory=list)

@dataclass
class ObjectSpec:
    obj_id: int
    name: str = ""
    color: Tuple[int,int,int] = (0,255,0)
    visible: bool = True

@dataclass
class ProjectState:
    source_path: str = ""
    fps: float = 25.0
    model_path: str = "facebook/sam3"   # can be HF id or local path
    da3_model_id: str = "depth-anything/DA3-BASE"
    concept_text: str = "person"

    # objects and per-frame prompts
    objects: Dict[int, ObjectSpec] = field(default_factory=dict)
    prompts: Dict[int, Dict[int, PromptBundle]] = field(default_factory=dict)

def _to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dict__") and not isinstance(obj, dict):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    return obj

def save_project(path: Path, state: ProjectState) -> None:
    path = Path(path)
    data = _to_jsonable(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def load_project(path: Path) -> ProjectState:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    s = ProjectState()
    s.source_path = data.get("source_path","")
    s.fps = float(data.get("fps",25.0))
    s.model_path = data.get("model_path","facebook/sam3")
    s.da3_model_id = data.get("da3_model_id","depth-anything/DA3-BASE")
    s.concept_text = data.get("concept_text","person")

    objs = data.get("objects", {})
    for k, v in objs.items():
        oid = int(k)
        s.objects[oid] = ObjectSpec(
            obj_id=oid,
            name=v.get("name", f"Obj {oid}"),
            color=tuple(v.get("color",[0,255,0])),
            visible=bool(v.get("visible", True)),
        )

    pr = data.get("prompts", {})
    for frame_str, frame_data in pr.items():
        fi = int(frame_str)
        s.prompts[fi] = {}
        for obj_str, b in frame_data.items():
            oid = int(obj_str)
            s.prompts[fi][oid] = PromptBundle(
                points=[tuple(x) for x in b.get("points", [])],
                boxes=[tuple(x) for x in b.get("boxes", [])],
            )
    if not s.objects:
        s.objects[1] = ObjectSpec(obj_id=1, name="Obj 1", color=(0,255,0), visible=True)
    return s
