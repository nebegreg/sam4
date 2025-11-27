"""
Presets de matting optimisés pour différents types de matériaux
Basés sur les meilleures pratiques de l'industrie VFX
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class MattingPreset:
    """Preset de paramètres pour le matting"""
    name: str
    description: str

    # Matte refinement
    grow_shrink: int
    fill_holes: int
    remove_dots: int
    border_fix: int
    feather: float
    trimap_band: int
    use_trimap: bool
    temporal_smooth: float

    # Advanced matting
    use_advanced_matting: bool
    advanced_mode: str  # "guided", "trimap", "both"
    guided_radius: int
    guided_eps: float
    multi_scale: bool

    # Edge refinement
    use_edge_aware: bool
    edge_sigma_space: float
    edge_sigma_color: float

    # RGB cleanup
    despill_mode: int  # 0=green, 1=blue, 2=physical
    despill_strength: float
    use_luminance_restore: bool
    pixel_spread: float


# Presets optimisés pour différents matériaux
PRESETS: Dict[str, MattingPreset] = {
    "hair_fine": MattingPreset(
        name="Hair - Fine Details",
        description="Optimisé pour cheveux fins, mèches, baby hair",
        grow_shrink=0,
        fill_holes=50,
        remove_dots=30,
        border_fix=1,
        feather=2.0,
        trimap_band=20,
        use_trimap=True,
        temporal_smooth=0.65,
        use_advanced_matting=True,
        advanced_mode="both",
        guided_radius=8,
        guided_eps=1e-5,
        multi_scale=True,
        use_edge_aware=True,
        edge_sigma_space=8.0,
        edge_sigma_color=20.0,
        despill_mode=2,  # Physical
        despill_strength=0.75,
        use_luminance_restore=True,
        pixel_spread=12.0
    ),

    "hair_thick": MattingPreset(
        name="Hair - Thick/Curly",
        description="Pour cheveux épais, bouclés, dreadlocks",
        grow_shrink=1,
        fill_holes=200,
        remove_dots=100,
        border_fix=2,
        feather=3.0,
        trimap_band=15,
        use_trimap=True,
        temporal_smooth=0.70,
        use_advanced_matting=True,
        advanced_mode="guided",
        guided_radius=10,
        guided_eps=1e-4,
        multi_scale=False,
        use_edge_aware=True,
        edge_sigma_space=10.0,
        edge_sigma_color=25.0,
        despill_mode=2,
        despill_strength=0.80,
        use_luminance_restore=True,
        pixel_spread=10.0
    ),

    "fur": MattingPreset(
        name="Fur/Animal Hair",
        description="Pour fourrure animale, pelage",
        grow_shrink=0,
        fill_holes=100,
        remove_dots=50,
        border_fix=2,
        feather=4.0,
        trimap_band=25,
        use_trimap=True,
        temporal_smooth=0.60,
        use_advanced_matting=True,
        advanced_mode="both",
        guided_radius=12,
        guided_eps=1e-5,
        multi_scale=True,
        use_edge_aware=True,
        edge_sigma_space=12.0,
        edge_sigma_color=30.0,
        despill_mode=2,
        despill_strength=0.70,
        use_luminance_restore=True,
        pixel_spread=15.0
    ),

    "smoke": MattingPreset(
        name="Smoke/Fog",
        description="Pour fumée, brouillard, vapeur",
        grow_shrink=2,
        fill_holes=0,
        remove_dots=0,
        border_fix=0,
        feather=8.0,
        trimap_band=30,
        use_trimap=False,
        temporal_smooth=0.75,
        use_advanced_matting=True,
        advanced_mode="guided",
        guided_radius=15,
        guided_eps=0.001,
        multi_scale=False,
        use_edge_aware=False,
        edge_sigma_space=15.0,
        edge_sigma_color=40.0,
        despill_mode=2,
        despill_strength=0.60,
        use_luminance_restore=False,
        pixel_spread=8.0
    ),

    "glass": MattingPreset(
        name="Glass/Transparent",
        description="Pour verre, objets transparents, reflets",
        grow_shrink=0,
        fill_holes=0,
        remove_dots=20,
        border_fix=1,
        feather=1.0,
        trimap_band=10,
        use_trimap=True,
        temporal_smooth=0.55,
        use_advanced_matting=True,
        advanced_mode="guided",
        guided_radius=6,
        guided_eps=1e-6,
        multi_scale=True,
        use_edge_aware=True,
        edge_sigma_space=5.0,
        edge_sigma_color=15.0,
        despill_mode=2,
        despill_strength=0.50,
        use_luminance_restore=False,
        pixel_spread=5.0
    ),

    "sharp_edges": MattingPreset(
        name="Sharp Edges",
        description="Pour objets avec bords nets (logos, graphiques)",
        grow_shrink=0,
        fill_holes=500,
        remove_dots=200,
        border_fix=3,
        feather=0.5,
        trimap_band=5,
        use_trimap=False,
        temporal_smooth=0.50,
        use_advanced_matting=False,
        advanced_mode="guided",
        guided_radius=3,
        guided_eps=1e-3,
        multi_scale=False,
        use_edge_aware=False,
        edge_sigma_space=3.0,
        edge_sigma_color=10.0,
        despill_mode=0,  # Green average
        despill_strength=0.85,
        use_luminance_restore=True,
        pixel_spread=8.0
    ),

    "fabric": MattingPreset(
        name="Fabric/Clothing",
        description="Pour vêtements, tissus, textures",
        grow_shrink=1,
        fill_holes=300,
        remove_dots=150,
        border_fix=2,
        feather=3.0,
        trimap_band=12,
        use_trimap=True,
        temporal_smooth=0.65,
        use_advanced_matting=True,
        advanced_mode="guided",
        guided_radius=8,
        guided_eps=1e-4,
        multi_scale=False,
        use_edge_aware=True,
        edge_sigma_space=8.0,
        edge_sigma_color=20.0,
        despill_mode=2,
        despill_strength=0.80,
        use_luminance_restore=True,
        pixel_spread=10.0
    ),

    "motion_blur": MattingPreset(
        name="Motion Blur",
        description="Pour objets en mouvement avec flou de bougé",
        grow_shrink=2,
        fill_holes=200,
        remove_dots=100,
        border_fix=1,
        feather=5.0,
        trimap_band=18,
        use_trimap=False,
        temporal_smooth=0.80,
        use_advanced_matting=True,
        advanced_mode="guided",
        guided_radius=12,
        guided_eps=1e-4,
        multi_scale=False,
        use_edge_aware=True,
        edge_sigma_space=12.0,
        edge_sigma_color=35.0,
        despill_mode=2,
        despill_strength=0.75,
        use_luminance_restore=True,
        pixel_spread=12.0
    ),

    "default": MattingPreset(
        name="Default/Balanced",
        description="Paramètres équilibrés pour usage général",
        grow_shrink=0,
        fill_holes=300,
        remove_dots=200,
        border_fix=2,
        feather=4.0,
        trimap_band=14,
        use_trimap=True,
        temporal_smooth=0.60,
        use_advanced_matting=True,
        advanced_mode="guided",
        guided_radius=8,
        guided_eps=1e-4,
        multi_scale=False,
        use_edge_aware=True,
        edge_sigma_space=10.0,
        edge_sigma_color=25.0,
        despill_mode=2,
        despill_strength=0.75,
        use_luminance_restore=True,
        pixel_spread=10.0
    ),
}


def get_preset(name: str) -> MattingPreset:
    """Récupère un preset par nom"""
    return PRESETS.get(name, PRESETS["default"])


def list_presets() -> list[str]:
    """Liste tous les presets disponibles"""
    return list(PRESETS.keys())


def get_preset_names_descriptions() -> list[tuple[str, str]]:
    """Retourne (name, description) pour tous les presets"""
    return [(p.name, p.description) for p in PRESETS.values()]
