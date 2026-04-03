"""Configuration constants for the AFlow visualization dashboard."""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
PROJECT_ROOT = BASE_DIR.parent

# Workspace resolution: AFLOW_WORKSPACE env var > PROJECT_ROOT (package location)
# The aflow-dashboard CLI sets AFLOW_WORKSPACE to the desired root.
_env_ws = os.environ.get("AFLOW_WORKSPACE")
_ws_root = Path(_env_ws).resolve() if _env_ws else PROJECT_ROOT

WORKSPACE_DIR = _ws_root / "workspace"
WORKSPACE_DIRS = [
    _ws_root / d.name
    for d in sorted(_ws_root.iterdir())
    if d.is_dir() and d.name.startswith("workspace")
]

# UI Configuration
PLOT_HEIGHT = 400
MAX_PREDICTIONS_DISPLAY = 200

# Cache TTLs (seconds)
CACHE_TTL_DATASETS = 30
CACHE_TTL_RESULTS = 10

# Color scheme (TensorBoard-inspired, consistent with aa-context-optimization)
COLORS = {
    "primary": "#FF6F00",
    "secondary": "#0091EA",
    "success": "#00C853",
    "warning": "#FFD600",
    "error": "#D50000",
    "train": "#FF6F00",
    "validation": "#FF6F00",
    "dev": "#00C853",
    "test": "#0091EA",
}

# Eve Persona Adherence dimension configuration
EVE_DIMENSIONS = ["verbosity", "tone_of_voice", "assertiveness", "empathy"]
EVE_BASELINE = {
    "verbosity": 0.871,
    "tone_of_voice": 0.870,
    "assertiveness": 0.847,
    "empathy": 0.516,
    "score": 0.776,
}
# Shared legend style — horizontal, centred below the X axis
LEGEND_BELOW = dict(
    orientation="h",
    yanchor="top",
    y=-0.18,
    xanchor="center",
    x=0.5,
)

EVE_DIMENSION_COLORS = {
    "verbosity": "#7B1FA2",
    "tone_of_voice": "#0091EA",
    "assertiveness": "#FF6F00",
    "empathy": "#00C853",
    "score": "#D50000",
}
