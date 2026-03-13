"""Configuration constants for the AFlow visualization dashboard."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
PROJECT_ROOT = BASE_DIR.parent
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
WORKSPACE_DIRS = [
    PROJECT_ROOT / d
    for d in sorted(PROJECT_ROOT.iterdir())
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
    "validation": "#FF6F00",
    "test": "#0091EA",
}

# Eve AICW dimension configuration
EVE_DIMENSIONS = ["verbosity", "tone", "assertiveness", "empathy"]
EVE_BASELINE = {
    "verbosity": 0.871,
    "tone": 0.870,
    "assertiveness": 0.847,
    "empathy": 0.516,
    "score": 0.776,
}
EVE_DIMENSION_COLORS = {
    "verbosity": "#7B1FA2",
    "tone": "#0091EA",
    "assertiveness": "#FF6F00",
    "empathy": "#00C853",
    "score": "#D50000",
}
