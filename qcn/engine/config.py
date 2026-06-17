from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Network types
NETWORK_TYPE_SBQI = "SBQI"
NETWORK_TYPE_OFBQI = "OFBQI"
NETWORK_TYPES = [NETWORK_TYPE_SBQI, NETWORK_TYPE_OFBQI]

# Simulation modes
SIM_MODE_DISTRIBUTION = "Degree Distribution"
SIM_MODE_EVOLUTION = "Evolution (N)"

# Physics constants — SBQI (satellite)
SAT_HEIGHT_KM = 500
SAT_ETA_0 = 0.1
SAT_R_REC = 0.75
SAT_W_LT = 0.25e-5
SAT_N_PAIRS = 50

# Physics constants — OFBQI (optic fiber)
FIBER_ATTENUATION_DIVISOR = 226.0

# Physics constants — OFBQI (photonic)
PHOTONIC_ATTENUATION_COEFF = 0.2
PHOTONIC_ATTENUATION_DIVISOR = 10.0
PHOTONIC_N_TRIALS = 1000

# Statistical fit constants
POISSON_DENSITY = 1e-4
POISSON_A = 5.2e4

# Simulation defaults
DEFAULT_NODES = 1000
DEFAULT_RADIUS_DISTRIBUTION = 1261.0
DEFAULT_DENSITY_COEFF_EVOLUTION = 2.0
DEFAULT_MC_REPS = 5
DEFAULT_STEPS = 5
DEFAULT_NODE_INCREMENT = 100
DEFAULT_FALLBACK_RADIUS = 1000.0

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Plot styles
STYLE_MAP = {
    NETWORK_TYPE_SBQI:  {"color": "blue",    "marker": "d", "label": "SBQI"},
    NETWORK_TYPE_OFBQI: {"color": "#FF6600", "marker": "o", "label": "OFBQI"},
}
