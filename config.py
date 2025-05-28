# === Global Experiment Parameters ===

# Number of steps for ε-scaling (logarithmic)
NUM_EPSILON_STEPS = 1000

# Range of ε values (perceptual resolution scale)
EPSILON_MIN = 0.001
EPSILON_MAX = 0.1

# Trajectory generation settings
TRAJECTORY_DIM = 5
TRAJECTORY_POINTS = 200
NOISE_LEVEL = 0.03
ROUNDING_PRECISION = 0.02
SEED = 42

# Neural observer configuration
OBSERVER_HIDDEN_DIM = 36
OBSERVER_OUTPUT_DIM = 2

# Output directories
RESULTS_DIR = "results"
PROJECTIONS_DIR = f"{RESULTS_DIR}/projections"
ENTROPY_DIR = f"{RESULTS_DIR}/entropy"
ANALYSIS_DIR = f"{RESULTS_DIR}/analysis"

# Plotting options (True — display plots on screen, False — save only)
SHOW_PLOTS = True
PLOT_DPI = 300
PLOT_FIGSIZE = (10, 6)
PLOT_ALPHA = 0.7

# Entropy derivative extremum detection
EXTREMUM_PROMINENCE = 0.05  # Minimum prominence of a peak (used in scipy.signal.find_peaks)

# Save entropy derivative data as CSV/JSON
SAVE_DERIVATIVE_DATA = True
