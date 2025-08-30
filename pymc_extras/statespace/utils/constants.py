import pytensor

ALL_STATE_DIM = "state"
ALL_STATE_AUX_DIM = "state_aux"
OBS_STATE_DIM = "observed_state"
OBS_STATE_AUX_DIM = "observed_state_aux"
SHOCK_DIM = "shock"
SHOCK_AUX_DIM = "shock_aux"
TIME_DIM = "time"
AR_PARAM_DIM = "lag_ar"
MA_PARAM_DIM = "lag_ma"
SEASONAL_AR_PARAM_DIM = "seasonal_lag_ar"
SEASONAL_MA_PARAM_DIM = "seasonal_lag_ma"
ETS_SEASONAL_DIM = "seasonal_lag"
EXOGENOUS_DIM = "exogenous"

NEVER_TIME_VARYING = ["initial_state", "initial_state_cov", "a0", "P0"]
VECTOR_VALUED = ["initial_state", "state_intercept", "obs_intercept", "a0", "c", "d"]

MISSING_FILL = -9999.0
JITTER_DEFAULT = 1e-8 if pytensor.config.floatX.endswith("64") else 1e-6

FILTER_OUTPUT_TYPES = ["filtered", "predicted", "smoothed"]

MATRIX_NAMES = ["x0", "P0", "c", "d", "T", "Z", "R", "H", "Q"]
LONG_MATRIX_NAMES = [
    "initial_state",
    "initial_state_cov",
    "state_intercept",
    "obs_intercept",
    "transition",
    "design",
    "selection",
    "obs_cov",
    "state_cov",
]

SHORT_NAME_TO_LONG = dict(zip(MATRIX_NAMES, LONG_MATRIX_NAMES))
LONG_NAME_TO_SHORT = dict(zip(LONG_MATRIX_NAMES, MATRIX_NAMES))

FILTER_OUTPUT_NAMES = [
    "filtered_states",
    "predicted_states",
    "filtered_covariances",
    "predicted_covariances",
    "predicted_observed_states",
    "predicted_observed_covariances",
]

SMOOTHER_OUTPUT_NAMES = ["smoothed_states", "smoothed_covariances"]
OBSERVED_OUTPUT_NAMES = ["predicted_observed_states", "predicted_observed_covariances"]

MATRIX_DIMS = {
    "x0": (ALL_STATE_DIM,),
    "P0": (ALL_STATE_DIM, ALL_STATE_AUX_DIM),
    "c": (ALL_STATE_DIM,),
    "d": (OBS_STATE_DIM,),
    "T": (ALL_STATE_DIM, ALL_STATE_AUX_DIM),
    "Z": (OBS_STATE_DIM, ALL_STATE_DIM),
    "R": (ALL_STATE_DIM, SHOCK_DIM),
    "H": (OBS_STATE_DIM, OBS_STATE_AUX_DIM),
    "Q": (SHOCK_DIM, SHOCK_AUX_DIM),
}

FILTER_OUTPUT_DIMS = {
    "filtered_states": (TIME_DIM, ALL_STATE_DIM),
    "smoothed_states": (TIME_DIM, ALL_STATE_DIM),
    "predicted_states": (TIME_DIM, ALL_STATE_DIM),
    "filtered_covariances": (TIME_DIM, ALL_STATE_DIM, ALL_STATE_AUX_DIM),
    "smoothed_covariances": (TIME_DIM, ALL_STATE_DIM, ALL_STATE_AUX_DIM),
    "predicted_covariances": (TIME_DIM, ALL_STATE_DIM, ALL_STATE_AUX_DIM),
    "predicted_observed_states": (TIME_DIM, OBS_STATE_DIM),
    "predicted_observed_covariances": (TIME_DIM, OBS_STATE_DIM, OBS_STATE_AUX_DIM),
}

POSITION_DERIVATIVE_NAMES = ["level", "trend", "acceleration", "jerk", "snap", "crackle", "pop"]
SARIMAX_STATE_STRUCTURES = ["fast", "interpretable"]
