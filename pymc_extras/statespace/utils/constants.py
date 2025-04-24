import pytensor

ALL_STATE_DIM = "state"
ALL_STATE_AUX_DIM = "state_aux"
OBS_STATE_DIM = "observed_state"
OBS_STATE_AUX_DIM = "observed_state_aux"
SHOCK_DIM = "shock"
SHOCK_AUX_DIM = "shock_aux"
TIME_DIM = "time"
AR_PARAM_DIM = "ar_lag"
MA_PARAM_DIM = "ma_lag"
SEASONAL_AR_PARAM_DIM = "seasonal_ar_lag"
SEASONAL_MA_PARAM_DIM = "seasonal_ma_lag"
ETS_SEASONAL_DIM = "seasonal_lag"

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
    "filtered_state",
    "predicted_state",
    "filtered_covariance",
    "predicted_covariance",
]

SMOOTHER_OUTPUT_NAMES = ["smoothed_state", "smoothed_covariance"]
OBSERVED_OUTPUT_NAMES = ["predicted_observed_state", "predicted_observed_covariance"]

ALL_KF_OUTPUT_NAMES = [
    "filtered_states",
    "predicted_states",
    "observed_states",
    "filtered_covariances",
    "predicted_covariances",
    "observed_covariances",
    "loglike_obs",
]

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
    "filtered_state": (TIME_DIM, ALL_STATE_DIM),
    "smoothed_state": (TIME_DIM, ALL_STATE_DIM),
    "predicted_state": (TIME_DIM, ALL_STATE_DIM),
    "filtered_covariance": (TIME_DIM, ALL_STATE_DIM, ALL_STATE_AUX_DIM),
    "smoothed_covariance": (TIME_DIM, ALL_STATE_DIM, ALL_STATE_AUX_DIM),
    "predicted_covariance": (TIME_DIM, ALL_STATE_DIM, ALL_STATE_AUX_DIM),
    "predicted_observed_state": (TIME_DIM, OBS_STATE_DIM),
    "predicted_observed_covariance": (TIME_DIM, OBS_STATE_DIM, OBS_STATE_AUX_DIM),
}

POSITION_DERIVATIVE_NAMES = ["level", "trend", "acceleration", "jerk", "snap", "crackle", "pop"]
SARIMAX_STATE_STRUCTURES = ["fast", "interpretable"]
