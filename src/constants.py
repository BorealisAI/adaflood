# Dataset
INPUT_DIM = "input_dim"
NUM_CLS = "num_classes"
PRED_CLS = "pred_cls"
MISSING_MASKS = "missing_masks"
USE_MARKS = "use_marks"

DELTAS = {
    'sin': 8, 'sin_long': 8, 'so_fold1': 16, 'mooc': None,
    'reddit': 8, 'wiki': 32, 'uber': 32, 'taxi': 32}


# Inputs
TIMES = "times"
MARKS = "marks"
MASKS = "masks"
LOG_MEAN = "log_mean"
LOG_STD = "log_std"
INDICES = "indices"

# Outputs
HISTORIES = "histories"
OUTPUT_TYPE = "output_type"
EVENT_LL = "event_ll"
SURV_LL = "surv_ll"
KL = "kl_divergence"
TIME_PREDS = "time_preds"
CLS_LL = "class_ll"
CLS_LOGITS = "class_logits"
CLS_PREDS = "class_preds"
ENGINE = "ENGINE"
LOG_WEIGHTS = "log_weights"
PROB_DIST = "prob_dist"
LOGNORM_DIST = "lognorm_dist"
DIST_MU = "dist_mu"
DIST_SIGMA = "dist_sigma"
NLLS = "nlls"

# Evaluatation
LOSS = "loss"
LOSSES = "losses"
CHECKPOINT_METRIC = "checkpoint_metric"

# Transformer
PAD = 0

# Forecast Outputs
NLL = "nll"

