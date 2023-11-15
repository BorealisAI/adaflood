# Dataset
INPUT_DIM = "input_dim"
NUM_CLS = "num_classes"
PRED_CLS = "pred_cls"
MISSING_MASKS = "missing_masks"
USE_MARKS = "use_marks"

# Inputs
TIMES = "times"
MARKS = "marks"
MASKS = "masks"
LOG_MEAN = "log_mean"
LOG_STD = "log_std"
INDICES = "indices"
#IS_FIRST_HALF = 'is_first_half' # for adaflood

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
ATTENTIONS = "attentions"
ENGINE = "ENGINE"
LOG_WEIGHTS = "log_weights"
AUX_LOG_WEIGHTS = "aux_log_weights"
PROB_DIST = "prob_dist"
AUX_PROB_DIST = "aux_prob_dist"
LOGNORM_DIST = "lognorm_dist"
DIST_MU = "dist_mu"
DIST_SIGMA = "dist_sigma"

# AdaFlood
ALPHA = 'alpha'
BETA = 'beta'

# Outputs of aux models
AUX1_HISTORIES = "aux1_histories"
AUX1_OUTPUT_TYPE = "aux1_output_type"
AUX1_EVENT_LL = "aux1_event_ll"
AUX1_SURV_LL = "aux1_surv_ll"
AUX1_KL = "aux1_kl_divergence"
AUX1_TIME_PREDS = "aux1_time_preds"
AUX1_CLS_LL = "aux1_class_ll"
AUX1_CLS_LOGITS = "aux1_class_logits"
AUX1_CLS_PREDS = "aux1_class_preds"
AUX1_ATTENTIONS = "aux1_attentions"
AUX1_ENGINE = "aux1_ENGINE"

AUX2_HISTORIES = "aux2_histories"
AUX2_OUTPUT_TYPE = "aux2_output_type"
AUX2_EVENT_LL = "aux2_event_ll"
AUX2_SURV_LL = "aux2_surv_ll"
AUX2_KL = "aux2_kl_divergence"
AUX2_TIME_PREDS = "aux2_time_preds"
AUX2_CLS_LL = "aux2_class_ll"
AUX2_CLS_LOGITS = "aux2_class_logits"
AUX2_CLS_PREDS = "aux2_class_preds"
AUX2_ATTENTIONS = "aux2_attentions"
AUX2_ENGINE = "aux2_ENGINE"


# Evaluatation
LOSS = "loss"
LOSSES = "losses"
CHECKPOINT_METRIC = "checkpoint_metric"

# Transformer
PAD = 0


# Classification

# Inputs
IMAGES = "images"
LABELS = "labels"
#IS_FIRST_HALF = 'is_first_half' # for adaflood

# Outputs
LOGITS = "logits"
AUX_LOGITS = "aux_logits"
AUX_LOSSES = "aux_losses"
AUX_PREDS = "aux_preds"
AUX_MIN_LOSS = "aux_min_loss"
AUX_EVAL_LOSSES = "aux_eval_losses"
AUX_TRAIN_LOSSES = "aux_train_losses"


