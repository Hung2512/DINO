_base_ = ['DINO_4scale.py']

# =====================================================================================
# Thyroid-specific overrides.
# Rationale documented in the accompanying analysis; kept in a separate file (rather than
# edited in-place in DINO_4scale.py) so other experiments using the base COCO-style config
# are unaffected.
# =====================================================================================

# --- A. Post-processing / selection -------------------------------------------------
# num_select was 50 out of only num_queries(20) * num_classes(3) = 60 possible (query, class)
# pairs -> almost no filtering at all, and (with class_agnostic selection) a visually salient
# class (thyroid hotspot) can occupy most of the ranked list, crowding out the weak-signal
# shoulder class. Two fixes:
#   1) class_agnostic_select = False: guarantee `num_select` candidates PER CLASS, so shoulder
#      always gets its own ranked shortlist regardless of how confident thyroid predictions are.
#   2) nms_iou_threshold > 0: also now correctly classwise (batched_nms), removing duplicate/
#      redundant boxes so the top of each class's ranked list is less cluttered.
num_select = 5
class_agnostic_select = False
nms_iou_threshold = 0.5

# --- B. Class-imbalanced classification loss -----------------------------------------
# Index 0 is the unused placeholder id (ID2NAME[0] == '__bg__', never assigned as a positive
# target), index 1 = shoulder, index 2 = thyroid. Shoulder sits over a near-uniform,
# low-contrast background ROI (weak visual signal) and is being systematically under-detected
# relative to the visually salient thyroid hotspot. Up-weight its contribution to the
# classification loss so the optimizer doesn't get to "ignore" it as an easy/low-loss class.
class_loss_weight = [0.0, 2.0, 1.0]

# --- B. Stability / calibration -------------------------------------------------------
# EMA tends to give better-calibrated confidence scores, which matters a lot here since RSI
# only trusts the highest-scoring box per class (top_k in calc_rsi_acc).
use_ema = True
ema_decay = 0.9997

# --- B. Training schedule --------------------------------------------------------------
# Base config uses StepLR(step_size=lr_drop=11) over only 50 epochs -> LR decays by 10x at
# epochs 11/22/33/44, i.e. it's already ~1e4x smaller than the initial LR for the last ~35
# epochs. That leaves very little effective training time for a harder/weaker class like
# shoulder to converge. Lengthen the schedule and push the decay point back proportionally.
epochs = 80
lr_drop = 30
