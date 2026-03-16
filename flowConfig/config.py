"""Configuration for two-stage traffic flow prediction."""

# ============ Paths ============
DATA_DIR = "data/flow"
MODEL_DIR = "models"
LOG_DIR = "logs"
DOCS_DIR = "docs"

# ============ Data ============
SEQ_LEN = 4
PRED_LEN = 1
TRAIN_RATIO = 0.8

# ============ Topology ============
TOPOLOGY = {
    "exogenous_distance_metric": "max",
}

# ============ Node alignment ============
NODE_ALIGNMENT = {
    "enabled": True,
    "mode": "intersection_adj_train_test",
    "dropped_preview_count": 20,
}

# ============ Exogenous LSTM ============
LSTM = {
    "input_window": 4,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.1,
}

# ============ Allocation (Dirichlet-like alpha) ============
ALLOCATION = {
    "num_frequencies": 2,
    "omega": 2 * 3.1415926 / 24,
    "prior_alpha0": 1.0,
    "prior_ak_bk": 0.1,
}

# ============ Propagation (Beta posterior) ============
PROPAGATION = {
    "input_window": 4,
    "hidden_dim": 32,
    "num_layers": 2,
    "prior_prob": 0.7,
    "prior_precision": 10,
    "use_variational": False,
}

# ============ Legacy train defaults ============
TRAIN = {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "loss": "Huber",
    "lr_scheduler": "ReduceLROnPlateau",
    "lr_factor": 0.5,
    "lr_patience": 10,
    "lr_min": 1e-6,
    "early_stop_patience": 20,
    "grad_clip": 1.0,
}

# ============ Stage-1: exogenous independent training ============
EXOGENOUS_TRAIN = {
    "epochs": 80,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "early_stop_patience": 12,
    "save_dir": "models/exogenous",
}

# ============ Stage-2: endogenous training ============
ENDOGENOUS_TRAIN = {
    "epochs": 120,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "lr_factor": 0.5,
    "lr_patience": 10,
    "lr_min": 1e-6,
    "early_stop_patience": 20,
    "grad_clip": 1.0,
    "checkpoint_name": "best_model.pt",
}

# ============ Scheduled sampling ============
SCHEDULED_SAMPLING = {
    "enabled": True,
    "type": "linear",
    "start_ratio": 1.0,
    "end_ratio": 0.2,
    "start_epoch": 0,
    "end_epoch": 80,
}

# ============ Output constraints ============
OUTPUT_CONSTRAINTS = {
    # Enforce non-negative flows in physical (de-normalized) space.
    "physical_non_negative": False,
}

# ============ Endogenous Gaussian compensation ============
ENDOGENOUS_GAUSSIAN = {
    # Enable Gaussian compensation after each endogenous step prediction.
    "enabled": True,
    # Learn one (mu, log_var) pair per node and apply only on endogenous nodes.
    "per_node": True,
    "init_mu": 0.0,
    "init_log_var": -4.0,
    # Training-time behavior.
    "train_use_sampling": True,
    # In eval/inference, use deterministic mean compensation by default.
    "eval_use_mean": True,
}

# ============ Device ============
DEVICE = "cuda"


def get_device():
    """Resolve torch device from configuration."""
    import torch

    if DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
