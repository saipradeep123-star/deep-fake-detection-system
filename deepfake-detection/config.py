"""
Central configuration for the DeepFake Detection System.
Edit these values instead of hunting through scripts.
"""
import os
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_VIDEO_DIR = os.path.join(DATA_DIR, "raw_videos")        # put downloaded videos here
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")         # extracted face sequences (.npy) land here
SPLITS_DIR = os.path.join(DATA_DIR, "splits")                # train/val/test CSVs

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
FACE_SIZE = 224                 # EfficientNet-B4 expects 224-380; 224 keeps it fast
SEQUENCE_LENGTH = 20             # number of sampled frames per video clip
FRAME_SAMPLE_STRIDE = 5          # take every Nth frame before selecting SEQUENCE_LENGTH of them
EAR_THRESHOLD = 0.21             # eye-aspect-ratio threshold below which an eye counts as "closed"
EAR_CONSEC_FRAMES = 2            # consecutive closed-eye frames required to count as a blink

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
CNN_BACKBONE = "efficientnet_b4"
CNN_FEATURE_DIM = 1792           # EfficientNet-B4's native feature dimension (timm, pooled)
LSTM_HIDDEN_DIM = 256
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
NUM_CLASSES = 2                  # real vs fake
FREEZE_CNN_BACKBONE = True       # start with frozen CNN; unfreeze for fine-tuning later

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 8
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4

# ---------------------------------------------------------------------------
# Inference / Flask app
# ---------------------------------------------------------------------------
MODEL_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_model.pt")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "app", "uploads")
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
MAX_CONTENT_LENGTH_MB = 200
