import os
import torch

class Config:
    # Paths
    DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
    TRAIN_PATH = os.path.join(DATA_ROOT, 'MINDsmall_train')
    DEV_PATH = os.path.join(DATA_ROOT, 'MINDsmall_dev')
    
    NEWS_FILENAME = 'news.tsv'
    BEHAVIORS_FILENAME = 'behaviors.tsv'
    
    PLOTS_DIR = os.path.join(DATA_ROOT, 'plots')
    CHECKPOINT_DIR = os.path.join(DATA_ROOT, 'checkpoints')
    
    # Hyperparameters
    MAX_TITLE_LENGTH = 30
    EMBEDDING_DIM = 64  # Small dimension for PoW
    BATCH_SIZE = 1024
    LEARNING_RATE = 0.001
    EPOCHS = 1          # Short training for PoW
    
    # Model
    HIDDEN_DIM = 64
    DROPOUT = 0.2
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @staticmethod
    def ensure_dirs():
        os.makedirs(Config.PLOTS_DIR, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

if __name__ == "__main__":
    Config.ensure_dirs()
    print(f"Project root: {Config.DATA_ROOT}")
    print(f"Device: {Config.DEVICE}")
