import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, log_loss

from config import Config
from dataset import MINDDataset
from model import HybridRecModel

def train_one_epoch(model, loader, news_text, user_history, edge_index, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_scores = []
    all_labels = []
    
    # Training strategy:
    # Use existing graph edges (user, news) as positive samples.
    # Generate negative samples randomly.
    
    # Create batches of indices
    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges)
    
    pbar = tqdm(range(0, num_edges, Config.BATCH_SIZE), desc="Training")
    for i in pbar:
        batch_idx = perm[i : i + Config.BATCH_SIZE]
        
        # Positive Samples
        u_ids = edge_index[0, batch_idx].to(device)
        pos_n_ids = edge_index[1, batch_idx].to(device)
        
        # Retrieve History for Batch Users
        u_hist_batch = user_history[u_ids]
        
        # Negative Sampling
        # Randomly select news not necessarily in the batch
        neg_n_ids = torch.randint(0, news_text.size(0), (len(batch_idx),), device=device)
        
        optimizer.zero_grad()
        
        # Forward Pass (Positive)
        pos_scores = model(news_text, u_hist_batch, u_ids, pos_n_ids)
        
        # Forward Pass (Negative)
        neg_scores = model(news_text, u_hist_batch, u_ids, neg_n_ids)
        
        # Loss Calculation
        # Concatenate scores and create labels (1 for pos, 0 for neg)
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Store for AUC calculation
        all_scores.append(torch.sigmoid(scores).detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
        
    # Calculate global AUC for the epoch
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    train_auc = roc_auc_score(all_labels, all_scores)
    
    return total_loss / (num_edges / Config.BATCH_SIZE), train_auc


# Metrics Implementation
def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best if best > 0 else 0

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true) if np.sum(y_true) > 0 else 0

def hit_rate_at_k(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    return 1 if np.sum(y_true) > 0 else 0

def evaluate(model, news_text, user_history, edge_index, device):
    model.eval()
    print("Evaluating on Validation Set (MINDsmall_dev)...")
    
    # Load Mappings to translate Dev Data to Training IDs
    train_dataset_path = os.path.join(Config.DATA_ROOT, 'processed', 'mind_train.pt')
    maps_path = train_dataset_path.replace('.pt', '_maps.pt')
    
    if not os.path.exists(maps_path):
        print("Warning: Maps file not found. Skipping ranking evaluation.")
        return 0, {}
        
    mappings = torch.load(maps_path)
    news_id_map = mappings['news_id_map']
    user_id_map = mappings['user_id_map']
    word2idx = mappings['word2idx']
    
    # Load Dev Behaviors
    dev_behaviors_path = os.path.join(Config.DEV_PATH, Config.BEHAVIORS_FILENAME)
    behaviors_cols = ['ImpressionID', 'UserID', 'Time', 'History', 'Impressions']
    dev_df = pd.read_csv(dev_behaviors_path, sep='\t', names=behaviors_cols, index_col=False)
    
    # Metrics container
    metrics = {'AUC': [], 'MRR': [], 'NDCG@5': [], 'NDCG@10': [], 'Hit@10': []}
    
    # Evaluate on a subset for speed
    eval_size = 1000 
    print(f"Evaluating on {eval_size} random samples for speed...")
    
    with torch.no_grad():
        for i in tqdm(range(min(len(dev_df), eval_size)), desc="Eval"):
            row = dev_df.iloc[i]
            uid_str = row['UserID']
            history_str = row['History']
            impressions_str = row['Impressions']
            
            # Skip unknown users to focus evaluation on trained embeddings
            if uid_str not in user_id_map:
                continue
            u_idx = user_id_map[uid_str]
            
            # Retrieve History tensor
            u_hist_batch = user_history[torch.tensor([u_idx], device=device)]
            
            # Parse Impressions
            # format: "NewsID-1 NewsID-0 ..."
            impr_list = impressions_str.split()
            news_candidates = []
            labels = []
            
            for item in impr_list:
                nid, label = item.rsplit('-', 1)
                if nid in news_id_map:
                    news_candidates.append(news_id_map[nid])
                    labels.append(int(label))
                else:
                    pass
            
            if len(labels) == 0 or sum(labels) == 0:
                continue
                
            # Prepare Batch
            batch_sz = len(news_candidates)
            # Expand user history to match candidate batch size
            u_hist_expanded = u_hist_batch.repeat(batch_sz, 1)
            
            u_tensor = torch.tensor([u_idx] * batch_sz, device=device)
            n_tensor = torch.tensor(news_candidates, device=device)
            
            # Forward Pass
            scores = model(news_text, u_hist_expanded, u_tensor, n_tensor)
            scores = torch.sigmoid(scores).cpu().numpy()
            labels = np.array(labels)
            
            # Calculate Metrics
            try:
                metrics['AUC'].append(roc_auc_score(labels, scores))
                metrics['MRR'].append(mrr_score(labels, scores))
                metrics['NDCG@5'].append(ndcg_score(labels, scores, k=5))
                metrics['NDCG@10'].append(ndcg_score(labels, scores, k=10))
                metrics['Hit@10'].append(hit_rate_at_k(labels, scores, k=10))
            except ValueError:
                continue

    # Compute Averages
    final_metrics = {k: np.mean(v) for k, v in metrics.items()}
    print(f"Validation Results: {final_metrics}")
    
    return final_metrics['AUC'], final_metrics

def main():
    Config.ensure_dirs()
    device = Config.DEVICE
    print(f"Using device: {device}")
    
    # Load Training Data
    dataset = MINDDataset(root=Config.DATA_ROOT, split='train')
    data = dataset[0]
    
    news_text = data['news'].x.to(device)
    edge_index = data['user', 'clicked', 'news'].edge_index.to(device)
    
    num_users = data['user'].num_nodes
    vocab_size = data.vocab_size
    
    print(f"Data Loaded. Users: {num_users}, News: {news_text.size(0)}")
    
    # Initialize Model
    model = HybridRecModel(vocab_size, num_users, 
                          embedding_dim=Config.EMBEDDING_DIM,
                          hidden_dim=Config.HIDDEN_DIM).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    
    
    # Pre-process User History for efficient batching
    print("Pre-processing user history...")
    max_history = 50
    
    # Group clicks by user
    from collections import defaultdict
    user_hist_dict = defaultdict(list)
    
    # Use numpy for faster iteration over edges
    u_np = edge_index[0].cpu().numpy()
    n_np = edge_index[1].cpu().numpy()
    
    for u, n in zip(u_np, n_np):
        user_hist_dict[u].append(n)
        
    # Construct fixed-size history tensor
    user_history_tensor = torch.zeros((num_users, max_history), dtype=torch.long)
    for u, hist in user_hist_dict.items():
        if not hist: continue
        # Truncate or Pad
        sl = hist[:max_history]
        user_history_tensor[u, :len(sl)] = torch.tensor(sl)
        # Pad with the last item if too short
        if len(sl) < max_history:
             user_history_tensor[u, len(sl):] = sl[-1]
             
    user_history_tensor = user_history_tensor.to(device)
    print("History tensor ready.")

    # Training Loop
    for epoch in range(Config.EPOCHS):
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        loss, _ = train_one_epoch(model, None, news_text, user_history_tensor, edge_index, optimizer, criterion, device)
        auc, metrics = evaluate(model, news_text, user_history_tensor, edge_index, device)
        
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"Epoch {epoch+1} Loss: {loss:.4f} | {metrics_str}")
        
        # Save Checkpoint
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    main()