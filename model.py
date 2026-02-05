import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class NewsEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings=None):
        super().__init__()
        # Basic Word Embedding with Mean Pooling
        # Should be replaced with stronger encoders like GloVe or BERT in final version
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
    def forward(self, news_title_tokens):
        # news_title_tokens: [batch_size, max_len]
        
        # Create mask for padding (0)
        mask = (news_title_tokens != 0).float().unsqueeze(-1)
        
        # Embed: [batch, len, dim]
        emb = self.embedding(news_title_tokens)
        
        # Mean Pooling
        sum_emb = (emb * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1e-9)
        
        return sum_emb / count

class HybridRecModel(nn.Module):
    def __init__(self, vocab_size, num_users, embedding_dim=64, hidden_dim=64):
        super().__init__()
        
        self.news_encoder = NewsEncoder(vocab_size, embedding_dim)
        
        # Trainable User Embeddings (Initial state / Bias)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        
        # GNN Projection Layers (SAGE-style)
        self.aggr_linear = nn.Linear(embedding_dim, hidden_dim) # For neighbor aggregation
        self.self_linear = nn.Linear(embedding_dim, hidden_dim) # For self-connection
        self.output_act = nn.Tanh()
        
        self.embedding_dim = embedding_dim

    def forward(self, news_text, user_history_batch, user_ids, candidate_news_ids):
        """
        news_text: [num_news_total, max_len]
        user_history_batch: [batch_size, max_history] - Padded indices of clicked news
        user_ids: [batch_size]
        candidate_news_ids: [batch_size]
        """
        
        # 1. Encode All News (Content-based)
        news_emb = self.news_encoder(news_text)
        
        # 2. Batched GNN / Aggregation
        # Retrieve embeddings for user history: [batch, max_hist, dim]
        hist_emb = news_emb[user_history_batch] 
        
        # Mean Aggregation
        # Averaging embeddings (including padding for simplicity in this implementation)
        user_hist_rep = hist_emb.mean(dim=1) # [batch, dim]
        
        # Update User Embedding (SAGE: Linear(Self) + Linear(Aggr))
        u_self = self.user_embedding(user_ids)
        
        user_emb_new = self.self_linear(u_self) + self.aggr_linear(user_hist_rep)
        user_emb_new = self.output_act(user_emb_new)
        
        # 3. Score
        batch_news_emb = news_emb[candidate_news_ids]
        scores = (user_emb_new * batch_news_emb).sum(dim=1)
        
        return scores