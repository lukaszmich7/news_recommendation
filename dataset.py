import os
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset, HeteroData
from config import Config
from tqdm import tqdm

class MINDDataset(InMemoryDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None):
        self.split = split
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return [Config.NEWS_FILENAME, Config.BEHAVIORS_FILENAME]

    @property
    def processed_file_names(self):
        return [f'mind_{self.split}.pt']

    def download(self):
        # Assumes data is already present locally
        pass

    def process(self):
        print(f"Processing {self.split} dataset...")
        
        # Define paths
        data_path = Config.TRAIN_PATH if self.split == 'train' else Config.DEV_PATH
        news_path = os.path.join(data_path, Config.NEWS_FILENAME)
        behaviors_path = os.path.join(data_path, Config.BEHAVIORS_FILENAME)

        # Load News data
        news_cols = ['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities']
        news_df = pd.read_csv(news_path, sep='\t', names=news_cols, index_col=False)
        
        # Create NewsID -> Index mapping
        news_id_map = {nid: i for i, nid in enumerate(news_df['NewsID'])}
        
        # Tokenize titles
        print("Tokenizing titles...")
        word2idx = {'<PAD>': 0, '<UNK>': 1}
        
        def tokenize_and_map(text):
            if not isinstance(text, str):
                return [0] * Config.MAX_TITLE_LENGTH
            tokens = text.lower().split()[:Config.MAX_TITLE_LENGTH]
            ids = []
            for token in tokens:
                if token not in word2idx:
                    if self.split == 'train': 
                        # Build vocab only on training split
                        word2idx[token] = len(word2idx)
                    else:
                        # For dev/test, use UNK if token is missing
                        pass 
                ids.append(word2idx.get(token, word2idx['<UNK>']))
            
            # Pad sequences
            if len(ids) < Config.MAX_TITLE_LENGTH:
                ids += [0] * (Config.MAX_TITLE_LENGTH - len(ids))
            return ids

        # Note: Ideally the vocabulary should be persisted and loaded.
        # For this implementation, it's built dynamically during processing.
        
        title_features = []
        for title in tqdm(news_df['Title']):
            title_features.append(tokenize_and_map(title))
            
        title_features = torch.tensor(title_features, dtype=torch.long)
        
        # Load Behaviors to retrieve Users and Interactions
        behaviors_cols = ['ImpressionID', 'UserID', 'Time', 'History', 'Impressions']
        behaviors_df = pd.read_csv(behaviors_path, sep='\t', names=behaviors_cols, index_col=False)
        
        # Create UserID -> Index mapping
        user_id_map = {uid: i for i, uid in enumerate(behaviors_df['UserID'].unique())}
        
        # Initialize Graph Data
        data = HeteroData()
        
        # Set Node Features
        data['news'].x = title_features
        data['news'].num_nodes = len(news_df)
        data['user'].num_nodes = len(user_id_map)
        
        # Build Edges (History Clicks)
        # Modeling history as 'user' -> 'clicked' -> 'news'
        src_users = []
        dst_news = []
        
        print("Building interaction graph...")
        for _, row in tqdm(behaviors_df.iterrows(), total=len(behaviors_df)):
            if pd.isna(row['History']):
                continue
            
            u_idx = user_id_map[row['UserID']]
            history_news_ids = row['History'].split()
            
            for nid in history_news_ids:
                if nid in news_id_map: 
                    n_idx = news_id_map[nid]
                    src_users.append(u_idx)
                    dst_news.append(n_idx)
                    
        edge_index = torch.tensor([src_users, dst_news], dtype=torch.long)
        data['user', 'clicked', 'news'].edge_index = edge_index
        
        # Store metadata
        data.vocab_size = len(word2idx)
        
        torch.save(self.collate([data]), self.processed_paths[0])
        
        # Save Mappings (required for evaluation)
        mappings = {
            'news_id_map': news_id_map,
            'user_id_map': user_id_map,
            'word2idx': word2idx
        }
        torch.save(mappings, self.processed_paths[0].replace('.pt', '_maps.pt'))
        
        print("Dataset processed and saved.")

if __name__ == "__main__":
    # Test dataset creation
    Config.ensure_dirs()
    dataset = MINDDataset(root=Config.DATA_ROOT, split='train')
    print(f"Dataset: {dataset}")
    print(f"Graph: {dataset[0]}")
    print(f"Vocab Size: {dataset[0].vocab_size}")