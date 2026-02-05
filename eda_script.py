import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config

def run_eda():
    """
    EDA for the MIND dataset.
    Generates plots for sparsity, cold start, content analysis, and class balance.
    """
    print("Starting EDA...")
    Config.ensure_dirs()
    
    # Load Data
    print("Loading data...")
    train_news_path = os.path.join(Config.TRAIN_PATH, Config.NEWS_FILENAME)
    train_behaviors_path = os.path.join(Config.TRAIN_PATH, Config.BEHAVIORS_FILENAME)
    dev_news_path = os.path.join(Config.DEV_PATH, Config.NEWS_FILENAME)
    
    # Define column names based on MIND specifications
    news_cols = ['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities']
    behaviors_cols = ['ImpressionID', 'UserID', 'Time', 'History', 'Impressions']
    
    # Read files
    train_news = pd.read_csv(train_news_path, sep='\t', names=news_cols, index_col=False)
    dev_news = pd.read_csv(dev_news_path, sep='\t', names=news_cols, index_col=False)
    train_behaviors = pd.read_csv(train_behaviors_path, sep='\t', names=behaviors_cols, index_col=False)
    
    print(f"Train News: {train_news.shape}")
    print(f"Train Behaviors: {train_behaviors.shape}")
    
    # 1. Sparsity Analysis
    print("Analyzing Sparsity...")
    train_behaviors['click_count'] = train_behaviors['History'].fillna('').apply(lambda x: len(x.split()) if x else 0)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(train_behaviors['click_count'], bins=50, kde=False)
    plt.title('Distribution of Clicks per User (History Length)')
    plt.xlabel('Number of Clicks')
    plt.ylabel('User Count')
    plt.savefig(os.path.join(Config.PLOTS_DIR, 'sparsity_clicks_distribution.png'))
    plt.close()
    
    # Estimate sparsity ratio (based on interaction matrix approximation)
    n_users = train_behaviors['UserID'].nunique()
    n_news = train_news['NewsID'].nunique()
    total_interactions = train_behaviors['click_count'].sum()
    sparsity = 1.0 - (total_interactions / (n_users * n_news))
    print(f"Sparsity Ratio: {sparsity:.6f}")
    
    # 2. Cold Start Analysis
    print("Analyzing Cold Start...")
    train_news_ids = set(train_news['NewsID'])
    dev_news_ids = set(dev_news['NewsID'])
    
    # Identify news present in Dev but missing from Train
    cold_start_news = dev_news_ids - train_news_ids
    cold_start_ratio = len(cold_start_news) / len(dev_news_ids) if dev_news_ids else 0
    print(f"Cold Start News Count: {len(cold_start_news)}")
    print(f"Cold Start Ratio (in Valid set): {cold_start_ratio:.2%}")
    
    # 3. Content Analysis
    print("Analyzing Content...")
    # Calculate title length in words
    train_news['title_len'] = train_news['Title'].fillna('').apply(lambda x: len(x.split()))
    
    plt.figure(figsize=(10, 6))
    sns.histplot(train_news['title_len'], bins=30, kde=True)
    plt.axvline(x=Config.MAX_TITLE_LENGTH, color='r', linestyle='--', label=f'Max Len ({Config.MAX_TITLE_LENGTH})')
    plt.title('News Title Length Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(Config.PLOTS_DIR, 'content_title_length.png'))
    plt.close()
    
    # 4. Class Balance
    print("Analyzing Class Balance...")
    # Parse impressions (format: "NewsID-1 NewsID-0 ...")
    def count_clicks(impressions):
        if pd.isna(impressions):
            return 0, 0
        items = impressions.split()
        pos = sum(1 for item in items if item.endswith('-1'))
        neg = sum(1 for item in items if item.endswith('-0'))
        return pos, neg

    counts = train_behaviors['Impressions'].apply(count_clicks)
    total_pos = sum(c[0] for c in counts)
    total_neg = sum(c[1] for c in counts)
    
    plt.figure(figsize=(6, 6))
    plt.pie([total_pos, total_neg], labels=['Clicked (1)', 'Not Clicked (0)'], autopct='%1.1f%%', startangle=140)
    plt.title('Class Balance (Impressions)')
    plt.savefig(os.path.join(Config.PLOTS_DIR, 'class_balance.png'))
    plt.close()

    print("EDA Completed. Plots saved to 'plots/' directory.")

if __name__ == "__main__":
    run_eda()