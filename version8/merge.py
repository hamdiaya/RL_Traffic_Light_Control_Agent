import pandas as pd
import logging
import matplotlib.pyplot as plt
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('merge_results.log'),
        logging.StreamHandler()
    ]
)

# File paths
VERSION_7_CSV = "test_results.csv"
VERSION_8_CSV = "agent_test_results.csv"
OUTPUT_CSV = "test_results_final.csv"

def load_version7_data():
    """Load and extract default metrics from version 7 CSV"""
    try:
        if not os.path.exists(VERSION_7_CSV):
            raise FileNotFoundError(f"Version 7 CSV not found at {VERSION_7_CSV}")
        df = pd.read_csv(VERSION_7_CSV)
        required_cols = ['episode', 'default_reward', 'default_wait_total', 'default_queue_total']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Version 7 CSV missing required columns: {required_cols}")
        return df[required_cols]
    except Exception as e:
        logging.error(f"Failed to load version 7 data: {e}")
        raise

def load_version8_data():
    """Load and rename agent metrics from version 8 CSV"""
    try:
        if not os.path.exists(VERSION_8_CSV):
            raise FileNotFoundError(f"Version 8 CSV not found at {VERSION_8_CSV}")
        df = pd.read_csv(VERSION_8_CSV)
        required_cols = ['episode', 'reward', 'wait_total', 'queue_total']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Version 8 CSV missing required columns: {required_cols}")
        df = df[required_cols].rename(columns={
            'reward': 'agent_reward',
            'wait_total': 'agent_wait_total',
            'queue_total': 'agent_queue_total'
        })
        return df
    except Exception as e:
        logging.error(f"Failed to load version 8 data: {e}")
        raise

def merge_data(v7_df, v8_df):
    """Merge version 7 and version 8 data on episode"""
    try:
        if len(v7_df) != len(v8_df):
            logging.warning(f"Episode count mismatch: version 7 has {len(v7_df)}, version 8 has {len(v8_df)}. Using common episodes.")
        merged_df = pd.merge(
            v8_df,
            v7_df,
            on='episode',
            how='inner'
        )
        if merged_df.empty:
            raise ValueError("No common episodes found between version 7 and version 8 CSVs")
        return merged_df[['episode', 'agent_reward', 'default_reward', 'agent_wait_total', 
                         'default_wait_total', 'agent_queue_total', 'default_queue_total']]
    except Exception as e:
        logging.error(f"Failed to merge data: {e}")
        raise

def plot_comparison_metrics(df):
    """Generate comparison plots for presentation"""
    try:
        plt.figure(figsize=(12, 8))
        
        # Reward comparison
        plt.subplot(2, 2, 1)
        plt.plot(df['episode'], df['agent_reward'], label='Agent', color='blue')
        plt.plot(df['episode'], df['default_reward'], label='Default', color='red')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Agent vs Default Reward')
        plt.legend()
        plt.grid(True)
        
        # Waiting time comparison
        plt.subplot(2, 2, 2)
        plt.plot(df['episode'], df['agent_wait_total'], label='Agent', color='blue')
        plt.plot(df['episode'], df['default_wait_total'], label='Default', color='red')
        plt.xlabel('Episode')
        plt.ylabel('Waiting Time (seconds)')
        plt.title('Waiting Time Comparison')
        plt.legend()
        plt.grid(True)
        
        # Queue length comparison
        plt.subplot(2, 2, 3)
        plt.plot(df['episode'], df['agent_queue_total'], label='Agent', color='blue')
        plt.plot(df['episode'], df['default_queue_total'], label='Default', color='red')
        plt.xlabel('Episode')
        plt.ylabel('Queue Length (vehicles)')
        plt.title('Queue Length Comparison')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('comparison_metrics.png')
        plt.close()
        logging.info("Comparison plots saved to 'comparison_metrics.png'")
    except Exception as e:
        logging.error(f"Failed to generate plots: {e}")
        raise

def main():
    """Main function to merge CSVs and generate plots"""
    try:
        v7_df = load_version7_data()
        v8_df = load_version8_data()
        merged_df = merge_data(v7_df, v8_df)
        merged_df.to_csv(OUTPUT_CSV, index=False)
        logging.info(f"Merged results saved to {OUTPUT_CSV}")
        plot_comparison_metrics(merged_df)
    except Exception as e:
        logging.error(f"Merge process failed: {e}")
        raise

if __name__ == "__main__":
    main()