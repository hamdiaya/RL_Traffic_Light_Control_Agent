import matplotlib.pyplot as plt
import pandas as pd
import os

# Load the CSV data
df = pd.read_csv('test_results.csv')

# Create the 'testing' folder if it doesn't exist
os.makedirs('testing', exist_ok=True)

# Define pastel color palette
pastel_red = '#FF9999'
pastel_blue = '#99CCFF'

# ---------- Time Series Plots (No markers) ----------
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

axs[0].plot(df['episode'], df['agent_reward'], label='Agent', color=pastel_red)
axs[0].plot(df['episode'], df['default_reward'], label='Default', color=pastel_blue)
axs[0].set_title('Reward over Episodes')
axs[0].set_xlabel('Episode')
axs[0].set_ylabel('Reward')
axs[0].legend()
axs[0].grid(True)
axs[0].set_xticks(range(0, 51, 10))

axs[1].plot(df['episode'], df['agent_wait_total'], label='Agent', color=pastel_red)
axs[1].plot(df['episode'], df['default_wait_total'], label='Default', color=pastel_blue)
axs[1].set_title('Total Wait Time over Episodes')
axs[1].set_xlabel('Episode')
axs[1].set_ylabel('Total Wait Time')
axs[1].legend()
axs[1].grid(True)
axs[1].set_xticks(range(0, 51, 10))
axs[1].set_yscale('log')

axs[2].plot(df['episode'], df['agent_queue_total'], label='Agent', color=pastel_red)
axs[2].plot(df['episode'], df['default_queue_total'], label='Default', color=pastel_blue)
axs[2].set_title('Total Queue over Episodes')
axs[2].set_xlabel('Episode')
axs[2].set_ylabel('Total Queue')
axs[2].legend()
axs[2].grid(True)
axs[2].set_xticks(range(0, 51, 10))

plt.tight_layout()
plt.savefig('testing/line_plots_over_episodes.png')
plt.show()

# ---------- Clean Bar Plot Function (No annotations) ----------
def bar_plot(title, ylabel, agent_val, default_val, filename):
    plt.figure(figsize=(6, 5))
    plt.bar(['Agent', 'Default'], [agent_val, default_val],
            color=[pastel_red, pastel_blue],
            edgecolor='black', linewidth=1.2)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'testing/{filename}')
    plt.show()

# Save separate bar plots
bar_plot('Average Reward Comparison', 'Average Reward',
         df['agent_reward'].mean(), df['default_reward'].mean(),
         'avg_reward_comparison.png')

bar_plot('Average Total Wait Time Comparison', 'Average Total Wait Time',
         df['agent_wait_total'].mean(), df['default_wait_total'].mean(),
         'avg_wait_time_comparison.png')

bar_plot('Average Total Queue Comparison', 'Average Total Queue',
         df['agent_queue_total'].mean(), df['default_queue_total'].mean(),
         'avg_queue_comparison.png')
