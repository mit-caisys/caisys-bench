import pandas as pd
import matplotlib.pyplot as plt

file_0 = '/home/cerangel/comp-ai-bench/video-qa/data/experiment_results/100_10_0.5_seq_shm/experiment_results/monitoring_logs/vllm_cache_log.csv'
file_1 = '/home/cerangel/comp-ai-bench/video-qa/data/experiment_results/100_10_0.5_rand_shm/experiment_results/monitoring_logs/vllm_cache_log.csv'

try:
    df_sticky = pd.read_csv(file_0)
    df_random = pd.read_csv(file_1)
except Exception as e:
    print(f"Error loading files: {e}")
    exit()

FIG_SIZE = (6, 3)
LABEL_FONT = 14
LEGEND_FONT = 12

plt.figure(figsize=FIG_SIZE)

plt.plot(df_sticky['elapsed_sec'], df_sticky['running_avg_hit_rate'] * 100, label='Sticky Routing')
plt.plot(df_random['elapsed_sec'], df_random['running_avg_hit_rate'] * 100, label='Random Routing')

plt.xlabel('Elapsed Time (sec)', fontsize=LABEL_FONT)
plt.ylabel('KV Cache Hit Rate (%)', fontsize=LABEL_FONT)
plt.legend(fontsize=LEGEND_FONT)

plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('kv_running_avg_hit_rate_updated.pdf')
plt.close()

plt.figure(figsize=FIG_SIZE)

plt.plot(df_sticky['elapsed_sec'], df_sticky['mm_running_avg_hit_rate'] * 100, label='Sticky Routing')
plt.plot(df_random['elapsed_sec'], df_random['mm_running_avg_hit_rate'] * 100, label='Random Routing')

plt.xlabel('Time (sec)', fontsize=LABEL_FONT)
plt.ylabel('MM Cache Hit Rate (%)', fontsize=LABEL_FONT)
plt.legend(fontsize=LEGEND_FONT)

plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('mm_running_avg_hit_rate_updated.pdf')
plt.close()

print("Plots generated successfully with updated formatting.")