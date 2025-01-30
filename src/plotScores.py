import matplotlib.pyplot as plt
import pandas as pd
import os

# Dateinamen
dateien = [
    "score_DQN_normiert_full_observationspace_corrected_padding.csv",
    "score_DQN_normiert_small_observationspace_corrected_padding.csv",
    "DQN_normiert_full_observationspace_corrected_padding_switchObjects_backup.csv",
    "score_DQN_norm_full_obssp_rew_dist_comparison_v01.csv",
    "score_DQN_normiert_full_observationspace_reward_changed.csv",
    "DQN_normiert_push_after_to_object_withNegativRewards.csv"
]

# Kürzere Titel
titel = [
    "DQN: Normal",
    "DQN: Small Observation Space",
    "DQN: Allow Switch Objects",
    "DQN: Reward Comparision RobToGoal vs ObjToGoal",
    "DQN: Reward Circle Function",
    "DQN: Push Negative Reward"
]

# Anzahl der Subplots bestimmen
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Daten plotten
for ax, datei, titel_text in zip(axes, dateien, titel):
    with open(f'../data/scores/{datei}') as f:
        lines = f.readlines()
    
    x = [float(line.strip()) for line in lines]
    x_series = pd.Series(x)
    rolling_mean = x_series.rolling(window=100).mean()
    
    ax.plot(rolling_mean, color='lightblue', label='Rollender Durchschnitt')
    ax.scatter(range(len(x)), x, color='darkblue', s=10, label='Originalwerte')
    ax.set_xlim(0, 500)
    ax.set_ylim(-11.5,15)
    ax.set_title(titel_text)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Erfüllungsgrad [%]")
    ax.legend()

plt.tight_layout()
plt.show()
