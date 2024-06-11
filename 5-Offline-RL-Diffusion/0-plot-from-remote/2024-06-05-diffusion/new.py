import matplotlib.pyplot as plt

# Data
algorithms = ['BCQ', 'CQL', 'IQL', 'EDAC']
rewards=[48.052905-44.7468, 48.925809-47.3176, 46.298388-42.6244, 53.13192219152377-50.3049]

#[44.7468, 47.3176, 42.6244, 50.3049]
#[48.052905, 48.925809, 46.298388, 53.13192219152377]
# [, , , ] #



# Filter out algorithms with rewards below the threshold (43 in this case)
filtered_data = [(algo, reward) for algo, reward in zip(algorithms, rewards) if reward > 0]
filtered_algorithms, filtered_rewards = zip(*filtered_data)  # Unzip the filtered data

# Plot
plt.figure(figsize=(10, 6))  # Optional: Set figure size for better visualization
bars = plt.bar(filtered_algorithms, filtered_rewards, color='skyblue')

# Add value labels above bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

plt.xlabel('Algorithms')
plt.ylabel('Normalized Episodic Reward')
plt.title('Comparison of the Increase of Normalized Episodic Rewards for Different Algorithms using GTA')
plt.ylim(0, max(filtered_rewards) + 5)  # Set y-axis limits

plt.show()