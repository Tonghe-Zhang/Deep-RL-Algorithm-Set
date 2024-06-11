import matplotlib.pyplot as plt
import numpy as np

# Data
algorithms = ['BCQ', 'CQL', 'IQL', 'EDAC']
with_gta = [48.052905, 48.925809, 46.298388, 53.13192219152377]
without_gta = [44.7468, 47.3176, 42.6244, 50.30492076737762]

# Filtered data according to the condition (showing only values above 43)
with_gta = [value if value >= 43 else None for value in with_gta]
without_gta = [value if value >= 43 else None for value in without_gta]

# Setting the positions and width for the bars
pos = np.arange(len(algorithms))
width = 0.35

# Create the plot
fig, ax = plt.subplots()

# Plotting the data
bars1 = ax.bar(pos - width/2, with_gta, width, label='With GTA', color='b')
bars2 = ax.bar(pos + width/2, without_gta, width, label='Without GTA', color='r')

# Adding labels, title, and custom x-axis tick labels, etc.
ax.set_ylabel('Normalized Episodic Rewards')
ax.set_title('Performance of Algorithms with and without GTA')
ax.set_xticks(pos)
ax.set_xticklabels(algorithms)
ax.legend()

# Adding bar values on top
for bar in bars1:
    if bar.get_height() > 0:  # Check to avoid displaying None
        height = bar.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

for bar in bars2:
    if bar.get_height() > 0:  # Check to avoid displaying None
        height = bar.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Display the plot
plt.ylim(top=60)  # Set a reasonable y limit to give some space above the bars
plt.show()
