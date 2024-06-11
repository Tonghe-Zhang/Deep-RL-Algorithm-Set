import matplotlib.pyplot as plt
import numpy as np

# Data
trimming_ratios = [0, 0.2, 0.4, 0.6, 0.8]
returns = [50.3, 47.5, 47.4, 23.5, 6.74]

# Filter data
filtered_trimming_ratios = [ratio for ratio, ret in zip(trimming_ratios, returns) if ret > 3.0]
filtered_returns = [ret for ret in returns if ret > 3.0]

# Create line plot
plt.plot(filtered_trimming_ratios, filtered_returns, color='skyblue', marker='o')

# Add labels
plt.xlabel('Trimming Ratio')
plt.ylabel('Normalized Episodic Return')
plt.title('EDAC Algorithm Performance')

# Add return values on top of bars
for i, ret in enumerate(filtered_returns):
    plt.text(filtered_trimming_ratios[i], ret + 0.5, str(ret), ha='center')

# Leave some blank between different bars
plt.xticks(np.arange(min(filtered_trimming_ratios), max(filtered_trimming_ratios)+0.2, 0.2))

plt.savefig("./trim_mean_return.png")

plt.show()
