import matplotlib.pyplot as plt

# Data
datasets = ["Purely Original(1M Transitions)", "Optimal Mixture (1M Transitions)", "Purely Synthetic(1M Transitions)", "Purely Synthetic (5M Transitions)"]
rewards = [48.92, 51.23, 49.90, 53.13]

# Create bar graph
plt.figure(figsize=(10, 6))
bars=plt.bar(datasets, rewards, color=['lightblue','skyblue', 'darkblue', 'black'])

# Add labels and title
plt.ylim([47,55])
plt.xlabel('Dataset')
plt.ylabel('Normalized Episodic Reward')
plt.title('Effect of the Quality and Quantity of Synthetic Dataset')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, round(yval, 2), ha='center', va='bottom')

# Show the plot
plt.grid(axis='y')
plt.savefig('./reward_comparison.png')
plt.show()
