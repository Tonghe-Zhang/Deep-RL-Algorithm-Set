import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def compare(syn_ratios:list[float],
            max_normalized_returns:list[float]):
    # Data
    # syn_ratios = [0.0, 0.2, 0.4, 0.6, 0.8]  # 1.0
    # max_normalized_returns = [50.3, 52.11251273478703, 51.14964126116913, 48.97281590001943, 48.28718077209817] #, 53.13

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(syn_ratios, max_normalized_returns, marker='o')

    # Add labels and title
    plt.xlabel('Synthetic Transition Ratio')
    plt.ylabel('Normalized Episodic Reward')
    plt.title('Synthetic Transition Ratio vs Normalized Episodic Reward')

    # Show the plot
    plt.grid(True)
    plt.savefig('./compare.png')
    plt.show()

def plot_returns_loss(syn_ratios=[0.2,0.4,0.6,0.8]):
    # Reading the CSV data into a dataframe

    max_normalized_returns=[]

    data_pct={}
    for syn_ratio in syn_ratios:
        data_pct[syn_ratio]=pd.read_csv(os.path.join(str(syn_ratio),'record','policy_training_progress.csv'))
    
    # Function to format axis in 1e6 scale
    def millions(x, pos):
        'The two args are the value and tick position'
        return '%1.1fM' % (x * 1e-6)
    # Plotting eval/normalized_episode_reward and its standard deviation
    plt.figure(figsize=(12, 9))

    cmap = cm.Blues
    colors = [cmap((0.7*i+2.1)/len(data_pct)) for i in range(len(data_pct))]

    for i, (syn_ratio,data) in enumerate(data_pct.items()):
        plt.plot(data['timestep'], data['eval/normalized_episode_reward'], label='Synthetic Ratio='+str(syn_ratio*100)+'%', color=colors[i])
        plt.fill_between(data['timestep'], 
                        data['eval/normalized_episode_reward'] - data['eval/normalized_episode_reward_std'], 
                        data['eval/normalized_episode_reward'] + data['eval/normalized_episode_reward_std'],
                        color=colors[i], alpha=0.2)
        max_normalized_returns.append(max(data['eval/normalized_episode_reward']))
    plt.xlabel('Step')
    plt.ylabel('Normalized Episode Reward')
    plt.title('Effect of Synthetic Dataset Dataset on EDAC Algorithm')
    plt.legend()
    
    plt.savefig("./returns.png")
    plt.show()
    
    print(f"max reward={max(data['eval/normalized_episode_reward'])}")
    
    #plt.grid(True)

    # Updating x-axis to scale of 1e6
    ax = plt.gca()
    formatter = ticker.FuncFormatter(millions)
    ax.xaxis.set_major_formatter(formatter)

    # Plotting loss of actor and critics on primary y-axis

    # actor loss
    plt.figure(figsize=(10, 6))

    plt.subplot(1,3,1)
    for i, (syn_ratio,data) in enumerate(data_pct.items()):
        plt.plot(data['timestep'], data['loss/actor'], label='Syn Rate='+str(syn_ratio*100)+'%', color=colors[i])
    plt.xlabel('Step')
    plt.ylabel('Actor Loss')
    plt.title('Effect of Synthetic Dataset on Actor Loss')
    plt.legend(loc='upper center', ncol=2)

    plt.subplot(1,3,2)
    for i, (syn_ratio,data) in enumerate(data_pct.items()):
        plt.plot(data['timestep'], data['loss/critics'], label='Syn Rate='+str(syn_ratio*100)+'%', color=colors[i])
    plt.xlabel('Step')
    plt.ylabel('Critic Loss')
    plt.title('Effect of Synthetic Dataset on Critic Loss')
    plt.legend(loc='upper center', ncol=2)

    plt.subplot(1,3,3)
    for i, (syn_ratio,data) in enumerate(data_pct.items()):
        plt.plot(data['timestep'], data['loss/alpha'], label='Syn Rate='+str(syn_ratio*100)+'%', color=colors[i])
    plt.xlabel('Step')
    plt.ylabel('Alpha Loss')
    plt.title('Effect of Synthetic Dataset on Alpha Loss')
    plt.tight_layout()
    plt.legend(loc='lower right', ncol=2)

    plt.suptitle('Synthetic Dataset\'s Effect on Losses')

    plt.savefig('./loss.png')
    plt.show()

    return max_normalized_returns

if __name__=="__main__":
    syn_ratios=[0.0,0.2,0.4,0.6,0.8,1.0]
    max_normalized_returns=plot_returns_loss(syn_ratios=syn_ratios)
    compare(syn_ratios=syn_ratios,max_normalized_returns=max_normalized_returns)
