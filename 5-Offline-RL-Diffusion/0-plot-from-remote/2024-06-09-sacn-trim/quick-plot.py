import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import matplotlib.cm as cm


def quick_plot():
    # Reading the CSV data into a dataframe

    trim_pcts=[0.2,0.4,0.6,0.8]

    data_pct={}
    for trim_pct in trim_pcts:
        data_pct[trim_pct]=pd.read_csv(os.path.join(str(trim_pct),'record','policy_training_progress.csv'))
    
    # Function to format axis in 1e6 scale
    def millions(x, pos):
        'The two args are the value and tick position'
        return '%1.1fM' % (x * 1e-6)
    # Plotting eval/normalized_episode_reward and its standard deviation
    plt.figure(figsize=(10, 6))

    cmap = cm.Blues
    colors = [cmap((0.7*i+2.1)/len(data_pct)) for i in range(len(data_pct))]

    for i, (trim_pct,data) in enumerate(data_pct.items()):
        plt.plot(data['timestep'], data['eval/normalized_episode_reward'], label='Trim '+str(trim_pct*100)+'%', color=colors[i])
        plt.fill_between(data['timestep'], 
                        data['eval/normalized_episode_reward'] - data['eval/normalized_episode_reward_std'], 
                        data['eval/normalized_episode_reward'] + data['eval/normalized_episode_reward_std'],
                        color=colors[i], alpha=0.2)
    plt.xlabel('Step')
    plt.ylabel('Normalized Episode Reward')
    plt.title('Effect of Trimming Dataset on EDAC Algorithm')
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
    for i, (trim_pct,data) in enumerate(data_pct.items()):
        plt.plot(data['timestep'], data['loss/actor'], label='Trim '+str(trim_pct*100)+'%', color=colors[i])
    plt.xlabel('Step')
    plt.ylabel('Actor Loss')
    plt.title('Effect of Trimming on Actor Loss')
    plt.legend(loc='upper center', ncol=2)

    plt.subplot(1,3,2)
    for i, (trim_pct,data) in enumerate(data_pct.items()):
        plt.plot(data['timestep'], data['loss/critics'], label='Trim '+str(trim_pct*100)+'%', color=colors[i])
    plt.xlabel('Step')
    plt.ylabel('Critic Loss')
    plt.title('Effect of Trimming on Critic Loss')
    plt.legend(loc='upper center', ncol=2)

    plt.subplot(1,3,3)
    for i, (trim_pct,data) in enumerate(data_pct.items()):
        plt.plot(data['timestep'], data['loss/alpha'], label='Trim '+str(trim_pct*100)+'%', color=colors[i])
    plt.xlabel('Step')
    plt.ylabel('Alpha Loss')
    plt.title('Effect of Trimming on Alpha Loss')
    plt.legend(loc='lower right', ncol=2)
    
    plt.suptitle('Trimming\'s Effect on Losses')

    plt.savefig('./loss.png')
    plt.show()


if __name__=="__main__":
    quick_plot()