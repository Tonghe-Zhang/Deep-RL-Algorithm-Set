import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import matplotlib.cm as cm


def compare(trim_pct:list[float],max_normalized_returns:list[float]):
    # Create plot
    plt.figure(figsize=(10, 7))
    plt.plot(trim_pct, max_normalized_returns, marker='o')

    # Annotate the points
    for i, j in zip(trim_pct, max_normalized_returns):
        plt.annotate(f'{j:.2f}', xy=(i, j), textcoords="offset points", xytext=(0,10), ha='center')

    # Add labels and title
    plt.xlabel('Trimming Percentage')
    plt.ylabel('Normalized Episodic Reward')
    plt.ylim([10,60])
    plt.title('Trimming Percentage vs Normalized Episodic Reward')

    # Show the plot
    plt.grid(True)
    plt.savefig('./compare.png')
    plt.show()



def quick_plot(trim_pcts=[0.1,0.2,0.3,0.4,0.5,0.6,0.8]):


    max_returns=[]

    # Reading the CSV data into a dataframe
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
        max_returns.append(max(data['eval/normalized_episode_reward']))

    plt.xlabel('Step')
    plt.ylabel('Normalized Episode Reward')
    plt.title('Effect of Trimming Dataset on EDAC Algorithm')
    plt.legend(loc='upper right',ncol=2)
    
    plt.savefig("./returns.png")
    plt.show()
    
    print(f"max reward={max(data['eval/normalized_episode_reward'])}")
    
    

    # Plotting loss of actor and critics on primary y-axis

    # actor loss
    plt.figure(figsize=(10, 6))

    plt.subplot(1,3,1)
    for i, (trim_pct,data) in enumerate(data_pct.items()):
        plt.plot(data['timestep'], data['loss/actor'], label=''+str(trim_pct*100)+'%', color=colors[i])
    plt.xlabel('Step')
    plt.ylabel('Actor Loss')
    plt.title('Effect of Trimming on Actor Loss')
    plt.legend(loc='upper center', ncol=2)

    plt.subplot(1,3,2)
    for i, (trim_pct,data) in enumerate(data_pct.items()):
        plt.plot(data['timestep'], data['loss/critics'], label=''+str(trim_pct*100)+'%', color=colors[i])
    plt.xlabel('Step')
    plt.ylabel('Critic Loss')
    plt.title('Effect of Trimming on Critic Loss')
    plt.legend(loc='upper center', ncol=2)

    plt.subplot(1,3,3)
    for i, (trim_pct,data) in enumerate(data_pct.items()):
        plt.plot(data['timestep'], data['loss/alpha'], label=''+str(trim_pct*100)+'%', color=colors[i])
    plt.xlabel('Step')
    plt.ylabel('Alpha Loss')
    plt.title('Effect of Trimming on Alpha Loss')
    plt.legend(loc='lower right', ncol=2)
    
    plt.suptitle('Trimming\'s Effect on Losses')

    plt.savefig('./loss.png')
    plt.show()

    return trim_pcts, max_returns







if __name__=="__main__":
    trim_pcts, max_returns=quick_plot()
    compare(trim_pcts, max_returns)