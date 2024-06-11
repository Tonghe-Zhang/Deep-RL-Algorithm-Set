import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def quick_plot():
    # Reading the CSV data into a dataframe
    data = pd.read_csv('policy_training_progress.csv')

    # Function to format axis in 1e6 scale
    def millions(x, pos):
        'The two args are the value and tick position'
        return '%1.1fM' % (x * 1e-6)

    # Plotting eval/normalized_episode_reward and its standard deviation
    plt.figure(figsize=(10, 6))
    plt.plot(data['timestep'], data['eval/normalized_episode_reward'], label='EDAC', color='blue')
    plt.fill_between(data['timestep'], 
                    data['eval/normalized_episode_reward'] - data['eval/normalized_episode_reward_std'], 
                    data['eval/normalized_episode_reward'] + data['eval/normalized_episode_reward_std'],
                    color='blue', alpha=0.2)
    plt.xlabel('Step')
    plt.ylabel('Normalized Episode Reward')
    plt.title('Normalized Episodic Reward')
    plt.legend()
    plt.grid(True)

    # Updating x-axis to scale of 1e6
    ax = plt.gca()
    formatter = ticker.FuncFormatter(millions)
    ax.xaxis.set_major_formatter(formatter)

    plt.savefig('return.png')
    plt.show()

    # Plotting loss of actor and critics on primary y-axis
    plt.figure(figsize=(10, 6))
    plt.plot(data['timestep'], data['loss/actor'], label='Actor Loss', color='red')
    plt.plot(data['timestep'], data['loss/critics'], label='Critics Loss', color='green')
    plt.xlabel('Step')
    plt.ylabel('Loss Actor/Critics')
    plt.title('Actor and Critic Loss')
    plt.legend(loc='upper center',ncol=2)
    plt.grid(True)

    # Updating x-axis to scale of 1e6
    ax = plt.gca()
    ax.xaxis.set_major_formatter(formatter)

    plt.savefig('loss_actor_critic.png')
    plt.show()

    # Plotting loss of alpha on secondary y-axis
    plt.figure(figsize=(10, 6))
    plt.plot(data['timestep'], data['loss/alpha'], label='Loss Alpha', color='orange')
    plt.xlabel('Step')
    plt.ylabel('Loss Alpha')
    plt.title('Alpha Loss')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Updating x-axis to scale of 1e6
    ax = plt.gca()
    ax.xaxis.set_major_formatter(formatter)

    plt.savefig('loss_alpha.png')
    plt.show()


if __name__=="__main__":
    quick_plot()