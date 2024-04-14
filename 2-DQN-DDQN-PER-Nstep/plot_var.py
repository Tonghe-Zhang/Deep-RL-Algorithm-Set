


import numpy as np
import matplotlib.pyplot as plt

log_file_name='DuelingQ.txt'
experiment_name='DuelingQNetwork and NormalReplayBuffer'


log_file_name='PER.txt'
experiment_name='Prioritized Experience Replay'

log_file_dir='evaluation_std_plot/'

def current_time_str()->str:
    import datetime
    return str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')

with open(log_file_dir+log_file_name,mode='r') as f:
    ins=f.readlines()
    indices=np.zeros(len(ins))
    mean=np.zeros(len(ins))
    std=np.zeros(len(ins))
    for i in range(len(ins)):
        ins_s=ins[i][:-1].split(' ')
        indices[i]=float(ins_s[6][:-1])
        mean[i]=float(ins_s[-4][:-1])   
        std[i]=float(ins_s[-1][:-1])
    plt.figure()
    plt.xlabel('step')
    plt.plot(indices,mean,label='Evaluation Mean')
    plt.plot(indices,std,label='Evaluation Standard Deviation')
    plt.title('Evaluation Result of \n'+experiment_name)
    plt.legend(loc='upper right')
    plt.ylim([0,max(mean)*1.2])
    plt.savefig('evaluation_std_plot/'+current_time_str()+log_file_name[:-4]+'.png')
    plt.show()
    