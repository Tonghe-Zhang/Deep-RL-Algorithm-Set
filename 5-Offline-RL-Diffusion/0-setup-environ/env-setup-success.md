[toc]

## Install mujoco environment using conda

#### Preparation

```cmd
pip install 'cython<3.0.0' -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```cmd
sudo apt-get install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

#### Download the Mujoco library from this [link](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz), and unzip and then zip it as “mujoco210.zip”

- ##### Create a hidden folder in the remote server

```
mkdir /root/.mujoco
```

- Upload mujoco210.zip to the hidden folder and then unzip it . 

* run these lines

```
echo -e 'export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia 
export PATH="$LD_LIBRARY_PATH:$PATH" 
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so' >> ~/.bashrc
```

- Source bashrc.

```
source ~/.bashrc
```

- Test that the library is installed.

  

#### Install mujoco-py.

```
conda create --name mujoco_py python=3.8
```

you will see a series of outputs, press y to continue whenever they ask you to decide.

```cmd
conda init bash
conda activate mujoco_py
```

and then you will go into the mujoco environment. 

(mojoco_py) root @…

##### then

```cmd
sudo apt update
sudo apt-get install patchelf
sudo apt-get install python3-dev build-essential libssl-dev libffi-dev libxml2-dev  
sudo apt-get install libxslt1-dev zlib1g-dev libglew1.5 libglew-dev python3-pip
```

##### Continue by

```cmd
cd ~/.mujoco
    git clone https://github.com/openai/mujoco-py
cd mujoco-py
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.dev.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install -e . --no-cache -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### Close your terminal, then start a new one, run

```
conda activate mujoco_py
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
pip3 install -U 'mujoco-py<2.2,>=2.1'
```

#### Check if mujoco-py is properly installed.

```cmd
python3
import mujoco_py
mujoco_py.__version__
```



#### How to get into your mujoco environment again after installation

##### First check the environments for conda

```cmd
conda init bash
conda info --envs
```

##### Output should be: 

base                  *  /root/miniconda3
mujoco_py                /root/miniconda3/envs/mujoco_py

##### And then:

```cmd
conda activate mujoco_py
```

#### First time activate you should add the env var

```cmd
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
```





## In your mujoco_py environment, Install d3rl py

```cmd
pip install d3rlpy==1.1.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### Result: 

```cmd
import d3rlpy
d3rlpy.__version__
'2.5.0'
```



## In your mujoco_py environment, Install d4rl py

#### Run these commands

```cmd
git clone https://github.com/Farama-Foundation/D4RL.git
```

```cmd
cd D4RL
```

```cmd
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```cmd
cd ..
```

#### Then test your environment

```cmd
import d4rl
```

#### Download halfcheetah environment to the d4rl package for the first time

```cmd
http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium-v2.hdf5 to /root/.d4rl/datasets/halfcheetah_medium-v2.hdf5
```

it could take some time, since this file is of size 227 MB. 

So you can first dowload it from

```cmd
http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium-v2.hdf5
```

Zip the file and upload to autodl remote server, 

then unzip it in 

```cmd
/root/.d4rl/datasets/halfcheetah_medium-v2.hdf5
```



## Test all the packages installed properly

Start  a  new terminal, then run

```cmd
conda init bash
conda activate mujoco_py
python3
```

then run in python script

```python
import mujoco
mujoco.__version__
import gym
gym.__version__
import mujoco_py
mujoco_py.__version__
import d4rl
d4rl.__version__
import d3rlpy
d3rlpy.__version__
```

cd d3rlpy
python reproductions/offline/bcq.py --dataset='halfcheetah-medium-replay-v0' --gpu=0



## Additional packages for running SAC-n

```cmd
pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple
```

ALso Upload OfflineRL-Kit-main.zip to /root/autodl-fs/ and then unzip it. 

Run these commands



## Run  SAC-n

In 

(mujoco_py) root@autodl-container-304c4b8481-d28e79ef:~/autodl-fs/OfflineRL-Kit-main# 

run

```cmd
nohup python run_edac.py --num-critics 10 --eta 1.0 --task halfcheetah-medium-v2 &
```

It costs you 7.5 hours. 

Our use this in any terminal: 

```cmd
cd /root/autodl-fs/OfflineRL-Kit-main
conda activate mujoco_py
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
nohup python run_edac.py --num-critics 10 --eta 1.0 --task halfcheetah-medium-v2 &
```



## Plot SAC-n



## Bug fixes:

##### Permission denied

root@autodl-container-b30e11b252-9e540512:~/.mujoco/mujoco210/bin# ./simulate ../model/humanoid.xml
bash: ./simulate: Permission denied

``` bash
chmod +x ./simulate
```

source /etc/autodl-motd

2. ```
   cat ~/.bashrc
   ```







