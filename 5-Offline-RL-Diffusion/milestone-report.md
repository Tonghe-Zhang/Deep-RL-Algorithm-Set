Slides

1. Literature

2. Design 

3. Basic Results:

   * CQL, BCQ, IQL  trim prt=0, 0.1, 0,2, 0.3, 0,4, 0.5  6  groups  45

   * SAC-n trim =0

4. Advanced results
   * Diffusion raw. 
   * **(tonight) Diffusion + Original mixture**  tbd

5. Future directions

 *  data augmentation  + model-based(transition)    

 *  diffusion ensemble  + uncertainty sampling 

 *  real-data fake data trust less 

    unseen state OOD: use fake data to generalize

​	unseen action OOD: pessimism only on real data

   















60 points

1. SAC-n full trj    70   good policy ==> 

   data quality not just distill parameters

   Ground truth Q value estimator in place of return (high variance)

   

2. CQL, BCQ, IQL  trim prt=0, 0.1, 0,2, 0.3, 0,4, 0.5    6 groups  45

   

80 points

1. Diffusion model:  conditioned on returns(from 2024 work, 30 steps traj)

   5.7k from 5k

   80+

   2023: unconditional high rewards generate single transition 

   Exploration

2. First diffusion enlarge dataset

   then trim the enlarged dataset

   or train a forward dynamics model to test and then trim





**5M diffusion dataset + original dataset combine**

ratio 5M



   



**0.3 M(high reward or random sampling) +  0.1M(orignal)**

3 + 1 ratio

























Future direction



importance sampling, reduce generated data retrieval probability

uncertainty negative bonus: see 











