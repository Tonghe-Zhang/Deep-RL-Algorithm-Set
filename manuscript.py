


import numpy as np

num_dist=10
len_dist=20

D=np.zeros((num_dist,len_dist))
for i in range(num_dist):
    D[i]=np.abs(np.random.randn(len_dist))
    D[i]/=sum(D[i])

print(D)
for i in range(num_dist):
    sample=np.random.choice(list(np.arange(len_dist)), size=1, p=D[i])



mean=0.3
variance=10
std=np.sqrt(variance)
num_samples=10
gaussian_sample_array=np.random.normal(loc=mean, scale=std,size=num_samples)
print(gaussian_sample_array)