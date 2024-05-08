# Policy Gradient for MDP and the PPO implementation

Fix MDP model. The spaces are continuous. The value function is defined below:
$$
V^{\pi_\theta}=\underset{S_0\sim \mu(\cdot), A_t\sim \pi_\theta(\cdot|S_t), S_{t+1}\sim P(\cdot|S_t, A_t), \forall t\in \Z_{\geq 0}}{\mathbb{E}}\left[\sum_{t=0}^{\infty}\gamma^t~r_t(S_t, A_t)\right]
$$
Denote by $\tau=(s_0, a_0, \ldots, s_t, a_t, \ldots) \in \left(\mathcal{S\times A}\right)^{\infty}$ and $R(\tau):=\sum_{t=0}^{\infty}\gamma^t~r_t(S_t, A_t)\bigg|_{\tau}$, then we have
$$
\begin{equation}
\begin{aligned}
\nabla_{\theta} V^{\pi_\theta}\bigg|_{\theta_t}=&
\nabla_{\theta}\int d\tau ~\mathbb{P}_{\mathcal{M}}^{\pi_\theta}(\tau) \left[\sum_{t=0}^{\infty}\gamma^t~r_t(S_t, A_t)\right]
\bigg|_{\theta_t}
\\=&
\int d\tau ~\nabla_{\theta}e^{\ln \mu(s_0)\prod_{k=0}^{\infty}\pi_\theta(a_t|s_t)P(s_{t+1}|s_t,a_t)}\left[\sum_{t=0}^{\infty}\gamma^t~r_t(S_t, A_t)\right]
\bigg|_{\theta_t}
\\=&
\int d\tau ~
\mathbb{P}_{\mathcal{M}}^{\pi_\theta}(\tau)
\nabla_{\theta} \left(\ln \mu(s_0)+\sum_{k=0}^{\infty}\ln \pi_\theta(a_kt|s_k)+\sum_{k=0}^{\infty}\ln P(s_{k+1}|s_k,a_k)\right)
\left[\sum_{t=0}^{\infty}\gamma^t~r_t(S_t, A_t)\right]
\bigg|_{\theta_t}
\\=&
\int d\tau ~
\mathbb{P}_{\mathcal{M}}^{\pi_\theta}(\tau)
\left(\sum_{k=0}^{\infty}\nabla_{\theta} \ln \pi_\theta(a_k|s_k)\right)
\left[\sum_{t=0}^{\infty}\gamma^t~r_t(S_t, A_t)\right]
\bigg|_{\theta_t}
\\=&
\mathbb{E}_{\mathcal{M}}^{\pi_\theta}
\left(\sum_{k=0}^{\infty}\nabla_{\theta} \ln \pi_\theta(A_k|S_k)\right)
\left[\sum_{t=0}^{\infty}\gamma^t~r_t(S_t, A_t)\right]
\bigg|_{\theta_t}
\\=&
\mathbb{E}_{\mathcal{M}}^{\pi_\theta}
\left[R(\tau)\sum_{k=0}^{\infty}\nabla_{\theta} \ln \pi_\theta(A_k|S_k)\right]
\bigg|_{\theta_t}
\end{aligned}
\end{equation}
$$
This is the REINFORCE algorithm. Directly executing Monte-Carlo estimate of this expression is too complicated and high-variance. We consider further simplifying the expression using the Markovian property of the policy and the MDP chain.

Starting from the penultimate line of the derivation above, using iterated expectation formula, we obtain
$$
\begin{equation}
\begin{aligned}
\nabla_{\theta} V^{\pi_\theta}\bigg|_{\theta_t}
=&
\mathbb{E}_{\mathcal{M}}^{\pi_\theta}
\left(\sum_{k=0}^{\infty}\nabla_{\theta} \ln \pi_\theta(A_k|S_k)\right)
\left[\sum_{t=0}^{\infty}\gamma^t~r_t(S_t, A_t)\right]
\bigg|_{\theta_t}
\\=&
\sum_{k=0}^{\infty}\gamma^k~
\mathbb{E}_{\mathcal{M}}^{\pi_\theta}
\left[
\nabla_{\theta} \ln \pi_\theta(A_k|S_k)
\sum_{t=0}^{\infty}\gamma^{t-k}~r_t(S_t, A_t)
\right]
\bigg|_{\theta_t}
\\=&
\sum_{k=0}^{\infty}\gamma^k~
\mathbb{E}_{\mathcal{M}}^{\pi_\theta}
\mathbb{E}_{\mathcal{M}}^{\pi_\theta}\left[
\nabla_{\theta} \ln \pi_\theta(A_k|S_k)
\sum_{t=0}^{\infty}\gamma^{t-k}~r_t(S_t, A_t)
\bigg|S_k, A_k\right]
\bigg|_{\theta_t}
\\=&
\sum_{k=0}^{\infty}\gamma^k~
\left(\sum_{t=0}^{k-1}
+
\sum_{t=k}^{\infty}\right)
\gamma^{t-k}
\mathbb{E}_{\mathcal{M}}^{\pi_\theta}
\nabla_{\theta} \ln \pi_\theta(A_k|S_k)
\mathbb{E}_{\mathcal{M}}^{\pi_\theta}\left[r_t(S_t, A_t)
\bigg|S_k, A_k\right]
\bigg|_{\theta_t}
\end{aligned}
\end{equation}
$$


We will soon show that
$$
\begin{equation}
\begin{aligned}
\sum_{t=0}^{k-1}
\gamma^{t-k}
\mathbb{E}_{\mathcal{M}}^{\pi_\theta}
\nabla_{\theta} \ln \pi_\theta(A_k|S_k)
\mathbb{E}_{\mathcal{M}}^{\pi_\theta}\left[r_t(S_t, A_t)
\bigg|S_k, A_k\right]
\bigg|_{\theta_t}=0
\end{aligned}
\end{equation}
$$
Indeed, due to **the Markov property of the policy**,  we have
$$
{\color{red}{S_{k\lt t}, A_{k\lt t}} \perp  A_{k} ~|~ S_{k}}
$$

> However this is not the case for $S_{k\geq t}, A_{k\geq t}$, in that they are not only dependent on the historic state $S_k$, but also reliant on the previous action $A_k$, which owns to the fact that the MDP is a **controlled** Markov chain. Precisely speaking, 
>
> $\mathbb{E}_{\mathcal{M}}^{\pi_\theta}\left[r_t(S_t, A_t)\bigg|S_k, A_k\right] = \mathbb{E}_{\mathcal{M}}^{\pi_\theta}\left[r_t(S_t, A_t)\bigg|S_k\right]$ holds only for $0\leq t\lt k$.  

The conditional independence relation enables us to avoid integrating over $\mathcal{S}$ when evaluating the innermost conditional expectation, which paves way for the subsequent normalization trick (with the help of Fubini’s theorem)
$$
\begin{equation}
\begin{aligned}
\forall t\lt k: \quad 
&
\mathbb{E}_{\mathcal{M}}^{\pi_\theta}
\nabla_{\theta} \ln \pi_\theta(A_k|S_k)
\mathbb{E}_{\mathcal{M}}^{\pi_\theta}\left[r_t(S_t, A_t)
{\color{red}{\bigg|S_k, A_k}}\right]
\bigg|_{\theta_t}
\\=&
\mathbb{E}_{\mathcal{M}}^{\pi_\theta}
\frac{\nabla_{\theta}~ \pi_\theta(a_k|s_k)}{\pi_\theta(s_k|s_k)}
\mathbb{E}_{\mathcal{M}}^{\pi_\theta}\left[r_t(S_t, A_t)
{\color{red}{\bigg|S_k}}\right]
\bigg|_{\theta_t}
\\=&
\int_{\mathcal{S}}ds_k~\mathbb{P}_{\mathcal{M}}^{\pi_\theta}(S_k=s_k)
\int_{\mathcal{S}}da_k~\cancel{\pi_{\theta}(a_k|s_k)}
\frac{\nabla_{\theta}~ \pi_\theta(a_k|s_k)}{\cancel{\pi_{\theta}(a_k|s_k)}}
\underbrace{\mathbb{E}_{\mathcal{M}}^{\pi_\theta}\left[r_t(s_t, a_t){\color{red}{\bigg|S_k}}\right]}_{\text{Irrelevant with $A_k$}}
\bigg|_{\theta_t}
\\=&
\int_{\mathcal{S}}ds_k~\mathbb{P}_{\mathcal{M}}^{\pi_\theta}(S_k=s_k)
\underbrace{\mathbb{E}_{\mathcal{M}}^{\pi_\theta}\left[r_t(s_t, a_t){\color{red}{\bigg|S_k}}\right]}_{\text{Irrelevant with $A_k$}}
\nabla_{\theta}~ 
\underbrace{\int_{\mathcal{S}}da_k~
\pi_\theta(a_k|s_k)}_{\text{Normalize to constant 1}}
\bigg|_{\theta_t}
\\=&
0
\end{aligned}
\end{equation}
$$
Taking this observation back to the policy gradient’s expression, we obtain
$$
\begin{equation}
\begin{aligned}
\nabla_{\theta} V^{\pi_\theta}\bigg|_{\theta_t}
=&
\sum_{k=0}^{\infty}\gamma^k~
\left({\color{red}{\cancel{\sum_{t=0}^{k-1}}}}
+
\sum_{t=k}^{\infty}\right)
\gamma^{t-k}
\mathbb{E}_{\mathcal{M}}^{\pi_\theta}
\nabla_{\theta} \ln \pi_\theta(A_k|S_k)
\mathbb{E}_{\mathcal{M}}^{\pi_\theta}\left[r_t(S_t, A_t)
\bigg|S_k, A_k\right]
\bigg|_{\theta_t}
\\=&
\sum_{k=0}^{\infty}\gamma^k~
\mathbb{E}_{\mathcal{M}}^{\pi_\theta}
\nabla_{\theta} \ln \pi_\theta(A_k|S_k)
\underbrace{
\sum_{t=k}^{\infty}
\gamma^{t-k}
\mathbb{E}_{\mathcal{M}}^{\pi_\theta}\left[r_t(S_t, A_t)\bigg|S_k, A_k\right]
}_{\text{Define as $Q_k^{\pi_\theta}(S_k, A_k)$}}
\bigg|_{\theta_t}
\\=&
\mathbb{E}_{\mathcal{M}}^{\pi_\theta}
\sum_{k=0}^{\infty}\gamma^k~
\nabla_{\theta} \ln \pi_\theta(A_k|S_k)
Q_k^{\pi_\theta}(S_k, A_k)
\bigg|_{\theta_t}
\end{aligned}
\end{equation}
$$
Now we come to the Action-value expression of the policy gradient. Notice that we have not posed any stationary assumption on the transition kernels nor the policy until now.

> We have just defined the Q-function for infinite-horizon MDP in its broadest sense. The definition
> $$
> Q_k^{\pi_\theta}(S_k, A_k):=\mathbb{E}_{\mathcal{M}}^{\pi_\theta}\left[\sum_{t=k}^{\infty}
> \gamma^{t-k}~r_t(S_t, A_t)\bigg|S_k, A_k\right]
> $$
> adapts to non-stationary processes as well.

We also notice that for any function series $\{B_k(\cdot;\theta):\mathcal{S}\to \R\}_{k\in \Z_{\geq 0}}$ that does not explicitly rely on actions, we can use the normalization trick again to obtain
$$
\begin{equation}
\begin{aligned}
\mathbb{E}_{\mathcal{M}}^{\pi_\theta}\nabla_{\theta} \ln \pi_\theta(A_k|S_k)B_k(S_k;\theta)\bigg|_{\theta_t}
=&
\int_{\mathcal{S}}ds_k~\mathbb{P}_{\mathcal{M}}^{\pi_\theta}(S_k=s_k)
\int_{\mathcal{S}}da_k~\cancel{\pi_{\theta}(a_k|s_k)}
\frac{\nabla_{\theta}~ \pi_\theta(a_k|s_k)}{\cancel{\pi_{\theta}(a_k|s_k)}}
B_k(S_k;\theta)\bigg|_{\theta_t}
\\
=&
\int_{\mathcal{S}}ds_k~\mathbb{P}_{\mathcal{M}}^{\pi_\theta}(S_k=s_k)
B_k(S_k;\theta)\nabla_{\theta}~\underbrace{\int_{\mathcal{S}}da_k~ \pi_\theta(a_k|s_k)}_{\text{Normalize to constant 1}}\bigg|_{\theta_t}
=0
\end{aligned}
\end{equation}
$$
Instantiating $B_k(s;\theta)$ as the value function
$$
V_k^{\pi_\theta}(S_k):=\mathbb{E}_{\mathcal{M}}^{\pi_\theta}\left[\sum_{t=k}^{\infty}
\gamma^{t-k}~r_t(S_t, A_t)\bigg|S_k\right]
$$
and denote by $A_k^{\pi_\theta}(S_k,A_k)$ the difference between $V$ and $Q$: 
$$
A_k^{\pi_\theta}(S_k,A_k):=Q_k^{\pi_\theta}(S_k, A_k)-V_k^{\pi_\theta}(S_k) 
=
Q_k^{\pi_\theta}(S_k, A_k)-\mathbb{E}_{A_k\sim \pi_\theta(\cdot|S_k)}Q_k^{\pi_\theta}(S_k, A_k)
$$
We further conclude that 
$$
\begin{equation}
\begin{aligned}
\nabla_{\theta} V^{\pi_\theta}\bigg|_{\theta_t}
=&
\mathbb{E}_{\mathcal{M}}^{\pi_\theta}
\sum_{k=0}^{\infty}\gamma^k~
\nabla_{\theta} \ln \pi_\theta(A_k|S_k)
Q_k^{\pi_\theta}(S_k, A_k)
\bigg|_{\theta_t}{\color{red}{-0}}
\\=&
\mathbb{E}_{\mathcal{M}}^{\pi_\theta}
{\color{red}{\sum_{k=0}^{\infty}\gamma^k~
\nabla_{\theta} \ln \pi_\theta(A_k|S_k)}}
\left(Q_k^{\pi_\theta}(S_k, A_k)
{\color{red}{-V_k^{\pi_\theta}(S_k)}}\right)
\bigg|_{\theta_t}
\\=&
\mathbb{E}_{\mathcal{M}}^{\pi_\theta}
\sum_{k=0}^{\infty}\gamma^k~
\nabla_{\theta} \ln \pi_\theta(A_k|S_k)
~A_k^{\pi_\theta}(S_k, A_k)
\bigg|_{\theta_t}
\end{aligned}
\end{equation}
$$
Now we come to the Advantage-function expression of the policy gradient. **Notice that all the derivation so far has not relied on stationary assumptions on the kernels, rewards nor the policies. So all these results apply to non-stationary MDPs.**



#### Policy Gradient for Stationary MDP

In what follows we focus on studying PG theorem for stationary MDPs. We will start from the expression of the Q-function:
$$
Q_k^{\pi_\theta}(S_k, A_k):=\mathbb{E}_{\mathcal{M}}^{\pi_\theta}\left[\sum_{t=k}^{\infty}
\gamma^{t-k}~r_t(S_t, A_t)\bigg|S_k, A_k\right] 
$$
Fix $k$. Substitute $t$ with $t+k$, we can further simplify the Q-function’s expression as
$$
Q_k^{\pi_\theta}(S_k, A_k):=\mathbb{E}_{\mathcal{M}}^{\pi_\theta}\left[\sum_{t=0}^{\infty}
\gamma^{t}~r_{t+k}(S_{t+k}, A_{t+k})\bigg|S_k, A_k\right]
=r_k(S_k, A_k)+\gamma ~\mathbb{E}_{\mathcal{M}}^{\pi_\theta}~\left[r_{k+1}(S_{k+1}, A_{k+1}) \mid S_k, A_k\right] + \ldots
$$
**If we suppose that the rewards are stationary i.e. $r_{t+k}(\cdot,\cdot)=r_k(\cdot, \cdot):=r(\cdot, \cdot)$ and the dynamics and policy are also stationary,**  we can assert that
$$
\mathbb{P}_{\mathcal{M}}^{\pi_\theta}(S_{t+k}, A_{t+k} | S_k=s, A_k=a)~r_{t+k}(S_{t+k}, A_{t+k}) = 
\mathbb{P}_{\mathcal{M}}^{\pi_\theta}(S_{t}, A_{t} | S_0=s, A_0=a)~r_{t}(S_{t}, A_{t})
$$
which then implies
$$
Q^{\pi_\theta}_k(s,a):=\mathbb{E}_{\mathcal{M}}^{\pi_\theta}\left[\sum_{t=0}^{\infty}
\gamma^{t}~r(S_{t}, A_{t})\bigg|S_0=s, A_0=a\right]
$$
and we also observe that this expression becomes irrelevant with $k$, so we will drop the subscript of $Q$ under the stationary assumption. **For stationary MDP,** we define
$$
Q^{\pi_\theta}(s,a):=\mathbb{E}_{\mathcal{M}}^{\pi_\theta}\left[\sum_{t=0}^{\infty}
\gamma^{t}~r(S_{t}, A_{t})\bigg|S_0=s, A_0=a\right]\\
V^{\pi_\theta}(s):=\mathbb{E}_{\mathcal{M}}^{\pi_\theta}\left[\sum_{t=0}^{\infty}
\gamma^{t}~r(S_{t}, A_{t})\bigg|S_0=s\right]\\
A^{\pi_\theta}(s,a):=\mathbb{E}_{\mathcal{M}}^{\pi_\theta}\left[\sum_{t=0}^{\infty}
\gamma^{t}~r(S_{t}, A_{t})\bigg|S_0=s, A_0=a\right]\\
$$
Then we can write the policy gradient as
$$
\begin{equation}
\begin{aligned}
\nabla_{\theta} V^{\pi_\theta}\bigg|_{\theta_t}
=
\mathbb{E}_{\mathcal{M}}^{\pi_\theta}
\sum_{k=0}^{\infty}\gamma^k~
\nabla_{\theta} \ln \pi_\theta(A_k|S_k)
Q^{\pi_\theta}(S_k, A_k)
\bigg|_{\theta_t}
\end{aligned}
\end{equation}
$$
Similarly we have 
$$
\begin{equation}
\begin{aligned}
\nabla_{\theta} V^{\pi_\theta}\bigg|_{\theta_t}
=
\mathbb{E}_{\mathcal{M}}^{\pi_\theta}
\sum_{k=0}^{\infty}\gamma^k~
\nabla_{\theta} \ln \pi_\theta(A_k|S_k)
A^{\pi_\theta}(S_k, A_k)
\bigg|_{\theta_t}
\end{aligned}
\end{equation}
$$
But this is still insufficient for a practical algorithm. Because we have an infinite sum and it incurs extremely high variance during Monte-Carlo estimate. So we try to simplify it even further. 
$$
\begin{equation}
\begin{aligned}
\nabla_{\theta} V^{\pi_\theta}\bigg|_{\theta_t}
=&
\sum_{k=0}^{\infty}\gamma^k~
\mathbb{E}_{\mathcal{M}}^{\pi_\theta}
\underbrace{\nabla_{\theta} \ln \pi{\color{red}{_k^{\color{black}{\theta}}}}(A_k|S_k)
~A{\color{red}{_k^{\color{black}{\theta}}}}(S_k, A_k)}_{\text{Denote this function by}f_{{\color{red}{_k}}}(S_k, A_k;\theta)}
\bigg|_{\theta_t}
\\=&
\frac{1}{1-\gamma}
\cdot 
({1-\gamma})
{\color{blue}{\int_{\mathcal{S}}ds \int_{\mathcal{A}}da}}
\left[\sum_{k=0}^{\infty}\gamma^{\color{red}{k}}~
\mathbb{P}_{\mathcal{M}}^{\pi_\theta}(S_{\color{red}{k}}={\color{blue}{s}}){\pi{\color{red}{_k^{\color{black}{\theta}}}}({\color{blue}{a|s}})}
~{\color{blue}{f}}_{{\color{red}{_k}}}({\color{blue}{s,a}};\theta)
\right]\bigg|_{\theta_t}
\\=&
\frac{1}{1-\gamma}
\cdot 
\underbrace{{\color{blue}{\int_{\mathcal{S}}ds \int_{\mathcal{A}}da~f(s,a;\theta)}}}_{\text{Statistical information}}
\underbrace{\left[({1-\gamma})
\sum_{k=0}^{\infty}\gamma^{\color{red}{k}}~
~\mathbb{P}_{\mathcal{M}}^{\pi_\theta}(S_{\color{red}{k}}={\color{blue}{s}}){\pi_\theta({\color{blue}{a|s}})}
\right]}_{\text{Temporal information}}
\bigg|_{\theta_t}
\\&\text{Stationarity helps decouple the temporal summation and statistical integration.}
\\=&
\frac{1}{1-\gamma}
\cdot
{\int_{\mathcal{S}}ds~d_{\mathcal{M}}^{\pi_\theta}({\color{blue}{s}})
\int_{\mathcal{A}}da~\pi_\theta(a|s)f(s,a;\theta)}
\bigg|_{\theta_t}
\\=&
\frac{1}{1-\gamma}
\cdot
\mathbb{E}_{s\sim d_{\mathcal{M}}^{\pi_\theta}(\cdot), a\sim \pi_\theta(\cdot|s)}
\nabla_{\theta} \ln \pi_{\theta}(a|s)A^{\pi_\theta}(s,a)
\bigg|_{\theta_t}
\end{aligned}
\end{equation}
$$

> 1. We have defined the **state visitation measure** in the derivations above:
>
> $$
> d_{\mathcal{M}}^{\pi_\theta}(s)=
> (1-\gamma)\sum_{k=0}^{\infty}\gamma^{k}~
> ~\mathbb{P}_{\mathcal{M}}^{\pi_\theta}(S_k={\color{blue}{s}})
> $$
>
> The additional $(1-\gamma)$ factor is introduced to ensure that $\int_{\mathcal{S}}d_{\mathcal{M}}^{\pi_\theta}(ds)=1$, so that it is indeed a legitimate probability measure.
>
> 2. We should be aware that **the result above only applies to MDPs with stationary kernels, rewards and policies**, since only under this setting will the advantage function become irrelevant with $k$​, which helps us set it free from the summation $\sum_{k=0}^\infty$. 

Now look at the policy gradient theorem for stationary MDPs.
$$
\begin{equation}
\begin{aligned}
\nabla_{\theta} V^{\pi_\theta}\bigg|_{\theta_t}
=&
\frac{1}{1-\gamma}
\mathbb{E}_{s\sim d_{\mathcal{M}}^{\pi_\theta}(\cdot), a\sim \pi_\theta(\cdot|s)}
\nabla_{\theta} \ln \pi_{\theta}(a|s)~A^{\pi_\theta}(s,a)
\bigg|_{\theta_t}
\end{aligned}
\end{equation}
$$
This expression is particularly concise, and it is very suitable for Monte-Carlo estimate, as it only involves a single term. It is straightforward to construct an unbiased estimator of this gradient:
$$
\widehat{\nabla_{\theta} V^{\pi_\theta}}\bigg|_{\theta_t}
=
\frac{1}{1-\gamma}
A^{\pi_{\theta_t}}(s,a)
~
\nabla_{\theta} \ln \pi_{\theta}(a|s)\bigg|_{\theta_t}
,
\quad \text{where } s\sim d_{\mathcal{M}}^{\pi_\theta}(\cdot), a\sim \pi_{\theta_t}(\cdot|s)
$$
And we can approximate the advantage function and the policy by two neural nets, then define the objective function as 
$$
L(\theta)=\frac{1}{N}\sum_{i=1}^N \ln \pi_{\theta}(a_i|s_i) A_\phi(s_i,a_i), 
\text{ where } s_i\sim d_{\mathcal{M}}^{\pi_\theta}(\cdot), a_i\sim \pi_{\theta_t}(\cdot|s_i)
$$
then run SGA or its Adam equivalent to maximize the rewards.



However, we should notice that the simplicity comes with a price: the distribution of the states $s$ is much more involved then the Markovian samples. The artificial distribution
$$
d_{\mathcal{M}}^{\pi_\theta}(s)
$$
does not necessarily correspond to any existing distribution that carries significant physical meaning. So when  estimating the expected advantage function, we should adopt a difference sampling mechanism. 

well it seems that DRL takes the stationary distribution as $d$. ,but I suspect that there could introduce some bias.  

**See Sham Kakade ’s NPG for detailed mechanism. I have seen it before.**



#### Practical estimate of the advantage function

In practice, we consider constructing  a low-variance low bias estimator of the advantage function using finite samples. We would like to use the empirical advantage function to estimate the policy gradient:
$$
\nabla_{\theta}V^{\pi_\theta}\approx \hat{g}=\frac{1}{N} \sum_{n=1}^N \sum_{t=0}^{\infty} \hat{A}_t^n \nabla_\theta \log \pi_\theta\left(a_t^n \mid s_t^n\right)
$$
where $(s_t,a_t)$ are samples across trajectories…??

1. First observe that the TD error already constitutes an unbiased estimator of the single-step advantage function, conditioned on previous state-action pair.

$$
\mathbb{E}[{\text{TD}_{\text{target}}(s_t,a_t,s_{t+1})|s_t,a_t}]
=
\mathbb{E}[{r_t(s_t,a_t)+\gamma V(s_{t+1})|s_t,a_t}]
\\=r_t(s_t,a_t)+\gamma \int_{\mathcal{S}}P(s_{t+1}|s_t,a_t)V(s_{t+1})
=Q_t(s_t,a_t)
$$

$$
\mathbb{E}[\underbrace{\delta_t(s_t,a_t,s_{t+1})}_{\text{TD error}}|s_t,a_t]=\mathbb{E}[{\text{TD}_{\text{target}}-V(s_t)|s_t,a_t}]=A(s_t,a_t)
$$

2. Generalizing the TD error, we find another way to construct an unbiased estimator is to use the series of discounted TD errors:

$$
\widehat{A}_t^{(k)}:=\sum_{l=0}^{k-1}\gamma^l~\delta_{t+l}(s_{t+l},a_{t+l},s_{t+l+1})
$$

​	We notice that 
$$
\begin{equation}
\begin{aligned}
\widehat{A}_t^{(k)}
=&
\sum_{l=0}^{k-1}\gamma^l~r_{t+l}(s_{t+l},a_{t+l})+
\sum_{l=0}^{k-1}\gamma^{l+1} V(s_{t+l+1})-\sum_{l=0}^{k-1}\gamma^{l} V(s_{t+l})
\\=&
\sum_{l=0}^{k-1}\gamma^l~r_{t+l}(s_{t+l},a_{t+l})+
\gamma \sum_{l=1}^{k}\gamma^{l} V(s_{t+l})-\sum_{l=0}^{k-1}\gamma^{l} V(s_{t+l})
\\=&
\underbrace{\sum_{l=0}^{k-1}\gamma^l~r_{t+l}(s_{t+l},a_{t+l})
-V(s_{t})}_{\text{Asymptotically unbiased estimator of $A(s_t,a_t)$}}
+\underbrace{\gamma^k V(s_{t+k})}_{\text{Bias term that vanishes as $k\to \infty$}}
\end{aligned}
\end{equation}
$$
​        which implies 
$$
\mathbb{E}[\lim_{k\to \infty}\widehat{A}_t^{(k)}|s_t,a_t]
:=Q(s_t,a_t)-V(s_t)+
\lim_{k\to \infty} \gamma^k~V(s_{t+k})
=A(s_t,a_t)+0
$$
​	Another fact is that the TD error is simply a special case of the empirical advantage function
$$
\underbrace{\delta_t(s_t,a_t,s_{t+1})}_{\text{TD error}}=\widehat{A}_t^{(1)}
$$
​	So we discovered two types of unbiased estimators of the advantage function:

| Estimator                            | Bias | Variance            | Information              |
| ------------------------------------ | ---- | ------------------- | ------------------------ |
| $\widehat{A}_t^{\infty}|s_t,a_t$     | 0    | High (infinite sun) | High  (multiple samples) |
| $\widehat{A}_t^{k}|s_t,a_t$          | +    | Medium              | Medium                   |
| $\widehat{A}_t^{1}=\delta_t|s_t,a_t$ | 0    | Low (single term)   | Low   (single sample)    |

We want to get the best of both worlds of $\delta_t$ and $\widehat{A}_t^\infty$ and strike a balance between bias and variance. It seems that we should truncate $k$ to some value, however the appropriate $k$ might be difficult to tune since it is discrete. To solve this issue, we introduce $\lambda-$weighting of each term in the infinite sum, in the hope that it can continuously interpolate between the two unbiased estimators. Let us define…
$$
\widehat{A}_t^{(k)}:=\sum_{l=0}^{k-1}\gamma^l~\delta_{t+l}(s_{t+l},a_{t+l},s_{t+l+1})
\\
\widehat{A}^{(k)}_t({\color{red}{\lambda}}):=\sum_{l=0}^{k-1}\gamma^l {\color{red}{\lambda^l}}~\delta_{t+l}(s_{t+l},a_{t+l},s_{t+l+1}), \text{where }\lambda \in (0,1).
$$
A direct consequence of the definition is that the $\lambda$-weighted empirical advantage sits in the middle of those unbiased estimators. 

* $\widehat{A}^{\infty}_t({\color{red}{\lambda}}=0):=\delta_{t}(s_{t},a_{t},s_{t})+0=\text{TD error}$
* $\widehat{A}^{\infty}_t({\color{red}{\lambda}}=1):=\sum_{l=0}^{\infty}\gamma^l {\color{red}{\lambda^l}}~\delta_{t+l}(s_{t+l},a_{t+l},s_{t+l+1})=\widehat{A}_t^{\infty}$ 

We will call $\widehat{A}_t^{(k)}(\lambda)$ as the Generalized Advantage Estimate (GAE).

>  Shulman’s 2016 ICLR paper “HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION” writes:
>
> GAE(γ, 1) is γ-just regardless of the accuracy of V , but it has high variance due to the sum of terms. GAE(γ, 0) is γ-just for V = V π,γ and otherwise induces bias, but it typically has much lower variance. The generalized advantage estimator for 0 < λ < 1 makes a compromise between bias and variance, controlled by parameter λ.

| Estimator                                 | Bias | Variance                                   | Information             |
| ----------------------------------------- | ---- | ------------------------------------------ | ----------------------- |
| $\widehat{A}_t^{\infty}|s_t,a_t$          | 0    | Very High (due to infinite sum)            | Full (multiple samples) |
| $\widehat{A}_t^{\infty}(\lambda)|s_t,a_t$ | +    | High                                       | Higher                  |
| $\widehat{A}_t^{k}(\lambda)|s_t,a_t$      | +    | Medium                                     | Medium                  |
| $\widehat{A}_t^{1}=\delta_t|s_t,a_t$      | 0    | High (due to fluctuation of a single term) | Low   (single sample)   |

In PPO implementation we adopt $$\widehat{A}_t^{k}(\lambda)|s_t,a_t$$ with a short $k$ as a slightly biased but low variance estimator of the advantage function. We tune the parameter $\lambda$ for bias-variance tradeoff, and we choose appropriate truncate length $k$ to suit recurrent neural networks, which also makes the policy on-policy-ish. 



##### Why should we use multi-step TD errors to estimate the advantage function? 

We sacrifice a little bias in the estimation to reduce the variance incurred by the single-step fluctuations or correlated infinite sum. A reduction in variance also helps to stabilize training. Moreover, using multiple steps also involves more information from more samples, so improves sample efficiency. 



>  1. **Reducing Bias and Variance**
>
> - **Single-step TD errors (TD(0))** **<u>are unbiased but can have high variance because they rely only on one step ahead, subjecting them to random fluctuations in rewards and state transitions.</u>** 
> - **Multi-step TD errors** combine several shorter horizon estimates into a longer horizon estimate. By **using information over multiple steps, you can reduce the variance of the estimates while maintaining reasonable control over the bias**. This leverage over multiple steps helps in **smoothening out the fluctuations and provides a more stable estimate**. 
>
> 2. **Stabilization of Training**
>
> - Employing multiple steps in advantage estimation helps to stabilize the training process. It can provide a smoother estimation of the expected returns which aids in the more stable convergence of the policy updates.
>
> 3. **Trade-off Between Bias and Variance**
>
> - Multi-step returns allow a configurational trade-off between bias and variance. A shorter horizon (few steps) leads to lower bias but higher variance, whereas a longer horizon (many steps) can potentially increase the bias due to compounding approximate values but will lower the variance. By tuning the number of steps, you can find an optimal point that works best for the specific environment and problem.
>
> 3. **Improved Sample Efficiency**
>
> - ***<u>By effectively using multi-step returns, you can make better use of the available samples. Multi-step methods can more quickly propagate information from rewards back to earlier states and actions, which can speed up learning. This is particularly useful in environments where rewards are sparse or delayed</u>***.
>
> 5. **Flexibility in Estimating Long-term Returns**
>
> - Multi-step TD methods allow the flexibility to estimate returns over varying horizons, giving a clearer picture of long-term outcomes from actions. This is particularly useful in environments with complex state dynamics and reward structures.

