# Paper-Reading
Attentively reading papers and deliberate practice. 


![](https://upload-images.jianshu.io/upload_images/703764-605e3cc2ecb664f6.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# Direct Preferences Optim, Motivations and Further Thinking 

## Forest Liu (USTC, Shanghai AI Lab)

[Paper Link](https://arxiv.org/pdf/2305.18290.pdf) 
>Motivation:
 RLHF is complex and unstable, so we aggregate the RM training and RL in one step. Therefore, preference data directly leads to the final LM.

>Key Insight:
 Leverage an analytical mapping from reward functions to optimal policies, which enables us to transform a loss function over reward functions into a loss function over policies. 

### Review on RLHF

1. Supervised fine-tuning (SFT): With a high-quality dataset of dialogue, instruction following, and summarization. (like 1000?)  
    
    >SFT: Learning from the textbook, straightforward.
    >RLHF: Learning by trying to progress, successful.

2. Preference sampling and Reward learning
    1. Assuming human preference distribution can be written as
        $$\begin{equation}
        p^*\left(y_1 \succ y_2 \mid x\right) =
        \frac{\exp \left(r^*\left(x, y_1\right)\right)}{\exp \left(r^*\left(x, y_1\right)\right)+\exp \left(r^*\left(x, y_2\right)\right)}
        \end{equation}$$

    2. The loss function to train RM denotes:
        $$\begin{equation}
        \mathcal{L}_R\left(r_{\phi}, \mathcal{D}\right) =
        -\mathbb{E}_{\left(x, y_w, y_l\right) \sim \mathcal{D}}\left[\log \sigma\left(r_\phi\left(x, y_w\right)-r_\phi\left(x, y_l\right)\right)\right],
        \sigma(x)=\frac{1}{1+e^{-x}}
        \end{equation}$$
        Questions: 
        1. Why RM accuracy is low to 64%?
        2. Why there is a log, to make the difference more obvious? <font color=blue>Easy to differentiate? </font>
3. Reinforcement-learning optimization.
    1. Optim goal
        $$\begin{equation}
        \max _{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y \mid x)}   \left[r_\phi(x, y)\right]
        -\beta \mathbb{D}_{\mathrm{KL}}\left[\pi_\theta(y \mid x) 
        \| \pi_{\mathrm{ref}}(y \mid x)\right]，
        \mathbb{D}_{\mathrm{KL}} = log\frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}
        \end{equation}$$
        If there is no $\mathbb{D}_{\mathrm{KL}}$, $\pi_\theta(\mathop{\arg \max}_{y} r_\phi(x,y)|x) = 1,$
    2. Not differentiable, so the reward function denotes
        $$\begin{equation}
        r(x, y) = r_\phi(x, y)-\beta\left(\log \pi_\theta(y \mid x)-\log \pi_{\mathrm{ref}}(y \mid x)\right)
        \end{equation}$$
        Questions: 
        1. Why is this one differentiable?
        2. Fine-tuning with params of certain layers fixed?
        
![](https://m.2008php.com/tuku/978149.html)
### DPO Outline

<font color=red>Motivation: Leverage an analytical mapping from reward functions to optimal policies, which enables us to transform a loss function over reward functions into a loss function over policies.</font>

1. Deriving the Objective

    We will derive Eq. 4 Analogously to Eq. 3. We optimize the following objective:

    $$\begin{equation}
    \max _{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi}[r(x, y)]-\beta \mathbb{D}_{\mathrm{KL}}\left[\pi(y \mid x) \| \pi_{\text {ref }}(y \mid x)\right]
    \end{equation}$$

    under any reward function $r(x, y)$, reference model $\pi_{\text {ref }}$ and a general non-parametric policy class. We now have:

    $$
    \begin{aligned}
    \max _{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi} & {[r(x, y)]-\beta \mathbb{D}_{\mathrm{KL}}\left[\pi(y \mid x) \| \pi_{\mathrm{ref}}(y \mid x)\right] } \\
    & =\max _{\pi} \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim \pi(y \mid x)}\left[r(x, y)-\beta \log \frac{\pi(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}\right] \\
    & =-\beta \min _{\pi} \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim \pi(y \mid x)}\left[\log \frac{\pi(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}-\frac{1}{\beta} r(x, y)\right] \\
    & =-\beta \min _{\pi} \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim \pi(y \mid x)}\left[\log \frac{\pi(y \mid x)}{\frac{1}{Z(x)} \pi_{\mathrm{ref}}(y \mid x) \exp \left(\frac{1}{\beta} r(x, y)\right)}-\log Z(x)\right]
    \end{aligned}
    $$

    where we have partition function (import concept in Probability Theory and Statistical Physics):

    $$\begin{equation}
    Z(x)=\sum_{y} \pi_{\mathrm{ref}}(y \mid x) \exp \left(\frac{1}{\beta} r(x, y)\right) .
    \end{equation}$$

    Note that the partition function is a function of only $x$ and the reference policy $\pi_{\text {ref }}$, but does not depend on the policy $\pi$.

    <font color=red>We can now define
    $$\begin{equation}
    \pi^{*}(y \mid x)=\frac{1}{Z(x)} \pi_{\text {ref }}(y \mid x) \exp \left(\frac{1}{\beta} r(x, y)\right)
    \end{equation}$$
    </font>

    which is a valid probability distribution as $\pi^{*}(y \mid x) \geq 0$ for all $y$ and $\sum_{y} \pi^{*}(y \mid x)=1$. Since $Z(x)$ is not a function of $y$, we can then re-organize the final objective in $\mathrm{Eq} 12$ as:

    $$\begin{equation}
    \min _{\pi} \mathbb{E}_{x \sim \mathcal{D}}\left[\mathbb{E}_{y \sim \pi(y \mid x)}\left[\log \frac{\pi(y \mid x)}{\pi^{*}(y \mid x)}\right]-\log Z(x)\right]= \\
    \min _{\pi} \mathbb{E}_{x \sim \mathcal{D}}\left[\mathbb{D}_{\mathrm{KL}}\left(\pi(y \mid x) \| \pi^{*}(y \mid x)\right)+Z(x)\right]
    \end{equation}$$

    Now, since $Z(x)$ does not depend on $\pi$, the minimum is achieved by the policy that minimizes the first KL term. <font color=blue, font face = century gothic> Gibbs' inequality </font> tells us that the KL-divergence is minimized at 0 if and only if the two distributions are identical(Widely known). Hence we have the optimal solution:

    $$\begin{equation}
    \pi(y \mid x)=\pi^{*}(y \mid x)=\frac{1}{Z(x)} \pi_{\mathrm{ref}}(y \mid x) \exp \left(\frac{1}{\beta} r(x, y)\right)
    \end{equation}$$
    
    for all $x \in \mathcal{D}$. 
    This completes the derivation. **Motivation: Trying to make the object a $\mathbb{D}_{\mathrm{KL}}$ + sth invariant with $y$.** 

    A.2 Deriving the DPO Objective Under the Bradley-Terry Model

    It is straightforward to derive the DPO objective under the Bradley-Terry preference model as we have

    $$\begin{equation}
    p^{*}\left(y_{1} \succ y_{2} \mid x\right)=\frac{\exp \left(r^{*}\left(x, y_{1}\right)\right)}{\exp \left(r^{*}\left(x, y_{1}\right)\right)+\exp \left(r^{*}\left(x, y_{2}\right)\right)}
    \end{equation}$$

    In Section 4 we showed that we can express the (unavailable) ground-truth reward through its corresponding optimal policy:

    $$\begin{equation}
    r^{*}(x, y)=\beta \log \frac{\pi^{*}(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}+\beta \log Z(x)
    \end{equation}$$

    Substituting Eq. 17 into Eq. 16 we obtain:

    $$\begin{aligned}
    p^{*}\left(y_{1} \succ y_{2} \mid x\right) & =\frac{\exp \left(\beta \log \frac{\pi^{*}\left(y_{1} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{1} \mid x\right)}+
    \beta \log Z(x)\right)}{\exp \left(\beta \log \frac{\pi^{*}\left(y_{1} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{1} \mid x\right)}+\beta \log Z(x)\right)+\exp \left(\beta \log \frac{\pi^{*}\left(y_{2} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{2} \mid x\right)}+\beta \log Z(x)\right)} \\
    & =\frac{1}{1+\exp \left(\beta \log \frac{\pi^{*}\left(y_{2} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{2} \mid x\right)}-\beta \log \frac{\pi^{*}\left(y_{1} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{1} \mid x\right)}\right)} \\
    & =\sigma\left(\beta \log \frac{\pi^{*}\left(y_{1} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{1} \mid x\right)}-\beta \log \frac{\pi^{*}\left(y_{2} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{2} \mid x\right)}\right) .
    \end{aligned}$$

    The last line is the per-instance loss, so *the loss function to be minimized* becomes:
    $$\begin{equation}
    \mathcal{L}_{\mathrm{DPO}}\left(\pi_\theta ; \pi_{\mathrm{ref}}\right) = 
    -\mathbb{E}_{\left(x, y_w, y_l\right) \sim \mathcal{D}}
    \left[log  \sigma  \left(\beta 
    log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}   
    -\beta log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]  
    \end{equation}$$

2.  Update
    In this section we derive the gradient of the DPO objective:
    $$\begin{equation}
    \nabla_{\theta} \mathcal{L}_{\mathrm{DPO}}\left(\pi_{\theta} ; \pi_{\text {ref }}\right)=-\nabla_{\theta} \mathbb{E}_{\left(x, y_{w}, y_{l}\right) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_{\theta}\left(y_{l} \mid x\right)}{\pi_{\text {ref }}\left(y_{l} \mid x\right)}-\beta \log \frac{\pi_{\theta}\left(y_{w} \mid x\right)}{\pi_{\text {ref }}\left(y_{w} \mid x\right)}\right)\right]
    \end{equation}$$

    We can rewrite the RHS of Equation 21 as
    $$\begin{equation}
    \nabla_{\theta} \mathcal{L}_{\mathrm{DPO}}\left(\pi_{\theta} ; \pi_{\mathrm{ref}}\right)=-\mathbb{E}_{\left(x, y_{w}, y_{l}\right) \sim \mathcal{D}}\left[\frac{\sigma^{\prime}(u)}{\sigma(u)} \nabla_{\theta}(u)\right],
    \end{equation}$$

    where $u=\beta \log \frac{\pi_{\theta}\left(y_{l} \mid x\right)}{\pi_{\text {ref }}\left(y_{l} \mid x\right)}-\beta \log \frac{\pi_{\theta}\left(y_{w} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{w} \mid x\right)}$.

    Using the properties of sigmoid function $\sigma^{\prime}(x)=\sigma(x)(1-\sigma(x))$ and $\sigma(-x)=1-\sigma(x)$, we obtain the final 
    $$\begin{aligned}
    & \nabla_\theta \mathcal{L}_{\mathrm{DPO}}\left(\pi_\theta ; \pi_{\mathrm{ref}}\right)= \\
    & -\beta \mathbb{E}_{\left(x, y_w, y_l\right) \sim \mathcal{D}}[\underbrace{\sigma\left(\hat{r}_\theta\left(x, y_l\right)-\hat{r}_\theta\left(x, y_w\right)\right)}_{\text {higher weight when reward estimate is wrong }}[\underbrace{\nabla_\theta \log \pi\left(y_w \mid x\right)}_{\text {increase likelihood of } y_w}-\underbrace{\nabla_\theta \log \pi(y \mid x)}_{\text {decrease likelihood of } y_l}]],
    \end{aligned}$$
    where $\hat{r}_\theta(x, y)=\beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}$ is the reward implicitly defined by the language model $\pi_\theta$ and reference model $\pi_{\mathrm{ref}}$ 

3. ***DPO Outline***
   1. Sample completions $y_1, y_2 \sim \pi_{\mathrm{ref}}(\cdot \mid x)$ for every prompt $x$, label with human preferences to construct the offline dataset of preferences 
   $\mathcal{D} = \left\{x^{(i)}, y_w^{(i)}, y_l^{(i)} \right\}_{i=1}^N$ and 
   2. Optimize the language model $\pi_\theta$ to minimize $\mathcal{L}_{\mathrm{DPO}}$ for the given $\pi_{\text {ref }}$ and $\mathcal{D}$ and desired $\beta$. In practice, one would like to reuse preference datasets publicly available, rather than generating samples and gathering human preferences. Since the preference datasets are sampled using $\pi^{\mathrm{SFT}}$, we initialize $\pi_{\mathrm{ref}}=\pi^{\mathrm{SFT}}$ whenever available. 
        However, when $\pi^{\mathrm{SFT}}$ is not available, we initialize $\pi_{\text {ref }}$ by maximizing likelihood of preferred completions $\left(x, y_w\right)$, that is, $\pi_{\text {ref }}=\arg \max _\pi \mathbb{E}_{x, y_w \sim \mathcal{D}}\left[\log \pi\left(y_w \mid x\right)\right]$. This procedure helps mitigate the distribution shift between the true reference distribution which is unavailable, and $\pi_{\mathrm{ref}}$ used by DPO.

### <font color = red >*Key Insight Revisited* </font>
*Leverage an analytical mapping from reward functions to optimal policies, which enables us to transform a loss function over reward functions into a loss function over policies.* 
1. Analytical mapping :
    $$\begin{equation}
    r^{*}(x, y)=\beta \log \frac{\pi^{*}(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}+\beta \log Z(x)
    \end{equation}$$
    Every reward function equivalence class can be mapped to one certain policy. And the optimal policy of loss function over reward function is the policy.
2. The transformation of loss function:
    Using the mapping between reward and policy, 
    from        
    $$\begin{equation} 
    \max _{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y \mid x)}   \left[r_\phi(x, y)\right]
        -\beta \mathbb{D}_{\mathrm{KL}}\left[\pi_\theta(y \mid x) 
        \| \pi_{\mathrm{ref}}(y \mid x)\right]
    \end{equation}$$
    to  
    $$\begin{equation}
    -\mathbb{E}_{\left(x, y_w, y_l\right) \sim \mathcal{D}}
    \left[log  \sigma  \left(\beta 
    log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}   
    -\beta log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]  
    = \mathcal{L}_{\mathrm{DPO}}\left(\pi_\theta ; \pi_{\mathrm{ref}}\right) 
    \end{equation}$$


### Back-up Theoretical Analysis

1. Your LM is secretly a *Reward Model*

   **Def 1** Two reward functions $r(x, y)$ and $r^{\prime}(x, y)$ are equivalent iff $r(x, y)-r^{\prime}(x, y)=f(x)$ for some function $f$, i.e.
   $r(x,y_1) - r(x,y_2) = r^{\prime}(x,y_1) - r^{\prime}(x,y_2), \forall y_1,y_2.$  

   **Lem 1.** Under the PL, and in particular the BT, preference framework, two reward functions from the same class induce the same preference distribution.
    *Proof.* We consider the general Plackett-Luce (with the Bradley-Terry model a special case for $K=2$ ) and denote the probability distribution over rankings induced by a particular reward function $r(x, y)$ as $p_{r}$.

    $$\begin{aligned}
    p_{r^{\prime}}\left(\tau \mid y_{1}, \ldots, y_{K}, x\right) & =\prod_{k=1}^{K} \frac{\exp \left(r^{\prime}\left(x, y_{\tau(k)}\right)\right)}{\sum_{j=k}^{K} \exp \left(r^{\prime}\left(x, y_{\tau(j)}\right)\right)} \\
    & =\prod_{k=1}^{K} \frac{\exp \left(r\left(x, y_{\tau(k)}\right)+f(x)\right)}{\sum_{j=k}^{K} \exp \left(r\left(x, y_{\tau(j)}\right)+f(x)\right)} \\
    & =\prod_{k=1}^{K} \frac{\exp (f(x)) \exp \left(r\left(x, y_{\tau(k)}\right)\right)}{\exp (f(x)) \sum_{j=k}^{K} \exp \left(r\left(x, y_{\tau(j)}\right)\right)} \\
    & =\prod_{k=1}^{K} \frac{\exp \left(r\left(x, y_{\tau(k)}\right)\right)}{\sum_{j=k}^{K} \exp \left(r\left(x, y_{\tau(j)}\right)\right)} \\
    & =p_{r}\left(\tau \mid y_{1}, \ldots, y_{K}, x\right),
    \end{aligned}$$
   
    **Lem 2.** Two reward functions from the same equivalence class induce the same optimal policy under the constrained $R L$ problem.

    *Proof.* Let us consider two reward functions from the same class, such that $r^{\prime}(x, y)=r(x, y)+f(x)$ and, let us denote as $\pi_{r}$ and $\pi_{r^{\prime}}$ the corresponding optimal policies. By Eq. 4 for all $x, y$ we have
    $$\begin{aligned}
    \pi_{r^{\prime}}(y \mid x) & =\frac{1}{\sum_{y} \pi_{\mathrm{ref}}(y \mid x) \exp \left(\frac{1}{\beta} r^{\prime}(x, y)\right)} \pi_{\mathrm{ref}}(y \mid x) \exp \left(\frac{1}{\beta} r^{\prime}(x, y)\right) \\
    & =\frac{1}{\sum_{y} \pi_{\mathrm{ref}}(y \mid x) \exp \left(\frac{1}{\beta}(r(x, y)+f(x))\right)} \pi_{\mathrm{ref}}(y \mid x) \exp \left(\frac{1}{\beta}(r(x, y)+f(x))\right) \\
    & =\frac{1}{\exp \left(\frac{1}{\beta} f(x)\right) \sum_{y} \pi_{\mathrm{ref}}(y \mid x) \exp \left(\frac{1}{\beta} r(x, y)\right)} \pi_{\mathrm{ref}}(y \mid x) \exp \left(\frac{1}{\beta} r(x, y)\right) \exp \left(\frac{1}{\beta} f(x)\right) \\
    & =\frac{1}{\sum_{y} \pi_{\mathrm{ref}}(y \mid x) \exp \left(\frac{1}{\beta} r(x, y)\right)} \pi_{\mathrm{ref}}(y \mid x) \exp \left(\frac{1}{\beta} r(x, y)\right) \\
    & =\pi_{r}(y \mid x).
    \end{aligned}$$

    **Thm 1.** Under mild assumptions, all reward classes consistent with the PL(and BT in particular) models can be represented with the reparameterization
    $r(x, y) = \beta log \frac{π(y|x)}{π_{ref}(y|x)}$ 
    for some model $π(y|x)$ and a given reference model $π_{ref}(y | x)$.

    *Proof.* Consider any reward function $r(x, y)$, which induces an optimal model $\pi_{r}(y \mid x)$ under the KL-constrained RL problem, with analytical solution given by 4 . Following Eq. 5. when we log-linearize both sides we obtain:

    $$\begin{equation}
    r(x, y)=\beta \log \frac{\pi_{r}(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}+\beta \log Z(x)
    \end{equation}$$

    where $Z(x)=\sum_{y} \pi_{\text {ref }}(y \mid x) \exp \left(\frac{1}{\beta} r(x, y)\right)$ (notice that $Z(x)$ also depends on the reward function $r$). Using the operator $r^{\prime}(x, y)=f\left(r, \pi_{\text {ref }}, \beta\right)(x, y)=r(x, y)-\beta \log Z(x)$, we see that this new reward function is within the equivalence class of $r$ and, we have:

    $$\begin{equation}
    r^{\prime}(x, y)=\beta \log \frac{\pi_{r}(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}
    \end{equation}$$

    which completes the proof.

    We can further expand on these results. We can see that if $r$ and $r^{\prime}$ are two reward functions in the same class, then

    $$\begin{equation}
    f\left(r, \pi_{\text {ref }}, \beta\right)(x, y)=\beta \log \frac{\pi_{r}(y \mid x)}{\pi_{\text {ref }}(y \mid x)}=\beta \log \frac{\pi_{r}^{\prime}(y \mid x)}{\pi_{\text {ref }}(y \mid x)}=f\left(r^{\prime}, \pi_{\text {ref }}, \beta\right)(x, y)
    \end{equation}$$

    where the second equality follows from Lemma 2 We have proven that the operator $f$ maps all reward functions from a particular equivalence class to the same reward function. 

    Next, we show that for every equivalence class of reward functions, the reparameterization outlined in Theorem 1 is *unique*.

    **Proposition 1.** Assume we have a reference model, such that $\pi_{\text {ref }}(y \mid x)>0$ for all pairs of prompts $x$ and answers $y$ and a parameter $\beta>0$. Then every equivalence class of reward functions, as defined in Section 5 has a unique reward function $r(x, y)$, which can be reparameterized as $r(x, y)=\beta \log \frac{\pi(y \mid x)}{\pi_{\text {ref }}(y \mid x)}$ for some model $\pi(y \mid x)$.

    *Proof.* We will use *contradiction*, a classic and straight forward way to prove uniqueness. Assume we have two reward functions from the same class, such that $r^{\prime}(x, y)=r(x, y)+f(x)$. Moreover, assume that $r^{\prime}(x, y)=\beta \log \frac{\pi^{\prime}(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}$ for some model $\pi^{\prime}(y \mid x)$ and $r(x, y)=\beta \log \frac{\pi(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}$ for some model $\pi(y \mid x)$, such that $\pi \neq \pi^{\prime}$. We then have

    $r^{\prime}(x, y)=r(x, y)+f(x)=\beta \log \frac{\pi(y \mid x)}{\pi_{\text {ref }}(y \mid x)}+f(x)=\beta \log \frac{\pi(y \mid x) \exp \left(\frac{1}{\beta} f(x)\right)}{\pi_{\text {ref }}(y \mid x)}=\beta \log \frac{\pi^{\prime}(y \mid x)}{\pi_{\text {ref }}(y \mid x)}$

    for all prompts $x$ and completions $y$. Then we must have $\pi(y \mid x) \exp \left(\frac{1}{\beta} f(x)\right)=\pi^{\prime}(y \mid x)$. Since these are distributions, summing over $y$ on both sides, we obtain that $\exp \left(\frac{1}{\beta} f(x)\right)=1$ and since $\beta>0$, we must have $f(x)=0$ for all $x$. Therefore $r(x, y)=r^{\prime}(x, y)$.


2. Instability of Actor-Critic Alogrithms



















### Experiment















### Code ([Hugging Face blog](https://huggingface.co/blog/dpo-trl))
DPO is relatively straightforward to implement; PyTorch code for the DPO loss is provided below:    
    ```
    import torch.nn.functional as F
    def dpo_loss(pi_logps, ref_logps, yw_idxs, yl_idxs, beta):
    """
    pi_logps: policy logprobs, shape (B,)
    ref_logps: reference model logprobs, shape (B,)
    yw_idxs: preferred completion indices in [0, B-1], shape (T,)
    yl_idxs: dispreferred completion indices in [0, B-1], shape (T,)
    beta: temperature controlling strength of KL penalty
    Each pair of (yw_idxs[i], yl_idxs[i]) represents the
    indices of a single preference pair.

    pi_yw_logps, pi_yl_logps = pi_logps[yw_idxs], pi_logps[yl_idxs]
    ref_yw_logps, ref_yl_logps = ref_logps[yw_idxs], ref_logps[yl_idxs]
    pi_logratios = pi_yw_logps - pi_yl_logps
    ref_logratios = ref_yw_logps - ref_yl_logps
    losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
    rewards = beta * (pi_logps - ref_logps).detach()
    return losses, rewards
    ```    
    
    ```
    def return_prompt_and_responses(samples) -> Dict[str, str, str]:
    return {
        "prompt": [
            "Question: " + question + "\n\nAnswer: "
            for question in samples["question"]
        ],
        "chosen": samples["response_j"],   # rated better than k
        "rejected": samples["response_k"], # rated worse than j
    }

    dataset = load_dataset(
    "lvwerra/stack-exchange-paired",
    split="train",
    data_dir="data/rl"
    )
    original_columns = dataset.column_names

    dataset.map(
    return_prompt_and_responses,
    batched=True,
    remove_columns=original_columns
    )
    ```




~~A.3 Deriving the DPO Objective Under the Plackett-Luce Model~~ 
Q: More efficient to train using $y_1,y_2,...$ with the PL model?

~~The Plackett-Luce model is a generalization of the Bradley-Terry model over rankings (rather than just pair-wise comparisons). Similar to the Bradley-Terry model, it stipulates that when presented with a set of possible choices, people prefer a choice with probability proportional to the value of some latent reward function for that choice. In our context, when presented with a prompt $x$ and a set of $K$ answers $y_{1}, \ldots, y_{K}$ a user would output a permutation $\tau:[K] \rightarrow[K]$, giving their ranking of the answers. The Plackett-Luce model stipulates that

$$
p^{*}\left(\tau \mid y_{1}, \ldots, y_{K}, x\right)=\prod_{k=1}^{K} \frac{\exp \left(r^{*}\left(x, y_{\tau(k)}\right)\right)}{\sum_{j=k}^{K} \exp \left(r^{*}\left(x, y_{\tau(j)}\right)\right)}
$$

Notice that when $K=2$, Equation 18 reduces to the Bradley-Terry model. However, for the general Plackett-Luce model, we can still utilize the results of Eq. 5. and substitute the reward function parameterized by its optimal policy. Similarly to Appendix A.2, the normalization constant $Z(x)$ cancels out and we're left with:

$$
p^{*}\left(\tau \mid y_{1}, \ldots, y_{K}, x\right)=\prod_{k=1}^{K} \frac{\exp \left(\beta \log \frac{\pi^{*}\left(y_{\tau(k)} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{\tau(k)} \mid x\right)}\right)}{\sum_{j=k}^{K} \exp \left(\beta \log \frac{\pi^{*}\left(y_{\tau(j)} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{\tau(j)} \mid x\right)}\right)}
$$

Similarly to the approach of Section 4. if we have access to a dataset $\mathcal{D}=$ $\left\{\tau^{(i)}, y_{1}^{(i)}, \ldots, y_{K}^{(i)}, x^{(i)}\right\}_{i=1}^{N}$ of prompts and user-specified rankings, we can use a parameterized model and optimize this objective with maximum-likelihood.:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{DPO}}\left(\pi_{\theta}, \pi_{\mathrm{ref}}\right)=-\mathbb{E}_{\tau, y_{1}, \ldots, y_{K}, x \sim \mathcal{D}}\left[\log \prod_{k=1}^{K} \frac{\exp \left(\beta \log \frac{\pi_{\theta}\left(y_{\tau(k)} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{\tau(k)} \mid x\right)}\right)}{\sum_{j=k}^{K} \exp \left(\beta \log \frac{\pi_{\theta}\left(y_{\tau(j)} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{\tau(j)} \mid x\right)}\right)}\right]
\end{aligned}
$$
