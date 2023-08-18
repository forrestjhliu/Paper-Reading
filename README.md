# Paper-Reading
Attentively reading papers and deliberate practice. 



![](https://upload-images.jianshu.io/upload_images/703764-605e3cc2ecb664f6.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# DPO: Direct Preferences Optim

## Forest Liu (USTC, Shanghai AI Lab)

[Paper Link](https://arxiv.org/pdf/2305.18290.pdf) 
>Motivation:
RLHF is complex and unstable, so we aggregate the RM training and RL in one step.Therefore, preference data directly leads to final LM.

### Review on RLHF

1. supervised fine-tuning (SFT): With high-quality dataset of dialogue, instruction following, summarization.(like 1000?)  
    
    >SFT: Learning from the textbook, straightforward.
    >RLHF: Learning by trying progress, successful.

2. preference sampling and reward learning
    1. assuming human preference distribution can be written as
        $$
        p^*\left(y_1 \succ y_2 \mid x\right) =
        \frac{\exp \left(r^*\left(x, y_1\right)\right)}{\exp \left(r^*\left(x, y_1\right)\right)+\exp \left(r^*\left(x, y_2\right)\right)}
        $$

    2. the loss function denotes:
        $$\mathcal{L}_R\left(r_{\phi}, \mathcal{D}\right) =
        -\mathbb{E}_{\left(x, y_w, y_l\right) \sim \mathcal{D}}\left[\log \sigma\left(r_\phi\left(x, y_w\right)-r_\phi\left(x, y_l\right)\right)\right],
        \sigma(x)=\frac{1}{1+e^{-x}}$$
        Questions: 
        1. Why RM accuracy is low to 64%?
        2. Why there is a log, to make the difference more obvious? <font color=blue>Easy to differentiate? </font>
3. reinforcement-learning optimization.
    1. Optim goal
        $$\max _{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y \mid x)}\left[r_\phi(x, y)\right]
        -\beta \mathbb{D}_{\mathrm{KL}}\left[\pi_\theta(y \mid x) \| \pi_{\mathrm{ref}}(y \mid x)\right]，
        \mathbb{D}_{\mathrm{KL}} = log\frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}$$
    2. Not differentiable, so reward function denotes
        $$r(x, y) = r_\phi(x, y)-\beta\left(\log \pi_\theta(y \mid x)-\log \pi_{\mathrm{ref}}(y \mid x)\right)
        $$
        Questions: 
        1. Why this one differnentiable?
        2. Fine-tuning with params of certain layers fixed?
        
![](https://m.2008php.com/tuku/978149.html)
### DPO Outline

1. Deriving the Objective

    We will derive Eq. 4 Analogously to Eq. 3. we optimize the following objective:

    $$
    \max _{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi}[r(x, y)]-\beta \mathbb{D}_{\mathrm{KL}}\left[\pi(y \mid x) \| \pi_{\text {ref }}(y \mid x)\right]
    $$

    under any reward function $r(x, y)$, reference model $\pi_{\text {ref }}$ and a general non-parametric policy class. We now have:

    $$
    \begin{aligned}
    \max _{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi} & {[r(x, y)]-\beta \mathbb{D}_{\mathrm{KL}}\left[\pi(y \mid x) \| \pi_{\mathrm{ref}}(y \mid x)\right] } \\
    & =\max _{\pi} \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim \pi(y \mid x)}\left[r(x, y)-\beta \log \frac{\pi(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}\right] \\
    & =-\beta \min _{\pi} \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim \pi(y \mid x)}\left[\log \frac{\pi(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}-\frac{1}{\beta} r(x, y)\right] \\
    & =-\beta \min _{\pi} \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim \pi(y \mid x)}\left[\log \frac{\pi(y \mid x)}{\frac{1}{Z(x)} \pi_{\mathrm{ref}}(y \mid x) \exp \left(\frac{1}{\beta} r(x, y)\right)}-\log Z(x)\right]
    \end{aligned}
    $$

    where we have partition function (import concept in Probability Theroy and Statistical Physics):

    $$
    Z(x)=\sum_{y} \pi_{\mathrm{ref}}(y \mid x) \exp \left(\frac{1}{\beta} r(x, y)\right) .
    $$

    Note that the partition function is a function of only $x$ and the reference policy $\pi_{\text {ref }}$, but does not depend on the policy $\pi$.

    <font color=red>We can now define
    $$
    \pi^{*}(y \mid x)=\frac{1}{Z(x)} \pi_{\text {ref }}(y \mid x) \exp \left(\frac{1}{\beta} r(x, y)\right)
    $$
    </font>

    which is a valid probability distribution as $\pi^{*}(y \mid x) \geq 0$ for all $y$ and $\sum_{y} \pi^{*}(y \mid x)=1$. Since $Z(x)$ is not a function of $y$, we can then re-organize the final objective in $\mathrm{Eq} 12$ as:

    $$
    \begin{aligned}
    \min _{\pi} \mathbb{E}_{x \sim \mathcal{D}}\left[\mathbb{E}_{y \sim \pi(y \mid x)}\left[\log \frac{\pi(y \mid x)}{\pi^{*}(y \mid x)}\right]-\log Z(x)\right]= \\
    \min _{\pi} \mathbb{E}_{x \sim \mathcal{D}}\left[\mathbb{D}_{\mathrm{KL}}\left(\pi(y \mid x) \| \pi^{*}(y \mid x)\right)+Z(x)\right]
    \end{aligned}
    $$

    Now, since $Z(x)$ does not depend on $\pi$, the minimum is achieved by the policy that minimizes the first KL term. <font color=blue, font face = century gothic> Gibbs' inequality </font> tells us that the KL-divergence is minimized at 0 if and only if the two distributions are identical(Widely known). Hence we have the optimal solution:

    $$
    \pi(y \mid x)=\pi^{*}(y \mid x)=\frac{1}{Z(x)} \pi_{\mathrm{ref}}(y \mid x) \exp \left(\frac{1}{\beta} r(x, y)\right)
    $$
    
    for all $x \in \mathcal{D}$. 
    This completes the derivation. **Motivation: Trying to make the object a $\mathbb{D}_{\mathrm{KL}}$ + sth invarient with $y$.** 

    A.2 Deriving the DPO Objective Under the Bradley-Terry Model

    It is straightforward to derive the DPO objective under the Bradley-Terry preference model as we have

    $$
    p^{*}\left(y_{1} \succ y_{2} \mid x\right)=\frac{\exp \left(r^{*}\left(x, y_{1}\right)\right)}{\exp \left(r^{*}\left(x, y_{1}\right)\right)+\exp \left(r^{*}\left(x, y_{2}\right)\right)}
    $$

    In Section 4 we showed that we can express the (unavailable) ground-truth reward through its corresponding optimal policy:

    $$
    r^{*}(x, y)=\beta \log \frac{\pi^{*}(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}+\beta \log Z(x)
    $$

    Substituting Eq. 17 into Eq. 16 we obtain:

    $$
    \begin{aligned}
    p^{*}\left(y_{1} \succ y_{2} \mid x\right) & =\frac{\exp \left(\beta \log \frac{\pi^{*}\left(y_{1} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{1} \mid x\right)}+
    \beta \log Z(x)\right)}{\exp \left(\beta \log \frac{\pi^{*}\left(y_{1} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{1} \mid x\right)}+\beta \log Z(x)\right)+\exp \left(\beta \log \frac{\pi^{*}\left(y_{2} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{2} \mid x\right)}+\beta \log Z(x)\right)} \\
    & =\frac{1}{1+\exp \left(\beta \log \frac{\pi^{*}\left(y_{2} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{2} \mid x\right)}-\beta \log \frac{\pi^{*}\left(y_{1} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{1} \mid x\right)}\right)} \\
    & =\sigma\left(\beta \log \frac{\pi^{*}\left(y_{1} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{1} \mid x\right)}-\beta \log \frac{\pi^{*}\left(y_{2} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{2} \mid x\right)}\right) .
    \end{aligned}
    $$

    The last line is the per-instance loss, so *the loss function to be minimized* becomes:
    $$
    \mathcal{L}_{\mathrm{DPO}}\left(\pi_\theta ; \pi_{\mathrm{ref}}\right) = 
    -\mathbb{E}_{\left(x, y_w, y_l\right) \sim \mathcal{D}}
    \left[log \sigma  \left(\beta 
    log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}   
    -\beta log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]  
    $$


2.  Update

    The gradient of the Loss function is
    $$
    \begin{aligned}
    & \nabla_\theta \mathcal{L}_{\mathrm{DPO}}\left(\pi_\theta ; \pi_{\mathrm{ref}}\right)= \\
    & -\beta \mathbb{E}_{\left(x, y_w, y_l\right) \sim \mathcal{D}}[\underbrace{\sigma\left(\hat{r}_\theta\left(x, y_l\right)-\hat{r}_\theta\left(x, y_w\right)\right)}_{\text {higher weight when reward estimate is wrong }}[\underbrace{\nabla_\theta \log \pi\left(y_w \mid x\right)}_{\text {increase likelihood of } y_w}-\underbrace{\nabla_\theta \log \pi(y \mid x)}_{\text {decrease likelihood of } y_l}]],
    \end{aligned}
    $$
    where $\hat{r}_\theta(x, y)=\beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}$ is the reward implicitly defined by the language model $\pi_\theta$ and reference model $\pi_{\mathrm{ref}}$ 
3. ***DPO Outline***
   1. Sample completions $y_1, y_2 ∼ \pi_{ref}(· | x)$
    for every prompt $x$, label with human preferences to construct the offline dataset of preferences $D = {{x^{(i)}, y_w^{(i)}, y_l^{(i)}}} _{i=1}$ 
   2.  optimize the language model πθ to minimize LDPO for the given
    πref and D and desired β. In practice, one would like to reuse preference datasets publicly available,
    rather than generating samples and gathering human preferences. Since the preference datasets
    are sampled using πSFT, we initialize πref = πSFT whenever available. However, when πSFT is
    not available, we initialize πref by maximizing likelihood of preferred completions (x, yw), that
    is, πref = arg maxπ Ex,yw∼D [log π(yw | x)]. This procedure helps mitigate the distribution shift
    between the true reference distribution which is unavailable, and πref used by DPO.

### Theoretical Analysis of DPO

1. Your LM is secretly a *Reward Model*
   **Def 1** Two reward functions $r(x, y)$ and $r^{\prime}(x, y)$ are equivalent iff $r(x, y)-r^{\prime}(x, y)=f(x)$ for some function $f$.

   **Lemma 1.** Under the Plackett-Luce, and in particular the Bradley-Terry, preference framework, two reward functions from the same class induce the same preference distribution.

   **Lemma 2.** Two reward functions from the same equivalence class induce the same optimal policy under the constrained $R L$ problem.


### Code ([Hugging Face blog](https://huggingface.co/blog/dpo-trl))
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
Q:More efficient to train using $y_1,y_2,...$ with PL model?

~~The Plackett-Luce model is a generalization of the Bradley-Terry model over rankings (rather than just pair-wise comparisons). Similar to to the Bradley-Terry model, it stipulates that when presented with a set of possible choices, people prefer a choice with probability proportional to the value of some latent reward function for that choice. In our context, when presented with a prompt $x$ and a set of $K$ answers $y_{1}, \ldots, y_{K}$ a user would output a permutation $\tau:[K] \rightarrow[K]$, giving their ranking of the answers. The Plackett-Luce model stipulates that

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
