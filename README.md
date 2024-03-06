# Description

This repository contains code for a class of methods aimed at "consolidating" an LLM's context into its weights. The name "SyCoMode" comes from "Systems Consolidation" in humans, the process by which memories residing in the hippocampus are transferred to the neocortex.

Loosely, the SyCoMode training objective is based on that originally delineated in [1], which is as follows:

$$ \Pr (w_n | \theta') \approx \Pr (w_n | w_{n-k}, ..., w_{n-1}; \theta)  $$

The objective delineated above is to learn some set of LLM model weights $\theta'$ that, without a prompt, approximate the behavior of the LLM given its default weights $\theta$ and a prompt $w_{n-1}, ..., w_{n-1}$. Thus, the prompt is effectively "consolidated" into the new weights $\theta'$.

Some possible applications of this technique include:
- **Minimization of Compute Costs:** by compressing a system prompt into an LLM's weights, you would no longer need to compute its attention values and hidden layers.
- **Learning Efficiency:** by mimicking in-context learning, SyCoMode could potentially be more efficient than causal language modeling.
- **Limiting Hallucinations:** once again, by mimicking in-context learning, SyCoMode might perform comparably to RAG [2], which has been shown to reduce hallucinations.


# Progress Thus Far
This work is currently unfinished. I have gotten some preliminary experiments to work, but nothing has been fully effective at consolidating prompts. Thus, I invite you to help me work on this project! Feel free to make a pull request into this repository or use the code in your own project. If you do, please cite this work as follows:

```
@misc{sycomode,
  title={Systems Consolidation in LLMs: From Context to Weights},
  author={Sansom, Jacob and Glasscock, Creighton and Ma, Ziqiao and Chai, Joyce},
  journal={GitHub Repository}
  url={https://github.com/jhsansom/SyCoMode}
  year={2024}
}
```

# Structure of Code
I have devised three simple experiments, located in `experiment1.py`, `experiment2.py`, and `experiment3.py`. Each file has a description at the top. To run an experiment, use the following command:

```
python experiment1.py \
    --num-mem-tokens=0  \
    --objective-function=causal_language_model \
    --lr=2e-5 \
    --num-iter=5 \
    --temp=1 \
    --model-name=huggyllama/llama-7b \
    --no-wandb
```

Each flag shown above contains its default value. Here is a more detailed description:
- `num-mem-tokens`: An integer value specifying how many custom memory tokens to compress the prompt into. If 0, compress the prompt into the weights of the model itself.
- `objective-function`: Reference the various objective functions in `objectives.py`.
- `lr`: The learning rate.
- `num-iter`: The number of gradient descent steps.
- `temp`: The temperature used in the softmax equation for outputting probability values.
- `model-name`: The model name, as stored on HuggingFace.
- `no-wandb`: Use this flag if you want to turn OFF W&B. Omit this flag if you DO want to track via W&B.


# Works Cited
- [1] Askell, Amanda, et al. "A general language assistant as a laboratory for alignment." arXiv preprint arXiv:2112.00861 (2021).
- [2] Lewis, Patrick, et al. "Retrieval-augmented generation for knowledge-intensive nlp tasks." Advances in Neural Information Processing Systems 33 (2020): 9459-9474.
