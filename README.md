# AIDocks

A micro-service to Build-Your-Own Mixture-of-Experts (MoE) models, LaserRMT optimization & fine-tuning **L**arge **L**anguage, Embeddings & ReRanking **M**odels.

## Pre-Requisites
0. Git
1. [Docker](https://docs.docker.com/get-docker/) & [docker-compose](https://docs.docker.com/compose/install/linux/)
2. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Quick Start

```bash
git clone https://github.com/l4b4r4b4b4/AIDocks
cd AIDocks
docker-compose up -d && \
docker-compose ps && \
docker-compose logs -f
```

Go to the API [documentation](http://localhost:8723/docs) to check and try the features and build your own MoE-model right now!

## Endpoints 🚀
### `/laser` - LaserRMT 
`request body`
```json
{
    "base_model_name" : "TinyLlama/TinyLlama-1.1B-Chat-v1.0","laser_model_name": "TinyLaser",
    "top_k_layers": 15,
    "publish": false,
    "trainer": "LHC88"
    
}
```
LaserRMT optimizes LLMs combining Layer-Selective Rank Reduction (LASER) and the Marchenko-Pastur law from Random Matrix Theory. This method targets model complexity reduction while maintaining or enhancing performance, making it more efficient than the traditional brute-force search method.

1. LASER Framework Adaptation: LaserRMT adapts the LASER technique, which reduces the complexity of neural networks by selectively pruning the weights of a model's layers.
2. Marchenko-Pastur Law Integration: The Marchenko-Pastur law, a concept from Random Matrix Theory used to determine the distribution of eigenvalues in large random matrices, guides the identification of redundant components in LLMs. This allows for effective complexity reduction without loss of key information.
3. Enhanced Model Performance: By systematically identifying and eliminating less important components in the model's layers, LaserRMT can potentially enhance the model's performance and interpretability.
4. Efficient Optimization Process: LaserRMT provides a more efficient and theoretically robust framework for optimizing large-scale language models, setting a new standard for language model refinement.

This approach opens new avenues for optimizing neural networks, underscoring the synergy between advanced mathematical theories and practical AI applications. LaserRMT sets a precedent for future developments in the field of LLM optimization.

### `/byo-moe` - BYO-MoE

`/byo-moe` is an endpoint for combining Mistral or Llama models of the same size into Mixtral Mixture of Experts models. The endpoint will combine the self-attention and layer normalization parameters from a "base" model with the MLP parameters from a set of "expert" models. `/byo-moe` uses its own JSON configuration syntax, which looks like so:
`request body`
```json
{
    "base_model": "cognitivecomputations/dolphin-2.6-mistral-7b-dpo",
    "gate_mode": "hidden",
    "dtype": "bfloat16",
    "experts":[
        {
            "source_model": "teknium/OpenHermes-2.5-Mistral-7B",
            "positive_prompts": [
                "instruction"
                "solutions"
                "chat"
                "questions"
                "comprehension"
            ]
        },   
        {
            "source_model": "openaccess-ai-collective/DPOpenHermes-7B",
            "positive_prompts": [
                "mathematics"
                "optimization"
                "code"
                "step-by-step"
                "science"
            ],
            "negative_prompts": [
                "chat"
                "questions"
            ]
        }
    ]
}
```
**Options**:

`gate_mode`: `hidden`, `cheap_embed`, or `random`

`dtype`: `float32`, `float16`, or `bfloat16`

#### Gate Modes

There are three methods for populating the MoE gates implemented.

##### "hidden"

Uses the hidden state representations of the positive/negative prompts for MoE gate parameters. Best quality and most effective option; the default. Requires evaluating each prompt using the base model so you might not be able to use this on constrained hardware (depending on the model). You can use `--load-in-8bit` or `--load-in-4bit` to reduce VRAM usage.

##### "cheap_embed"

Uses only the raw token embedding of the prompts, using the same gate parameters for every layer. Distinctly less effective than "hidden". Can be run on much, much lower end hardware.

##### "random"

Randomly initializes the MoE gates. Good for if you are going to fine tune the model afterwards, or maybe if you want something a little unhinged? I won't judge.

### `/train/llm` Optimized LLM fine-tuning (DPO & SFT) with `unsloth`
Coming soon ...

### `/train/emb` LoRA-PEFT for Embeddings
Including JinaAI!

Coming soon ...

### `/train/rerank` ReRanker fine-tuning
Coming soon ...

## Explaining Resources
1. [MergeKit Mixtral](https://github.com/cg123/mergekit/tree/mixtral)
2. [Mixture of Experts for Clowns (at a Circus)](https://goddard.blog/posts/clown-moe/#fn-mlp)
3. [Fernando Fernandes Neto, David Golchinfar and Eric Hartford. "Optimizing Large Language Models Using Layer-Selective Rank Reduction and Random Matrix Theory." 2024.](https://github.com/cognitivecomputations/laserRMT)
4. [Introducing ‘LASER’: MIT and Microsoft’s Game-Changing Approach to Optimizing Large Language Models](https://multiplatform.ai/introducing-laser-mit-and-microsofts-game-changing-approach-to-optimizing-large-language-models/)
5. [This Paper from MIT and Microsoft Introduces ‘LASER’: A Novel Machine Learning Approach that can Simultaneously Enhance an LLM’s Task Performance and Reduce its Size with no Additional Training](https://www.marktechpost.com/2024/01/02/this-paper-from-mit-and-microsoft-introduces-laser-a-novel-machine-learning-approach-that-can-simultaneously-enhance-an-llms-task-performance-and-reduce-its-size-with-no-additional-training/)
6. [An Empirical view of Marchenko-Pastur Theorem](https://medium.com/swlh/an-empirical-view-of-marchenko-pastur-theorem-1f564af5603d)