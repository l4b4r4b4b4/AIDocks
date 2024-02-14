# AIDocks

The AI Trainer's Dry Dock.

## Features
- 🚀 Fine-Tune **E**mbeddings, **R**e**R**ankerings & **L**arge **L**anguage **M**odels (LLMs), 
- 🚀 Dataset templates,
- 🚀 Build-Your-Own Mixture-of-Experts (MoE),
- 🚀 Optimize LLMs with LASER-Random Matrix Theory, 
- 🚀 Quantize models for optimal model size &
- 🚀 Publish models to 🤗 HuggingFace Hub.

## Roadmap
(unsorted)

- Auto Hardware Detection -> Model recommendation for fine-tuning and inference
- Fine-tune other models with [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- Combined LLM & retrieval model fine-tuning with human feedback
- The Truth Tables: Distributed (private & shared) Knowledge/Document Management in Chroma over sup- and sub-domain graph in Neo4j.
- Model Conditioning: Chat-based LLM alignment for domain-(field) expertise with auto & human scoring on retrieval relevance, AI reasoning & conclusion.
    - Memory & History
    - Domain specific knowledge retrieval & expert prompting
    - Multiple Conversation
    - Multiple human & AI participants
    - General & Agent Specific Knowledge attachment by domain tags
    - Auto & Human eval for retrieval, reasoning & conclusion results
- AI Task Library

**Disclaimer**
In very early development stage. So feedback and contributions are highly appreciated!

## Pre-Requisites
0. CUDA-GPU
1. [Docker](https://docs.docker.com/get-docker/) & [docker-compose](https://docs.docker.com/compose/install/linux/)
2. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Quick Start

```bash
git clone --recurse-submodules https://github.com/l4b4r4b4b4/AIDocks
cd AIDocks
docker-compose up -d && \
docker-compose ps && \
docker-compose logs -f
```

Go to the [interactive API documentation](http://localhost:8723/docs) to explore all available endpoints & features!
## Services
## Docks WebApp
### Docks API 
### Vision
Llava 1.6 service incl. [Gradio Frontend](http://localhost:8045), [Controller](http://localhost:8046) & [Model Worker](http://localhost:8047)
## llm-inference

## Endpoints 🚀
The following endpoints are exposed:
1. `/train`
2. `/compose`
3. `/optimize`
4. `/quantize`
5. `/publish`

### `/train` Training & Fine-Tuning
The training routes expose different endpoints to fine-tune embeddings or reranking models used for retrieval and LLMs.

#### `/train/llm` LLM fine-tuning (DPO & SFT)
[Try API endpoint](http://localhost:8723/docs#/default/traing_llm__post)
Finetune Mistral, Llama 2-5x faster with 50% less memory with [unsloth](https://github.com/unslothai/unsloth)

**Example datasets** when using ChatML for
1. [SFT](./api/examples/llm/chatml/sft.jsonl)
2. [DPO](./api/examples/llm/chatml/dpo.jsonl)

**Supported Models**
- Llama,
- Yi,
- Mistral,
- CodeLlama,
- Qwen (llamafied),
- Deepseek and their derived models (Open Hermes etc).

**Features**
1. All kernels written in OpenAI's Triton language. Manual backprop engine
2. 0% loss in accuracy - no approximation methods - all exact
3. No change of hardware. Supports NVIDIA GPUs since 2018+. Minimum CUDA Capability 7.0 (V100, T4, Titan V, RTX 20, 30, 40x, A100, H100, L40 etc) Check your GPU! GTX 1070, 1080 works, but is slow
4. Works on Linux and Windows via WSL
5. Download 4 bit models 4x faster from 🤗 Huggingface! Eg: unsloth/mistral-7b-bnb-4bit
6. Supports 4bit and 16bit QLoRA / LoRA finetuning via bitsandbytes

#### `/train/emb` Embeddings
LoRA-PEFT for Embeddings using [peft](https://github.com/huggingface/peft) and [accelerate](https://github.com/huggingface/accelerate) library.

**Supported Models**
- Theoretically any [HuggingFace embeddings model](https://huggingface.co/spaces/mteb/leaderboard).
- Some Models like [jina-embeddings-v2-base-en](https://huggingface.co/jinaai/jina-embeddings-v2-base-en) need a set HuggingFace Access Key with read permission.

**Example datasets**
- [Train](./api/examples/emb/binary/train.json)
- [Eval](./api/examples/emb/binary/eval.json)

#### `/train/rerank` ReRankerings

LoRA-PEFT for re-ranking models.

**Supported Models**
- bge-reranker using [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker) or
- Any [HuggingFace embeddings model](https://huggingface.co/spaces/mteb/leaderboard).

**Example datasets**
- [bge-reranker](./api/examples/rr/bge-reranker.jsonl)
- [embeddings](./api/examples/emb/binary/train.json)

### `/compose` - BYO-MoE
[Try API endpoint](http://localhost:8723/docs#/default/build_your_own_mixture_of_experts_byo_moe__post)

`/compose` is an endpoint for combining Mistral or Llama models of the same size into Mixture-of-Experts models. The endpoint will combine the self-attention and layer normalization parameters from a "base" model with the MLP parameters from a set of "expert" models.

`/compose` endpoint can be used with minimal or no GPU.

`/compose` endpoint uses its own JSON configuration syntax, which looks like so:
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

Uses the hidden state representations of the positive/negative prompts for MoE gate parameters. Best quality and most effective option; the default. Requires evaluating each prompt using the base model so you might not be able to use this on constrained hardware (depending on the model). 

Coming Soon: use `--load-in-8bit` or `--load-in-4bit` to reduce VRAM usage.

##### "cheap_embed"

Uses only the raw token embedding of the prompts, using the same gate parameters for every layer. Distinctly less effective than "hidden". Can be run on much, much lower end hardware.

##### "random"

Randomly initializes the MoE gates. Good for if you are going to fine tune the model afterwards, or maybe if you want something a little unhinged? I won't judge.

### `/optimize` - LaserRMT 
[Try API endpoint](http://localhost:8723/docs#/default/laser_llm_laser__post)
`request body`
```json
{
    "base_model_name" : "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "laser_model_name": "TinyLaser",
    "top_k_layers": 15
}
```
LaserRMT optimizes LLMs combining Layer-Selective Rank Reduction (LASER) and the Marchenko-Pastur law from Random Matrix Theory. This method targets model complexity reduction while maintaining or enhancing performance, making it more efficient than the traditional brute-force search method.

1. LASER Framework Adaptation: LaserRMT adapts the LASER technique, which reduces the complexity of neural networks by selectively pruning the weights of a model's layers.
2. Marchenko-Pastur Law Integration: The Marchenko-Pastur law, a concept from Random Matrix Theory used to determine the distribution of eigenvalues in large random matrices, guides the identification of redundant components in LLMs. This allows for effective complexity reduction without loss of key information.
3. Enhanced Model Performance: By systematically identifying and eliminating less important components in the model's layers, LaserRMT can potentially enhance the model's performance and interpretability.
4. Efficient Optimization Process: LaserRMT provides a more efficient and theoretically robust framework for optimizing large-scale language models, setting a new standard for language model refinement.

This approach opens new avenues for optimizing neural networks, underscoring the synergy between advanced mathematical theories and practical AI applications. LaserRMT sets a precedent for future developments in the field of LLM optimization.

### `/quantize/{method}`
[Try API endpoint](http://localhost:8723/docs#/default/quantize__post)
#### AWQ
Generate AWQ-quantizations optimized for GPU-inference.

### `/publish` to HuggingFace 🤗

[Try API endpoint](http://localhost:8723/docs#/default/publish_endpoint_publish_post)
Publish generated local models to 🤗 HuggingFace Hub.

## Explaining Resources
Some explaining resources for concepts, technologies and tools used in this repository.

1. [MergeKit Mixtral](https://github.com/cg123/mergekit/tree/mixtral)
2. [Mixture of Experts for Clowns (at a Circus)](https://goddard.blog/posts/clown-moe)
3. [Fernando Fernandes Neto, David Golchinfar and Eric Hartford. "Optimizing Large Language Models Using Layer-Selective Rank Reduction and Random Matrix Theory." 2024.](https://github.com/cognitivecomputations/laserRMT)
4. [The Truth is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction](https://arxiv.org/pdf/2312.13558.pdf)
5. [An Empirical view of Marchenko-Pastur Theorem](https://medium.com/swlh/an-empirical-view-of-marchenko-pastur-theorem-1f564af5603d)
