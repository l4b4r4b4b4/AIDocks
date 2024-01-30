from fastapi import FastAPI, BackgroundTasks
import torch
from pydantic import BaseModel
from typing import Optional, List
from icecream import ic
from src.rmt_laser_snr import ModelModifier
from enum import Enum
import yaml
import json
import subprocess
import os
from enum import Enum
from huggingface_hub import create_repo
from huggingface_hub import login
from huggingface_hub import HfApi

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    print("We will use the GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

app = FastAPI(
    title="AIDocks",
    description="Build your Own: Mixture-of-Experts (MoE) models, Laser-optimize your LLMs, fine-tune Embeddings & ReRanking models",
    version="0.0.1",
)


@app.get("/")
async def health():
    ic("GET /")
    return {"msg": "This is AIDocks https://github.com/l4b4r4b4b4/ai-docks"}


class ModelType(str, Enum):
    base = "base"
    local = "local"


@app.get("/models/{model_type}")
async def get_models(model_type: ModelType):
    if model_type == "base":
        base_path = "/.hf-cache/hub"
    elif model_type == "local":
        base_path = "models"
    else:
        return {"models": "Error"}

    models = [
        name.lstrip("models--")
        for name in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, name))
    ]
    return {"models": models}


@app.get("/datasets")
async def get_datasets():
    base_path = "/.hf-cache/datasets"
    datasets = [
        name
        for name in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, name))
    ]
    return {"datasets": datasets}


class LaserInput(BaseModel):
    base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    laser_model_name: str = "TinyLlama-1.1B-Chat-v1.0-Laser"
    top_k_layers: Optional[int] = 15
    benchmark_datasets: Optional[List[str]] = [
        "gsm8k",
        "mmlu",
        "winogrande",
        "arc_challenge",
        "hellaswag",
        "truthfulqa_mc2",
        "wikitext2",
        "ptb",
    ]
    seq_len: Optional[int] = 256


@app.post("/laser/")
async def laser_llm(request_body: LaserInput, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_laser, request_body)
    return {"message": "Laser is running in the background ..."}


async def run_laser(request_body: LaserInput):
    base_model_name = request_body.base_model_name
    laser_model_name = request_body.laser_model_name
    benchmark_datasets = request_body.benchmark_datasets
    seq_len = request_body.seq_len
    modifier = ModelModifier(
        base_model_name, datasets=benchmark_datasets, seq_len=seq_len
    )

    layer_numbers = list(range(31, -1, -1))
    layer_numbers = [f".{l}." for l in layer_numbers]
    ic(layer_numbers)

    layer_types = [
        "mlp.gate_proj",
        "mlp.down_proj",
        "mlp.up_proj",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ]

    modifier.assess_layers_snr(layer_types, layer_numbers)
    # Select top k layers
    top_k_layers = modifier.select_layers_for_modification(request_body.top_k_layers)
    print(top_k_layers, flush=True)

    modifier.test_and_modify_layers(top_k_layers)
    modifier.save_model(str(f"models/{laser_model_name}"))
    return "ok"


class Expert(BaseModel):
    source_model: str
    positive_prompts: List[str]
    negative_prompts: Optional[List[str]] = None


class GateMode(str, Enum):
    hidden = "hidden"
    cheap_embed = "cheap_embed"
    random = "random"


class DType(str, Enum):
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"


class BYOMoEConfig(BaseModel):
    base_model: str = "cognitivecomputations/dolphin-2.6-mistral-7b-dpo"
    gate_mode: Optional[GateMode] = GateMode.hidden
    dtype: Optional[DType] = DType.bfloat16
    experts: Optional[List[Expert]] = [
        Expert(
            source_model="teknium/OpenHermes-2.5-Mistral-7B",
            positive_prompts=[
                "instruction",
                "solutions",
                "chat",
                "questions",
                "comprehension",
            ],
        ),
        Expert(
            source_model="openaccess-ai-collective/DPOpenHermes-7B",
            positive_prompts=[
                "mathematics",
                "optimization",
                "code",
                "step-by-step",
                "science",
            ],
            negative_prompts=["chat" "questions"],
        ),
        Expert(
            source_model="azale-ai/Starstreak-7b-beta",
            positive_prompts=["chat", "questions", "answer", "indonesian", "indonesia"],
        ),
        Expert(
            source_model="azale-ai/Starstreak-7b-beta",
            positive_prompts=["chat", "questions", "answer", "arabic", "arab"],
        ),
        Expert(
            source_model="davidkim205/komt-mistral-7b-v1",
            positive_prompts=["chat", "questions", "answer", "korean", "korea"],
        ),
        Expert(
            source_model="OpenBuddy/openbuddy-zephyr-7b-v14.1",
            positive_prompts=["chat", "questions", "answer", "chinese", "china"],
        ),
        Expert(
            source_model="manishiitg/open-aditi-hi-v1",
            positive_prompts=["chat", "questions", "answer", "hindi", "india"],
        ),
        Expert(
            source_model="VAGOsolutions/SauerkrautLM-7b-v1-mistral",
            positive_prompts=[
                "chat",
                "questions",
                "answer",
                "german",
                "deutsch",
                "Germany",
            ],
        ),
    ]


class BYOMoEInput(BaseModel):
    moe_name: str = "TinyLlama-ClownCar"
    config: BYOMoEConfig


@app.post("/byo-moe/")
async def build_your_own_mixture_of_experts(
    request_body: BYOMoEInput, background_tasks: BackgroundTasks
):
    ic("/byo-moe")
    background_tasks.add_task(byo_moe, request_body)
    return {"message": "BYO-MoE task is running in the background ..."}


async def byo_moe(request_body: BYOMoEInput):
    ic(request_body)
    json_config = request_body.config.json(exclude_none=True)
    config = json.loads(json_config)
    config_path = f"data/moe-configs/{request_body.moe_name}.yaml"
    with open(config_path, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    command = str(f"mergekit-moe {config_path} models/{request_body.moe_name}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    output, error = process.communicate()

    if error:
        print(f"An error occurred: {repr(error)}")
    else:
        print(output.decode())


class PublishInput(BaseModel):
    local_model_name: str = "TinyLlama-ClownCar"
    pub_model_name: str = "TinyLlama-ClownCar"
    trainer: str = "LHC88"


@app.post("/publish")
async def publish_endpoint(
    request_body: PublishInput, background_tasks: BackgroundTasks
):
    background_tasks.add_task(run_publish, request_body)
    return {"message": "Laser is running in the background ..."}


def run_publish(input: PublishInput):
    ic("Publishing to HF ...")
    trainer = input.trainer
    local_model_name = input.local_model_name
    pub_model_name = input.pub_model_name
    repo_id = f"{trainer}/{pub_model_name}"
    try:
        login(token=r"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")
        try:
            create_repo(repo_id)
        except Exception as e:
            raise e
        else:
            pass

        api = HfApi()

        api.upload_folder(
            folder_path=f"models/{local_model_name}",
            repo_id=repo_id,
            repo_type="model",
            revision="main",
        )
    except Exception as e:
        ic(e)
    else:
        pass
