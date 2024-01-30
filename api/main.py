from fastapi import FastAPI, BackgroundTasks, Path, HTTPException
import torch
from pydantic import BaseModel
from typing import Optional, List
from icecream import ic
from src.rmt_laser_snr import ModelModifier
import yaml
import json
import subprocess
import os
from enum import Enum
from huggingface_hub import create_repo
from huggingface_hub import login
from huggingface_hub import HfApi

from datasets import load_dataset

from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import DPOTrainer
from unsloth import FastMistralModel

from trl import SFTTrainer
from transformers import TrainingArguments
from transformers.utils import logging
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

os.environ["WANDB_DISABLED"] = "true"

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    print("We will use the GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

HAS_BFLOAT16 = torch.cuda.is_bf16_supported()

app = FastAPI(
    title="AIDocks",
    description="Build your Own: Mixture-of-Experts (MoE) models, Laser-optimize your LLMs, fine-tune Embeddings & ReRanking models",
    version="0.0.1",
)


@app.get("/")
async def health():
    ic("GET /")
    return {"msg": "This is AIDocks https://github.com/l4b4r4b4b4/ai-docks"}


class TrainingConfig(BaseModel):
    lora_rank: int = 64
    lora_alpha: int = 16
    max_steps: int = 4263
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    learning_rate: float = 0.7e-4
    max_prompt_length: int = 1024 * 2
    max_total_length: int = 1024 * 4
    num_train_epochs: int = 1
    logging_steps: int = 1
    resume_from_checkpoint: str = "checkpoint-3500"
    use_gradient_checkpointing: bool = True
    load_in_4bit: bool = False
    max_seq_length: int = 8192
    training_jsonl: str = "llm_dpo_training_data.jsonl"
    optim_fn: str = "adamw_8bit"
    eval_jsonl: Optional[str] = None


class Method(str, Enum):
    dpo = "dpo"
    sft = "sft"


class LLMTrainingInput(BaseModel):
    base_model_name: str = "TinyLlama-ClownCar"
    peft_model_name: str = "TinyLlama-SpecialClownCar"
    method: Method = Method.dpo
    config: TrainingConfig


@app.post("/train/llm")
async def training_endpoint(
    request_body: LLMTrainingInput,
    background_tasks: BackgroundTasks,
):
    if request_body == "dpo":
        background_tasks.add_task(run_dpo_training, request_body)
    elif request_body == "sft":
        background_tasks.add_task(run_sft_training, request_body)
    return {"message": "Training is running in the background ..."}


def run_dpo_training(input: LLMTrainingInput):
    ic("Training to HF ...")
    # Configuration Input
    output_model_path = str(f"models/{input.peft_model_name}")
    model_name = input.base_model_name
    load_in_4bit = input.config.load_in_4bit
    resume_from_checkpoint = input.config.resume_from_checkpoint
    use_gradient_checkpointing = input.config.use_gradient_checkpointing
    max_seq_length = input.config.max_seq_length
    training_jsonl = input.config.training_jsonl
    eval_jsonl = input.config.eval_jsonl
    lora_rank = input.config.lora_rank
    lora_alpha = input.config.lora_alpha
    per_device_train_batch_size = input.config.per_device_train_batch_size
    gradient_accumulation_steps = input.config.gradient_accumulation_steps
    num_train_epochs = input.config.num_train_epochs
    logging_steps = input.config.logging_steps
    optim_fn = input.config.optim_fn
    max_total_length = input.config.max_total_length
    max_prompt_length = input.config.max_prompt_length
    if max_total_length <= max_prompt_length:
        raise HTTPException(
            status_code=400,
            detail="max_total_length must be greater than max_prompt_length",
        )

    # Setup
    dataset_training = load_dataset(
        "json", data_files=f"data/{training_jsonl}", split="train"
    )
    if eval_jsonl:
        dataset_eval = load_dataset(
            "json", data_files=f"data/{eval_jsonl}", split="train"
        )

    # PatchDPOTrainer()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        # task_type="CAUSAL_LM",
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=3407,
        max_seq_length=max_seq_length,
    )
    if eval_jsonl:
        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=TrainingArguments(
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_ratio=0.1,
                num_train_epochs=num_train_epochs,
                fp16=not HAS_BFLOAT16,
                bf16=HAS_BFLOAT16,
                logging_steps=logging_steps,
                optim=optim_fn,
                seed=42,
                output_dir=output_model_path,
            ),
            beta=0.1,
            train_dataset=dataset_training,
            eval_dataset=dataset_eval,
            tokenizer=tokenizer,
            max_length=max_total_length,
            max_prompt_length=max_prompt_length,
        )
    else:
        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=TrainingArguments(
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_ratio=0.1,
                num_train_epochs=num_train_epochs,
                fp16=not HAS_BFLOAT16,
                bf16=HAS_BFLOAT16,
                logging_steps=logging_steps,
                optim=optim_fn,
                seed=42,
                output_dir=output_model_path,
            ),
            beta=0.1,
            train_dataset=dataset_training,
            tokenizer=tokenizer,
            max_length=max_total_length,
            max_prompt_length=max_prompt_length,
        )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    ic(start_gpu_memory)
    available_memory = (
        round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3) - start_gpu_memory
    )
    ic(available_memory)
    torch.cuda.empty_cache()
    if resume_from_checkpoint:
        dpo_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        dpo_trainer.train()


def run_sft_training(input: LLMTrainingInput):
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    lr_scheduler_type = "linear"
    weight_decay = 0.01
    warmup_steps = 100
    random_state = 3407
    model_name = input.base_model_name
    load_in_4bit = input.config.load_in_4bit
    max_seq_length = input.config.max_total_length
    max_steps = input.config.max_steps
    learning_rate = input.config.learning_rate
    gradient_accumulation_steps = input.config.gradient_accumulation_steps
    per_device_train_batch_size = input.config.per_device_train_batch_size
    optimizer = input.config.optim_fn
    use_gradient_checkpointing = input.config.use_gradient_checkpointing
    lora_rank = input.config.lora_rank
    lora_alpha = input.config.lora_alpha
    output_model_path = str(f"models/{input.peft_model_name}")

    dataset = load_dataset(
        "json", data_files="names_sft_training_data.jsonl", split="train"
    )

    model, tokenizer = FastMistralModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        token=os.getenv(
            "HUGGINGFACEHUB_API_TOKEN"
        ),  # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    model = FastMistralModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0,  # Currently only supports dropout = 0
        bias="none",  # Currently only supports bias = "none"
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=3407,
        max_seq_length=max_seq_length,
    )

    max_steps = int(
        max_steps / (per_device_train_batch_size * gradient_accumulation_steps)
    )
    max_steps = max_steps * 2
    max_steps = int(max_steps)

    logging.set_verbosity_info()

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        packing=False,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            bf16=True,
            num_epochs=3,
            logging_steps=40,
            output_dir=output_model_path,
            optim=optimizer,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            seed=random_state,
        ),
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    torch.cuda.empty_cache()
    trainer_stats = trainer.train()
    ic(trainer_stats)


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


@app.post("/optimize")
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


class ComposeConfig(BaseModel):
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


class ComposeInput(BaseModel):
    moe_name: str = "MultiLang-Mistral-ClownCar"
    config: ComposeConfig


@app.post("/compose")
async def build_your_own_mixture_of_experts(
    request_body: ComposeInput, background_tasks: BackgroundTasks
):
    ic("/compose")
    background_tasks.add_task(byo_moe, request_body)
    return {"message": "BYO-MoE task is running in the background ..."}


async def byo_moe(request_body: ComposeInput):
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

class QuantMethod(str, Enum):
    awq = "awq"
    gptq = "gptq"
    gguf = "gguf"


class GSOptions(int, Enum):
    gs1 = 32
    gs2 = 64
    gs3 = 128


class QuantOptions(int, Enum):
    q4 = 4
    q8 = 8
    
class VersionOptions(str, Enum):
    gemm = "GEMM"


class QuantizeInput(BaseModel):
    base_model_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    quant_model_name: str = "TinyLlama-1.1B-Chat-v1.0"
    quant_method: QuantMethod = QuantMethod.awq
    gs: GSOptions = GSOptions.gs1
    zero_point: bool = True
    version: VersionOptions = VersionOptions.gemm
    quant: QuantOptions = QuantOptions.q4
    calibration_data: Optional[str] = None


@app.post("/quantize")
async def quantize_endpoint(
    request_body: QuantizeInput, background_tasks: BackgroundTasks
):
    background_tasks.add_task(run_quantization, request_body)
    return {"message": "Quantizeing is running in the background ..."}


def run_quantization(input: QuantizeInput):
    ic("Publishing to HF ...")
    calibration_data = input.calibration_data
    base_model_path = input.base_model_path
    quant_model_name = input.quant_model_name
    quant_method = input.quant_method.value
    gs = input.gs.value
    quant = input.quant.value
    zero_point = input.zero_point
    version = input.version.value
    quant_path = f"models/{quant_model_name}-{quant}bit-{str(quant_method).upper()}"
    modules_to_not_convert = ["gate"]
    quant_config = {
        "zero_point": zero_point,
        "q_group_size": gs,
        "w_bit": quant,
        "version": version,
        "modules_to_not_convert": modules_to_not_convert,
    }

    def load_data():
        text_path = "20k_random_data.txt"
        data = load_dataset(
            "text", data_files={"train": [text_path], "test": text_path}, split="train"
        )
        return [
            text
            for text in data["text"]
            if text.strip() != "" and len(text.split(" ")) > 20
        ]

    # Load model
    # NOTE: pass safetensors=True to load safetensors
    model = AutoAWQForCausalLM.from_pretrained(
        base_model_path, safetensors=True, use_cache=True  # **{"low_cpu_mem_usage": True, }
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # Quantize
    if calibration_data:
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            modules_to_not_convert=modules_to_not_convert,
            calib_data=load_data()
        )
    else:
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            modules_to_not_convert=modules_to_not_convert,
        )

    # Save quantized model
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

    print(f'Model is quantized and saved at "{quant_path}"')


class PublishInput(BaseModel):
    local_model_name: str = "TinyLlama-ClownCar"
    pub_model_name: str = "TinyLlama-ClownCar"
    trainer: str = "LHC88"


@app.post("/publish")
async def publish_endpoint(
    request_body: PublishInput, background_tasks: BackgroundTasks
):
    background_tasks.add_task(run_publish, request_body)
    return {"message": "Publishing is running in the background ..."}


def run_publish(input: PublishInput):
    ic("Publishing to HF ...")
    trainer = input.trainer
    local_model_name = input.local_model_name
    pub_model_name = input.pub_model_name
    repo_id = f"{trainer}/{pub_model_name}"
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if hf_token is None:
        raise ValueError("Environment variable HUGGINGFACEHUB_API_TOKEN is not set")
    try:
        login(token=hf_token)
        try:
            create_repo(repo_id)
        except Exception as e:
            ic(e)
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
