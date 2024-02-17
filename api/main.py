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
from transformers import (
    # AutoModel,
    TrainingArguments,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
)

from trl import DPOTrainer, SFTTrainer
from unsloth import FastMistralModel

from transformers.utils import logging
from awq import AutoAWQForCausalLM

# PEFT imports
from accelerate import Accelerator
import logging
from accelerate.logging import get_logger
from datasets import load_dataset
from accelerate.utils import set_seed
import datasets
import transformers
import random
from src.AutoModelForSentenceEmbedding import (
    AutoModelForSentenceEmbedding,
    get_loss,
    get_cosine_embeddings,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from torch.utils.data import DataLoader

from src.dora import dora
import math
import evaluate
from tqdm import tqdm


os.environ["WANDB_DISABLED"] = "true"
logger = get_logger(__name__)
model_output_dir = "models"
import torch
import warnings

gpu_ok = False
if torch.cuda.is_available():
    device_cap = torch.cuda.get_device_capability()
    if device_cap in ((7, 0), (8, 0), (9, 0)):
        gpu_ok = True

if not gpu_ok:
    warnings.warn(
        "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
        "than expected."
    )

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    print("We will use the GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

HAS_BFLOAT16 = torch.cuda.is_bf16_supported()


def save_model_hook(models, weights, output_dir):
    for i, model in enumerate(models):
        try:
            model.save_pretrained(output_dir, state_dict=weights[i])
            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()
        except Exception as e:
            ic(e)
            ic("Skipping saving weights!")


def load_model_hook(models, input_dir):
    while len(models) > 0:
        model = models.pop()
        # pop models so that they are not loaded again
        if hasattr(model, "active_adapter") and hasattr(model, "load_adapter"):
            model.load_adapter(input_dir, model.active_adapter, is_trainable=True)


app = FastAPI(
    title="AIDocks",
    description="Build your Own: Mixture-of-Experts (MoE) models, Laser-optimize your LLMs, fine-tune Embeddings & ReRanking models",
    version="0.0.1",
)


@app.get("/")
async def application_health():
    ic("GET /")
    return {"msg": "This is AIDocks https://github.com/Holocene Intelligence/ai-docks"}


class DoraInput(BaseModel):
    input_dim: int = 10
    learning_rate: int = 0.001
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    datasets: List[str] = ["wikitext", "wikitext-2", "wikitext-103"]


@app.post("/dora")
async def dora_endpoint(
    request_body: DoraInput,
    background_tasks: BackgroundTasks,
):
    input_dim = request_body.input_dim
    learning_rate = request_body.learning_rate
    base_model = request_body.base_model
    background_tasks.add_task(dora, learning_rate, input_dim, base_model, datasets)

    return request_body


class ModelType(str, Enum):
    base = "Base Models"
    local = "Fine-tuned Models"


@app.get("/models/{model_type}")
async def get_base_models_or_fine_tunes(model_type: ModelType):
    if model_type == ModelType.base:
        base_path = "/.hf-cache/hub"
    elif model_type == ModelType.local:
        base_path = "models"
    else:
        return {"models": "Error"}

    models = [
        (name.lstrip("models--"))
        for name in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, name))
    ]
    return {"models": models}


@app.get("/datasets")
async def get_loaded_datasets():
    base_path = "/.hf-cache/datasets"
    datasets = [
        name
        for name in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, name))
    ]
    return {"datasets": datasets}


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
    training_data: str = "llm_dpo_training_data.jsonl"
    eval_data: Optional[str] = None
    optim_fn: str = "adamw_8bit"


class TrainModelMethod(str, Enum):
    peft = "peft"
    dpo = "dpo"
    sft = "sft"


class TrainModelCategory(str, Enum):
    emb = "emb"
    rr = "rr"
    llm = "llm"


class LLMTrainingInput(BaseModel):
    base_model_path: str = "TinyLlama-ClownCar"
    peft_model_name: str = "TinyLlama-SpecialClownCar"
    category: TrainModelCategory = TrainModelCategory.llm
    method: TrainModelMethod = TrainModelMethod.dpo
    config: TrainingConfig


@app.post("/train")
async def training_endpoint(
    request_body: LLMTrainingInput,
    background_tasks: BackgroundTasks,
):
    train_category = request_body.category.value
    train_method = request_body.method.value
    if train_category == "emb":
        ic("Embeddings PEFT")
        background_tasks.add_task(run_emb_training, request_body)
    elif train_category == "rr":
        ic("ReRankings PEFT")
        background_tasks.add_task(run_rr_training, request_body)
    elif train_category == "llm":
        if train_method == "dpo":
            background_tasks.add_task(run_dpo_training, request_body)
        elif train_method == "sft":
            background_tasks.add_task(run_sft_training, request_body)
    # TODO combined retrieval & generation fine-tuning
    return {"message": "Training is running in the background ..."}


def emb_training(input: LLMTrainingInput):
    ic("PEFT Training Embeddings model ...")
    gradient_accumulation_steps = input.config.gradient_accumulation_steps
    base_model_path = input.base_model_path
    resume_from_checkpoint = input.config.resume_from_checkpoint
    per_device_train_batch_size = input.config.per_device_train_batch_size
    num_train_epochs = input.config.num_train_epochs
    learning_rate = input.config.learning_rate
    dataset_train = input.config.training_data
    dataset_eval = input.config.eval_data
    base_model_name = base_model_path.split("/")[1]
    output_model_name = str(f"{base_model_name}_ga_{gradient_accumulation_steps}")
    output_model_path = str(f"{model_output_dir}/{output_model_name}")

    lr_scheduler_type = "cosine"
    num_warmup_steps = 1
    seed = 42
    max_length_0 = 256
    max_length_1 = 8192
    max_train_steps = None
    per_device_eval_batch_size = 1
    weight_decay = 0.0
    gradient_checkpointing = True
    with_tracking = False
    truncate = True
    report_to = "none"
    use_peft = True
    checkpointing_steps = "epoch"
    accelerator = (
        Accelerator(log_with="none", project_dir="models/embed")
        if False
        else Accelerator()
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if model_output_dir is not None:
            os.makedirs(output_model_path, exist_ok=True)
    accelerator.wait_for_everyone()
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    dataset = load_dataset(
        "json",
        data_files={
            "train": dataset_train,
            "eval": dataset_eval,
        },
    )
    accelerator.print(dataset)

    def preprocess_function_emb(examples, preprompt=""):
        q1 = [f"{preprompt}{x}" for x in examples["query"]]

        q1_tk = tokenizer(
            q1, padding="max_length", max_length=max_length_0, truncation=truncate
        )
        result = {f"query_{k}": v for k, v in q1_tk.items()}

        q2 = []
        for x in examples["content"]:
            q2.append(x)
        q2_tk = tokenizer(
            q2, padding="max_length", max_length=max_length_1, truncation=truncate
        )
        for k, v in q2_tk.items():
            result[f"document_{k}"] = v

        result["labels"] = examples["score"]

        return result

    dataset.cleanup_cache_files()
    processed_datasets = dataset.map(
        preprocess_function_emb,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    for index in random.sample(range(len(processed_datasets["train"])), 3):
        logger.info(
            f"Sample {index} of the training set: {processed_datasets['train'][index]}."
        )

    embeddings_model = AutoModelForSentenceEmbedding(base_model_path, tokenizer)
    peft_config = LoraConfig(
        r=input.config.lora_rank,
        lora_alpha=input.config.lora_alpha,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["key", "query", "value"],
    )
    embeddings_model = get_peft_model(embeddings_model, peft_config)
    embeddings_model.print_trainable_parameters()
    accelerator.print(embeddings_model)
    # get dataloaders
    train_dataloader = DataLoader(
        processed_datasets["train"],
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=per_device_train_batch_size,
        pin_memory=True,
    )

    eval_dataloader = DataLoader(
        processed_datasets["eval"],
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=per_device_eval_batch_size,
        pin_memory=True,
    )
    optimizer = torch.optim.Adam(embeddings_model.parameters(), lr=learning_rate)
    torch.cuda.empty_cache()
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )
    # Prepare everything with our `accelerator`.
    (
        embeddings_model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        embeddings_model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    )
    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)
    metric = evaluate.load("roc_auc")  # roc_auc | accuracy | f1
    total_batch_size = (
        per_device_train_batch_size
        * accelerator.num_processes
        * gradient_accumulation_steps
    )

    # saving and loading checkpoints for resuming training
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(processed_datasets['train'])}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    torch.cuda.empty_cache()
    if resume_from_checkpoint:
        if resume_from_checkpoint is not None or resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {resume_from_checkpoint}")
            accelerator.load_state(resume_from_checkpoint)
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            completed_steps = resume_step // gradient_accumulation_steps

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, num_train_epochs):
        torch.cuda.empty_cache()
        embeddings_model.train()
        # if with_tracking:
        total_loss = 0
        if (
            resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader
        print(active_dataloader)
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(embeddings_model):
                q1_embs = embeddings_model(
                    **{
                        k.replace("query_", ""): v
                        for k, v in batch.items()
                        if "query_" in k
                    }
                )
                q2_embs = embeddings_model(
                    **{
                        k.replace("document_", ""): v
                        for k, v in batch.items()
                        if "document_" in k
                    }
                )
                loss = get_loss(
                    get_cosine_embeddings(q1_embs, q2_embs), batch["labels"]
                )
                total_loss += accelerator.reduce(loss.detach().float(), reduction="sum")
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                embeddings_model.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if (step + 1) % 100 == 0:
                logger.info(f"Step: {step+1}, Loss: {total_loss/(step+1)}")
                if with_tracking:
                    accelerator.log(
                        {"train/loss": total_loss / (step + 1)},
                        step=completed_steps,
                    )

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if output_dir is not None:
                        output_dir = os.path.join(output_model_path, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= max_train_steps:
                break

        embeddings_model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                q1_embs = embeddings_model(
                    **{
                        k.replace("query_", ""): v
                        for k, v in batch.items()
                        if "query_" in k
                    }
                )
                q2_embs = embeddings_model(
                    **{
                        k.replace("document_", ""): v
                        for k, v in batch.items()
                        if "document_" in k
                    }
                )
                prediction_scores = get_cosine_embeddings(q1_embs, q2_embs)
            prediction_scores, references = accelerator.gather_for_metrics(
                (prediction_scores, batch["labels"])
            )
            metric.add_batch(
                # predictions=prediction_scores,
                prediction_scores=prediction_scores,
                references=references,
            )

        result = metric.compute()
        result = {f"eval/{k}": v for k, v in result.items()}
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}:", result)
        if with_tracking:
            result["train/epoch_loss"] = total_loss / len(train_dataloader)
            accelerator.log(result, step=completed_steps)

        if model_output_dir is not None:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                if isinstance(checkpointing_steps, str):
                    accelerator.save_state(
                        os.path.join(output_model_path, f"epoch_{epoch}")
                    )
                accelerator.unwrap_model(embeddings_model).save_pretrained(
                    output_model_path,
                    state_dict=accelerator.get_state_dict(
                        accelerator.unwrap_model(embeddings_model)
                    ),
                )
                tokenizer.save_pretrained(output_model_path)
            accelerator.wait_for_everyone()
    accelerator.end_training()


def run_emb_training(input: LLMTrainingInput):
    ic("PEFT Training Embeddings model ...")
    emb_training(input)


def run_rr_training(input: LLMTrainingInput):
    ic("PEFT Training ReRankings model ...")
    base_model_name = input.base_model_path
    peft_model_name = input.peft_model_name
    learning_rate = input.config.learning_rate
    num_train_epochs = input.config.num_train_epochs
    per_device_train_batch_size = input.config.per_device_train_batch_size
    gradient_accumulation_steps = input.config.gradient_accumulation_steps
    # If model == bge-reranker
    if "bge-reranker" in base_model_name.lower():
        command = [
            "torchrun",
            "--nproc_per_node",
            "1",
            "-m",
            "FlagEmbedding.reranker.run",
            "--output_dir",
            str(f"models/rr/{peft_model_name}"),
            "--model_name_or_path",
            base_model_name,
            "--train_data",
            str(f"data/{input.config.training_data}"),
            "--learning_rate",
            str(learning_rate),
            "--fp16",
            "--num_train_epochs",
            str(num_train_epochs),
            "--per_device_train_batch_size",
            str(per_device_train_batch_size),
            "--gradient_accumulation_steps",
            str(gradient_accumulation_steps),
            "--dataloader_drop_last",
            "True",
            "--train_group_size",
            "4",
            "--max_len",
            "512",
            "--weight_decay",
            "0.01",
            "--logging_steps",
            "10",
        ]

        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(f"Error occurred: {stderr.decode('utf-8')}")
        else:
            print(f"Output: {stdout.decode('utf-8')}")
    else:
        emb_training(input)


def run_dpo_training(input: LLMTrainingInput):
    ic("DPO Training to HF ...")
    # Configuration Input
    output_model_path = str(f"models/llm/{input.peft_model_name}")
    model_name = input.base_model_path
    load_in_4bit = input.config.load_in_4bit
    resume_from_checkpoint = input.config.resume_from_checkpoint
    use_gradient_checkpointing = input.config.use_gradient_checkpointing
    max_seq_length = input.config.max_seq_length
    training_jsonl = input.config.training_data
    eval_jsonl = input.config.eval_data
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
    model_name = input.base_model_path
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
    top_k_layers: Optional[int] = 2
    seqlen: Optional[int] = 128
    # load_in_8bit: Optional[bool] = False


@app.post("/optimize")
async def laser_llm(request_body: LaserInput, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_laser, request_body)
    return {"message": "Laser is running in the background ..."}


async def run_laser(request_body: LaserInput):
    base_model_name = request_body.base_model_name
    laser_model_name = request_body.laser_model_name
    # load_in_8bit = request_body.load_in_8bit
    seqlen = request_body.seqlen
    modifier = ModelModifier(
        base_model_name, seqlen=seqlen  # , load_in_8bit=load_in_8bit
    )
    # TODO get max n layers from model config
    layer_numbers = list(range(request_body.top_k_layers, -1, -1))
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
    modifier.save_model(str(f"models/llm/{laser_model_name}"))
    finish_msg = (
        f"Model {laser_model_name} is ready for use at `models/llm/{laser_model_name}`"
    )
    ic(finish_msg)
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
    base_model: str = "openaccess-ai-collective/DPOpenHermes-7B"
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
            source_model="codellama/CodeLlama-13b-hf",
            positive_prompts=[
                "coding",
                "programming",
                "code",
                "programming language",
            ],
            negative_prompts=["chat", "questions", "python"],
        ),
        Expert(
            source_model="codellama/CodeLlama-13b-Python-hf",
            positive_prompts=[
                "python",
                "pip",
                "coding",
                "programming",
                "code",
                "programming language",
            ],
            negative_prompts=["chat", "questions"],
        ),
        Expert(
            source_model="cognitivecomputations/dolphin-2.6-mistral-7b-dpo",
            positive_prompts=[
                "mathematics",
                "optimization",
                "step-by-step",
                "science",
            ],
            negative_prompts=["chat", "questions"],
        ),
        # Expert(
        #     source_model="Ashishkr/llama-2-medical-consultation",
        #     positive_prompts=[
        #         "medical consultation",
        #         "health",
        #     ],
        #     negative_prompts=["chat", "questions"],
        # ),
        Expert(
            source_model="tom92119/llama-2-7b-bedtime-story",
            positive_prompts=[
                "bedtime story",
                "Once upon a time",
                "storytelling",
                "narrator",
            ],
            negative_prompts=["chat", "questions"],
        ),
        Expert(
            source_model="Norquinal/Mistral-7B-storywriter",
            positive_prompts=[
                "story",
                "Once upon a time",
                "storytelling",
                "narrator",
            ],
            negative_prompts=["chat", "questions"],
        ),
        Expert(
            source_model="meetkai/functionary-small-v2.2",
            positive_prompts=[
                "function calls",
                "functions",
                "constrained grammar",
                "API calls",
                "LLM Tools",
            ],
            negative_prompts=[
                "chat",
                "questions",
                "instruction",
                "solutions",
                "chat",
                "comprehension",
                "mathematics",
                "optimization",
                "code",
                "step-by-step",
                "science",
            ],
        ),
        Expert(
            source_model="azale-ai/Starstreak-7b-beta",
            positive_prompts=["indonesian", "indonesia"],
        ),
        Expert(
            source_model="gagan3012/Mistral_arabic_dpo",
            positive_prompts=["arabic", "arab"],
        ),
        Expert(
            source_model="davidkim205/komt-mistral-7b-v1",
            positive_prompts=["korean", "korea"],
        ),
        Expert(
            source_model="OpenBuddy/openbuddy-zephyr-7b-v14.1",
            positive_prompts=["chinese", "china"],
        ),
        Expert(
            source_model="manishiitg/open-aditi-hi-v1",
            positive_prompts=["hindi", "india"],
        ),
        Expert(
            source_model="VAGOsolutions/SauerkrautLM-7b-v1-mistral",
            positive_prompts=[
                "german",
                "deutsch",
                "Germany",
            ],
        ),
        Expert(
            source_model="bineric/NorskGPT-Mistral-7b",
            positive_prompts=["Norway", "Norwegian", "Norsk"],
        ),
        Expert(
            source_model="Droidfanat/llama-2-7b-custom-russian",
            positive_prompts=[
                "Russian",
                "Russia",
                "Русский",
                "Россия",
            ],
        ),
    ]


class ComposeInput(BaseModel):
    moe_name: str = "XPurpose-ClownCar-v0"
    config: ComposeConfig


@app.post("/compose")
async def compose_endpoint(
    request_body: ComposeInput, background_tasks: BackgroundTasks
):
    ic("/compose")
    background_tasks.add_task(compose, request_body)
    return {"message": "BYO-MoE task is running in the background ..."}


async def compose(request_body: ComposeInput):
    ic(request_body)
    json_config = request_body.config.json(exclude_none=True)
    config = json.loads(json_config)
    config_path = f"data/config/compose/{request_body.moe_name}.yaml"
    with open(config_path, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    command = str(f"mergekit-moe {config_path} models/llm/{request_body.moe_name}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    output, error = process.communicate()

    if error:
        print(f"An error occurred: {repr(error)}")
    else:
        print(output.decode())


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
        base_model_path,
        safetensors=True,
        use_cache=True,  # **{"low_cpu_mem_usage": True, }
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # Quantize
    if calibration_data:
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            modules_to_not_convert=modules_to_not_convert,
            calib_data=load_data(),
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
    local_model_type: str = "llm"
    local_model_name: str = "TinyLlama-ClownCar"
    pub_model_name: str = "TinyLlama-ClownCar"
    revision_tag: Optional[str] = "main"
    trainer: str = "LHC88"


@app.post("/publish")
async def publish_endpoint(
    request_body: PublishInput, background_tasks: BackgroundTasks
):
    background_tasks.add_task(run_publish, request_body)
    return {"message": "Publishing is running in the background ..."}


def run_publish(input: PublishInput):
    trainer = input.trainer
    local_model_type = input.local_model_type
    local_model_name = input.local_model_name
    pub_model_name = input.pub_model_name
    revision_tag = input.revision_tag

    start_msg = str(
        f"Publishing {local_model_name} to HuggingFace Hub as {pub_model_name} ..."
    )
    ic(start_msg)
    repo_id = f"{trainer}/{pub_model_name}"
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if hf_token is None:
        raise ValueError("Environment variable HUGGINGFACEHUB_API_TOKEN is not set")
    try:
        login(token=hf_token)
        try:
            create_repo(repo_id)
        except Exception as create_repo_exception:
            ic(create_repo_exception)
        else:
            pass

        api = HfApi()

        api.upload_folder(
            folder_path=f"models/{local_model_type}/{local_model_name}",
            repo_id=repo_id,
            repo_type="model",
            revision=revision_tag,
        )
    except Exception as e:
        ic(e)
    else:
        pass
