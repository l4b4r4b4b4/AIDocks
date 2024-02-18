import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from src.AutoModelForSentenceEmbedding import (
    AutoModelForSentenceEmbedding,
)

torch.manual_seed(0)

from enum import Enum


class PromptTemplate(str, Enum):
    chatml = "chatml"
    instruct = "instruct"
    tiny_llama = "tiny_llama"


def get_llm_prompt(
    system_instruction,
    user_prompt="",
    prompt_format: PromptTemplate = PromptTemplate.chatml,
    sep=" ",
):
    if prompt_format == PromptTemplate.chatml:
        return f"""<|im_start|>system
{system_instruction}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""
    elif prompt_format == PromptTemplate.tiny_llama:
        return f"""<|system|>
{system_instruction}</s>
<|user|>
{user_prompt}</s>
<|assistant|>
"""
    elif prompt_format == PromptTemplate.instruct:
        if user_prompt != "":
            system_instruction = system_instruction + sep + user_prompt

        return f"[INST] {system_instruction} [/INST]"

    else:
        raise Exception('prompt_format must be "chatml" or "mixtral"')


class TextDataset(Dataset):
    def __init__(
        self,
        prompts,
        answers,
        labels,
        tokenizer,
        max_input_length=256,
        max_output_length=256,
    ):
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.prompts = prompts
        self.answers = answers
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        prompt = self.prompts[index]
        answer = self.answers[index]
        label = self.labels[index]

        # Tokenize the prompt and answer separately
        encoded_prompt = self.tokenizer.encode_plus(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_input_length,
        )
        encoded_answer = self.tokenizer.encode_plus(
            answer,
            truncation=True,
            padding="max_length",
            max_length=self.max_output_length,
        )

        return {
            "prompt_input_ids": torch.tensor(
                encoded_prompt["input_ids"], dtype=torch.long
            ),
            "prompt_attention_mask": torch.tensor(
                encoded_prompt["attention_mask"], dtype=torch.long
            ),
            "answer_input_ids": torch.tensor(
                encoded_answer["input_ids"], dtype=torch.long
            ),
            "answer_attention_mask": torch.tensor(
                encoded_answer["attention_mask"], dtype=torch.long
            ),
            "label": torch.tensor(label, dtype=torch.float),
        }

    def __len__(self):
        return len(self.prompts)


# This layer is dropped into your pre-trained PyTorch model where nn.Linear is used
class DoRALayer(nn.Module):
    def __init__(self, d_in, d_out, rank=4, weight=None, bias=None):
        super().__init__()

        if weight is not None:
            self.weight = nn.Parameter(weight, requires_grad=False)
        else:
            self.weight = nn.Parameter(torch.Tensor(d_out, d_in), requires_grad=False)

        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = nn.Parameter(torch.Tensor(d_out), requires_grad=False)

        # m = Magnitude column-wise across output dimension
        self.m = nn.Parameter(self.weight.norm(p=2, dim=0, keepdim=True))

        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.lora_A = nn.Parameter(torch.randn(d_out, rank) * std_dev)
        self.lora_B = nn.Parameter(torch.zeros(rank, d_in))

    def forward(self, x):
        lora = torch.matmul(self.lora_A, self.lora_B)
        adapted = self.weight + lora
        column_norm = adapted.norm(p=2, dim=0, keepdim=True)
        norm_adapted = adapted / column_norm
        calc_weights = self.m * norm_adapted
        return F.linear(x, calc_weights, self.bias)


# Training function
# @torch.compile
def train(
    model,
    tokenizer,
    embeddings_model,
    criterion,
    optimizer,
    data_loader,
    max_input_length,
    max_output_length,
    epochs=5,
):
    model.train()
    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1} ...")
        for sample in data_loader:
            optimizer.zero_grad()
            # outputs = model(sample["prompt_input_ids"].to("cuda"))
            input_ids = sample["prompt_input_ids"].to("cuda")
            outputs = model.generate(
                input_ids,
                num_return_sequences=1,
                use_cache=False,
                output_hidden_states=False,
                output_attentions=False,
                pad_token_id=tokenizer.eos_token_id,
                
                # max_length=max_input_length + max_output_length,
            )  # max_length=100,
            # ic(type(outputs), outputs)
            input_length = len(input_ids[0])
            # Slice the output to remove the input tokens
            response_tokens = outputs[0][input_length:]

            output_string = tokenizer.decode(response_tokens, skip_special_tokens=True)
            ic(output_string)
            # Convert logits to token IDs
            # answer_enc = tokenizer(
            #     [output_string],
            #     return_tensors="pt",
            #     padding="max_length",
            #     max_length=self.output_length,
            # )
            # loss = criterion(logits, answers, labels)
            # loss.backward()
            # optimizer.step()
        # print(f"Epoch {epoch+1}, Loss: {loss.item()}")


def replace_linear_with_dora(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Get the input and output dimensions of the current nn.Linear layer
            d_in = module.in_features
            d_out = module.out_features

            # Create a new DoRALayer with the same dimensions
            setattr(
                model,
                name,
                DoRALayer(
                    d_out=d_out,
                    d_in=d_in,
                    weight=module.weight.data.clone(),
                    bias=module.bias.data.clone(),
                ),
            )
        else:
            # Recursively apply this function to submodules
            replace_linear_with_dora(module)


def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")


def dora(
    learning_rate=0.001,
    load_in_8bit=False,
    batch_size=64,
    base_model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_input_length=256,
    max_output_length=256,
    epochs=100,
    datasets: List[str] = ["orca_dpo", "ultrafeedback", "openhermes"],
):
    model = torch.compile(
        AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
            load_in_8bit=load_in_8bit,
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, use_fast=True, trust_remote_code=True
    )
    embeddings_model = AutoModelForSentenceEmbedding(model, tokenizer)
    # criterion = nn.MSELoss()
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    labels = [random.randint(0, 1) for _ in range(100)]

    dataset = TextDataset(
        prompts=[
            get_llm_prompt(
                system_instruction="You are a friendly chatbot who answers the user's questions truthfully.",
                user_prompt="What is the color of blood?",
                prompt_format=PromptTemplate.tiny_llama,
            )
        ]
        * 200,
        answers=["red"] * 100 + ["green"] * 100,
        labels=[1] * 100 + [0] * 100,
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_output_length=max_output_length,
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print_model_parameters(model)

    train(
        model,
        tokenizer,
        embeddings_model,
        criterion,
        optimizer,
        data_loader,
        max_input_length,
        max_output_length,
        epochs=epochs,
    )

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        inputs, responses, targets = next(iter(data_loader))
        predictions = model(inputs)
        loss = criterion(predictions, responses, targets)
        print(f"Final Evaluation Loss: {loss.item()}")

    replace_linear_with_dora(model)

    print_model_parameters(model)

    # Continue training with the Dora model
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
    )
    print("Continuing training with DoRA layers...")
    train(model, criterion, optimizer, data_loader, epochs=5)  # Continue training

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        inputs, targets = next(iter(data_loader))
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        print(f"Final (DoRA) Evaluation Loss: {loss.item()}")
