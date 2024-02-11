import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import gptq_data_utils
from tqdm import tqdm
import numpy as np
import gc
from icecream import ic

from src.utils.load_benchmark_dataset import get_benchmark_data

from src.utils.assets import PromptTemplate
from src.utils.prompt_template import get_llm_prompt
import torch.nn.functional as F

from src.AutoModelForSentenceEmbedding import (
    AutoModelForSentenceEmbedding,
    get_loss,
    get_cosine_embeddings,
)
import time

class ModelModifier:
    def __init__(
        self,
        model_name,
        prompt_template: PromptTemplate = PromptTemplate.chatml,
        input_length=1024,
        output_length=1024,
    ):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map={"": 0}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, use_fast=True
        )
        self.layer_snr = {}
        self.modified_layers = set()
        self.original_weights = {}
        self.input_length = input_length
        self.output_length = output_length
        self.embeddings_model = AutoModelForSentenceEmbedding(
            model_name, self.tokenizer
        )

    def calculate_snr_for_layer(self, layer_type, layer_number):
        for name, module in self.model.named_modules():
            if layer_type in name and str(layer_number) in name:
                weights = module.weight.double()
                S = torch.linalg.svdvals(weights)
                weights = weights.detach().cpu()
                S = S.detach().cpu()
                sigma_estimated = self.estimate_sigma_with_full_iqr(S)
                n, m = weights.shape
                mp_threshold = self.marchenko_pastur_threshold(sigma_estimated, n, m)

                signal = S[S > mp_threshold].sum()
                noise = S[S <= mp_threshold].sum()
                snr = signal / noise if noise != 0 else float("inf")
                del S, weights
                torch.cuda.empty_cache()  # Clear PyTorch's CUDA memory cache
                gc.collect()
                return snr

    def update_model_reduce_layer(self, layer_type, layer_number):
        layer_id = f"{layer_type}+{layer_number}"
        if layer_id in self.modified_layers:
            print(f"Layer {layer_id} has already been modified. Skipping.")
            return False

        for name, module in self.model.named_modules():
            if layer_type in name and str(layer_number) in name:
                print(f"Reconstructing layer: {name}")
                original_dtype = module.weight.dtype
                self.original_weights[name] = module.weight.detach().clone()
                weights = module.weight.double()
                U, S, V = torch.linalg.svd(weights, full_matrices=False)

                # Estimate sigma using the full IQR method
                sigma_estimated_full_iqr = self.estimate_sigma_with_full_iqr(S)

                # Calculate Marchenko-Pastur threshold
                n, m = weights.shape
                mp_threshold_full_iqr = self.marchenko_pastur_threshold(
                    sigma_estimated_full_iqr, n, m
                )

                # Retain only the singular values above the MP threshold
                S_reduced = torch.zeros_like(S)
                k = (S > mp_threshold_full_iqr).sum().item()
                S_reduced[:k] = S[:k]
                print(f"Reduced from {S.shape} to {k}")

                # Reconstruct the matrix using the thresholded singular values
                reconstructed_weights = U @ torch.diag(S_reduced) @ V
                reconstructed_weights = reconstructed_weights.to(original_dtype)
                module.weight = torch.nn.Parameter(reconstructed_weights)
                self.modified_layers.add(layer_id)
                return True

    @staticmethod
    def marchenko_pastur_threshold(sigma, n, m):
        beta = n / m if n < m else m / n
        threshold = sigma * np.sqrt((1 + np.sqrt(beta)) ** 2)
        return threshold

    ## Calculate an estimate of the standard deviation of the singular values based on Inter Quantile Range

    @staticmethod
    def estimate_sigma_with_full_iqr(S):
        q75 = torch.quantile(S, 0.75)
        q25 = torch.quantile(S, 0.25)
        iqr = q75 - q25
        sigma_estimated = (
            iqr / 1.349
        )  ## 0.6745 * sigma is the expected range between the quantiles (Q1 and Q3)
        return sigma_estimated

    def restore_model_original_layer(self, layer_type, layer_number):
        layer_id = f"{layer_type}+{layer_number}"
        for name, module in self.model.named_modules():
            if layer_type in name and layer_number in name:
                if name in self.original_weights:
                    module.weight = torch.nn.Parameter(self.original_weights[name])
                    print(f"Restored original weights for layer: {name}", flush=True)
                    if layer_id in self.modified_layers:
                        self.modified_layers.remove(layer_id)
                        break
                else:
                    print(f"No original weights saved for layer: {name}", flush=True)
        return

    def calculate_cosine_similarity(self, embeddings_vec_1, embeddings_vec_2):

        return F.cosine_similarity(embeddings_vec_1, embeddings_vec_2, dim=1)

    def calculate_perplexity(self, inputs, labels):
        # Encode the input and expected answer

        # Calculate the loss
        with torch.no_grad():
            outputs = self.model(inputs, labels=labels)
            loss = outputs.loss

        # Calculate the perplexity
        perplexity = torch.exp(loss)

        return perplexity.item()

    def calculate_model_performance(
        self,
        datasets=["orca_dpo", "ultrafeedback"],  # "openhermes"
        n_samples=256,
        input_length=1024,
        output_length=1024,
    ):
        performance_scores = []
        rejection_scores = []
        loss_accum = 0.0
        loss_fct = torch.nn.CrossEntropyLoss().cuda()
        model = self.model
        tokenizer = self.tokenizer
        embeddings_model = self.embeddings_model
        model.to("cuda")
        for dataset in datasets:
            benchmark_dataset = get_benchmark_data(
                dataset, n_samples, input_length, output_length
            )

            for sample in benchmark_dataset.data:
                prompt = get_llm_prompt(sample.instruction, sample.prompt)
                prompt_enc = tokenizer([prompt], return_tensors="pt")
                prompt_enc.to("cuda")
                expected_answer = sample.chosen
                expected_answer_enc = tokenizer(
                    [expected_answer],
                    return_tensors="pt",
                    padding="max_length",
                    max_length=self.output_length,
                )
                expected_answer_enc.to("cuda")
                expected_answer_embs = embeddings_model(**expected_answer_enc)
                # ic(expected_answer_embs)
                rejected_answer = sample.rejected
                rejected_answer_enc = tokenizer(
                    [rejected_answer],
                    return_tensors="pt",
                    padding="max_length",
                    max_length=self.output_length,
                )
                rejected_answer_enc.to("cuda")
                rejected_answer_embs = embeddings_model(**rejected_answer_enc)

                # Generate the model's output
                # model_output = model(
                #     prompt_enc.input_ids,
                #     use_cache=False,
                #     output_hidden_states=False,
                #     output_attentions=False,
                # )[0]
                model_output = model.generate(
                    **prompt_enc,
                    max_new_tokens=self.output_length,
                    use_cache=False,
                    output_hidden_states=False,
                    output_attentions=False,
                )
                ic(model_output)
                output_ids = model_output[0].argmax(dim=-1)
                output_string = tokenizer.decode(output_ids, skip_special_tokens=True)
                ic(output_string)
                time.sleep(60)
                model_output_embs = embeddings_model(**model_output)
                cosine_similarity_gain = get_cosine_embeddings(
                    model_output_embs, expected_answer_embs
                )
                cosine_similarity_loss = get_cosine_embeddings(
                    model_output_embs, rejected_answer_embs
                )
                ic(cosine_similarity_gain, cosine_similarity_loss)
                # rejection_score = self.calculate_cosine_similarity(
                #     rejected_answer_enc, model_output
                # )
                # performance_scores.append(score)
                # rejection_scores.append(rejection_score)
                # Calculate the perplexity
                # loss_accum += 1
                # loss_accum += loss_fct(model_output, expected_answer_enc)
                # model_output.to("cuda")

                # ic(model_output)
                # # expected_answer_enc.cuda()
                # ic(type(expected_answer_enc), expected_answer_enc)
                # loss = loss_fct(model_output, expected_answer_enc.input_ids)
                # loss_accum += loss.item()

        avg_loss = loss_accum / (n_samples * len(datasets))
        average_perplexity = torch.exp(torch.tensor(avg_loss)).item()
        performance = -average_perplexity
        return performance

    def assess_layers_snr(self, layer_types, layer_numbers):
        for name, _ in self.model.named_modules():
            for layer_number in layer_numbers:
                for layer_type in layer_types:
                    if layer_type in name and str(layer_number) in name:
                        layer_name = f"{layer_type}+{layer_number}"
                        print("*" * 50, flush=True)
                        print(
                            f"Calculating Signal to Noise Ratio at layer {layer_name}",
                            flush=True,
                        )
                        snr = self.calculate_snr_for_layer(layer_type, layer_number)
                        self.layer_snr[layer_name] = snr
                        print(
                            f"Signal to Noise Ratio at layer {layer_name} = {snr}",
                            flush=True,
                        )
                        print("*" * 50, flush=True)

    def select_layers_for_modification(self, k):
        sorted_layers = sorted(
            self.layer_snr.items(), key=lambda x: x[1], reverse=False
        )
        return [layer[0] for layer in sorted_layers[:k]]

    def test_and_modify_layers(self, candidate_layers):
        initial_performance = self.calculate_model_performance()

        print(f"Initial Model Performance: {initial_performance}")

        for layer in candidate_layers:
            # Modify the layer
            layer_type = layer.split("+")[0]
            layer_number = layer.split("+")[1]
            self.update_model_reduce_layer(
                layer_type=layer_type, layer_number=layer_number
            )

            # Test the model's performance
            new_performance = self.calculate_model_performance()
            print(
                f"Tested Model Performance after modifying {layer}: {new_performance}"
            )

            # If the perplexity does not improve significantly, revert the change
            if new_performance <= initial_performance:
                self.restore_model_original_layer(
                    layer_type=layer_type, layer_number=layer_number
                )
                print(
                    f"Reverted changes in {layer} due to lack of improvement.",
                    flush=True,
                )
            else:
                initial_performance = new_performance
                print(
                    f"Modification kept for {layer}. New baseline perplexity: {initial_performance}",
                    flush=True,
                )

    def save_model(self, save_dir):

        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
