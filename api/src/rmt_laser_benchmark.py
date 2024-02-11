import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import gptq_data_utils
from tqdm import tqdm
import numpy as np
import gc

class ModelModifier:
    def __init__(
        self,
        model_name,
        seqlen=64,
    ):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map={"": 0}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.layer_snr = {}
        self.modified_layers = set()
        self.original_weights = {}
        self.seqlen = seqlen

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
                snr = signal / noise if noise != 0 else float('inf')
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
                mp_threshold_full_iqr = self.marchenko_pastur_threshold(sigma_estimated_full_iqr, n, m)

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
        threshold = sigma * np.sqrt((1 + np.sqrt(beta))**2)
        return threshold
    
    ## Calculate an estimate of the standard deviation of the singular values based on Inter Quantile Range

    @staticmethod
    def estimate_sigma_with_full_iqr(S):
        q75 = torch.quantile(S, 0.75)
        q25 = torch.quantile(S, 0.25)
        iqr = q75 - q25
        sigma_estimated = iqr / 1.349 ## 0.6745 * sigma is the expected range between the quantiles (Q1 and Q3)
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

    def calculate_cosine_similarity(self, expected_answer, model_response):
        # Tokenize the texts
        expected_answer_tokens = self.tokenizer(expected_answer, return_tensors='pt')
        model_response_tokens = self.tokenizer(model_response, return_tensors='pt')

        # Get the embeddings
        with torch.no_grad():
            expected_answer_embeddings = self.model(**expected_answer_tokens).last_hidden_state
            model_response_embeddings = self.model(**model_response_tokens).last_hidden_state

        # Calculate the cosine similarity
        cosine_similarity = torch.nn.functional.cosine_similarity(
            expected_answer_embeddings.mean(dim=1),
            model_response_embeddings.mean(dim=1)
        )

        return cosine_similarity.item()

    def calculate_perplexity(self, prompt_input, expected_answer):
        # Encode the input and expected answer
        inputs = self.tokenizer.encode(prompt_input, return_tensors='pt')
        labels = self.tokenizer.encode(expected_answer, return_tensors='pt')

        # Calculate the loss
        with torch.no_grad():
            outputs = self.model(inputs, labels=labels)
            loss = outputs.loss

        # Calculate the perplexity
        perplexity = torch.exp(loss)

        return perplexity.item()

    def calculate_model_performance(self, inputs, expected_answers):
        performance_scores = []
        perplexities = []

        for prompt_input, expected_answer in zip(inputs, expected_answers):
            # Generate the model's output
            model_output = self.model.generate(prompt_input)

            # Calculate the cosine similarity and perplexity
            score = self.calculate_cosine_similarity(model_output, expected_answer)
            perplexity = self.calculate_perplexity(prompt_input, expected_answer)

            performance_scores.append(score)
            perplexities.append(perplexity)

        return performance_scores, perplexities
    
    def assess_layers_snr(self, layer_types, layer_numbers):
        for name, _ in self.model.named_modules():
            for layer_number in layer_numbers:
                for layer_type in layer_types:
                    if layer_type in name and str(layer_number) in name:
                        layer_name = f"{layer_type}+{layer_number}"
                        print("*"*50, flush=True)
                        print(f"Calculating Signal to Noise Ratio at layer {layer_name}", flush=True)
                        snr = self.calculate_snr_for_layer(layer_type, layer_number)
                        self.layer_snr[layer_name] = snr
                        print(f"Signal to Noise Ratio at layer {layer_name} = {snr}", flush=True)
                        print("*"*50, flush=True)

    def select_layers_for_modification(self, k):
        sorted_layers = sorted(self.layer_snr.items(), key=lambda x: x[1], reverse=False)
        return [layer[0] for layer in sorted_layers[:k]]
    
    def test_and_modify_layers(self, candidate_layers):
        initial_perplexity = self.calculate_model_performance(datasets=['wikitext2', 'ptb', 'c4'], seqlen=128)
        print(f"Initial Model Perplexity: {initial_perplexity}")

        for layer in candidate_layers:
            # Modify the layer
            layer_type = layer.split("+")[0]
            layer_number = layer.split("+")[1]
            self.update_model_reduce_layer(layer_type=layer_type,layer_number=layer_number)
            
            # Test the model's performance
            new_perplexity = self.calculate_model_performance(datasets=['wikitext2', 'ptb', 'c4'], seqlen=128)
            print(f"Tested Model Perplexity after modifying {layer}: {new_perplexity}")

            # If the perplexity does not improve significantly, revert the change
            if new_perplexity > initial_perplexity:
                self.restore_model_original_layer(layer_type=layer_type,layer_number=layer_number)
                print(f"Reverted changes in {layer} due to lack of improvement.", flush=True)
            else:
                initial_perplexity = new_perplexity
                print(f"Modification kept for {layer}. New baseline perplexity: {initial_perplexity}", flush=True)


    def save_model(self, save_dir):

        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
