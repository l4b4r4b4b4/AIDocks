import torch
from torch import nn
from transformers import (
    AutoModel,
)


def get_cosine_embeddings(q1_embs, q2_embs):
    return torch.sum(q1_embs * q2_embs, axis=1)


def get_loss(cosine_score, labels):
    return torch.mean(
        torch.square(
            labels * (1 - cosine_score)
            + torch.clamp((1 - labels) * cosine_score, min=0.0)
        )
    )


class AutoModelForSentenceEmbedding(nn.Module):
    def __init__(self, model_name, tokenizer, normalize=True):
        super(AutoModelForSentenceEmbedding, self).__init__()

        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True
        )  # , load_in_8bit=True, device_map={"":0})
        self.model.to("cuda")
        self.normalize = normalize
        self.tokenizer = tokenizer

    def forward(self, **kwargs):
        model_output = self.model(**kwargs)
        embeddings = self.mean_pooling(model_output, kwargs["attention_mask"])
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)