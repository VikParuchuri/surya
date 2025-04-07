import torch
import torch.nn as nn


class SimpleTokenEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)

    def embed(
        self,
        input_tokens: torch.Tensor,
    ) -> torch.Tensor:
        token_embeds = self.token_embed(input_tokens)
        embeddings = token_embeds
        return embeddings
