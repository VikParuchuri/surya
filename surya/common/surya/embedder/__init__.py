import torch
import torch.nn as nn


class BboxEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        special_token_count = config.special_token_count

        self.w_embed = nn.Embedding(
            config.bbox_size + special_token_count, config.hidden_size
        )
        self.h_embed = nn.Embedding(
            config.bbox_size + special_token_count, config.hidden_size
        )
        self.cx_embed = nn.Embedding(
            config.bbox_size + special_token_count, config.hidden_size
        )
        self.cy_embed = nn.Embedding(
            config.bbox_size + special_token_count, config.hidden_size
        )
        self.xskew_embed = nn.Embedding(
            config.bbox_size + special_token_count, config.hidden_size
        )
        self.yskew_embed = nn.Embedding(
            config.bbox_size + special_token_count, config.hidden_size
        )

        # Corners
        self.x1_embed = nn.Embedding(
            config.bbox_size + special_token_count, config.hidden_size
        )
        self.y1_embed = nn.Embedding(
            config.bbox_size + special_token_count, config.hidden_size
        )
        # x2 + y2 are unused, but kept for compat
        self.x2_embed = nn.Embedding(
            config.bbox_size + special_token_count, config.hidden_size
        )
        self.y2_embed = nn.Embedding(
            config.bbox_size + special_token_count, config.hidden_size
        )
        self.x3_embed = nn.Embedding(
            config.bbox_size + special_token_count, config.hidden_size
        )
        self.y3_embed = nn.Embedding(
            config.bbox_size + special_token_count, config.hidden_size
        )
        # x4 + y4 are unused, but kept for compat
        self.x4_embed = nn.Embedding(
            config.bbox_size + special_token_count, config.hidden_size
        )
        self.y4_embed = nn.Embedding(
            config.bbox_size + special_token_count, config.hidden_size
        )

        self.xspecial_token_count_embed = nn.Embedding(
            config.bbox_size + special_token_count, config.hidden_size
        )
        self.yspecial_token_count_embed = nn.Embedding(
            config.bbox_size + special_token_count, config.hidden_size
        )

        self.config = config

    def forward(self, boxes: torch.LongTensor) -> torch.Tensor:
        cx, cy, w, h, xskew, yskew = boxes.to(torch.long).unbind(dim=-1)
        is_special = (cx == self.config.blank_bbox_token_id).to(torch.long)

        xskew_actual = ((xskew - self.config.bbox_size // 2) / 2).to(torch.long)
        yskew_actual = ((yskew - self.config.bbox_size // 2) / 2).to(torch.long)

        x1 = (cx - w // 2 - xskew_actual).clamp(0, self.config.bbox_size).to(torch.long)
        y1 = (cy - h // 2 - yskew_actual).clamp(0, self.config.bbox_size).to(torch.long)
        x3 = (cx + w // 2 + xskew_actual).clamp(0, self.config.bbox_size).to(torch.long)
        y3 = (cy + h // 2 + yskew_actual).clamp(0, self.config.bbox_size).to(torch.long)

        size_embeds = (
                self.w_embed(w) + self.h_embed(h) + self.cx_embed(cx) + self.cy_embed(cy)
        )
        skew_embeds = self.xskew_embed(xskew) + self.yskew_embed(yskew)
        corner_embeds = (
                self.x1_embed(x1)
                + self.y1_embed(y1)
                + self.x3_embed(x3)
                + self.y3_embed(y3)
        )
        special_embed = self.xspecial_token_count_embed(is_special)
        embedded = size_embeds + skew_embeds + corner_embeds + special_embed

        return embedded / 10  # Normalize by number of embeddings

class SimpleTokenEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.bbox_embed = BboxEmbedding(config)

    def embed(
        self, input_tokens: torch.Tensor, input_bboxes: torch.Tensor
    ) -> torch.Tensor:
        token_embeds = self.token_embed(input_tokens)
        bbox_embeds = self.bbox_embed(input_bboxes)

        return token_embeds + bbox_embeds
