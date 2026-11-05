"""CrossModalMultipanelVQA (report 1) with the report-2 changes as switches.

* answer-type mask: logits outside the valid labels of a question (yes/no, or the options
  a..e that the question actually lists) are removed before softmax/argmax;
* image size: ViT runs at any multiple of 16 with interpolated position embeddings;
* spotlight: the text->patch cross-attention map is returned so the trainer can add a
  region-alignment loss towards the target panel;
* blind: the image tokens are zeroed, giving a same-architecture question-only ablation.
"""
import torch
from torch import nn
from transformers import AutoModel, RobertaConfig, RobertaModel, ViTConfig, ViTModel

NEG = -1e4  # safe "minus infinity" under fp16/bf16 autocast


class CrossModalMultipanelVQA(nn.Module):
    def __init__(self, n_labels, vision="google/vit-base-patch16-224", text="roberta-base",
                 image_size=224, heads=8, dropout=0.1, blind=False, tiny=False):
        super().__init__()
        if tiny:  # random 1-layer encoders: pipeline smoke tests only, no weight download
            heads = 2
            self.vision = ViTModel(ViTConfig(hidden_size=64, num_hidden_layers=1, num_attention_heads=2,
                                             intermediate_size=128, image_size=224), add_pooling_layer=False)
            self.text = RobertaModel(RobertaConfig(vocab_size=50265, hidden_size=64, num_hidden_layers=1,
                                                   num_attention_heads=2, intermediate_size=128),
                                     add_pooling_layer=False)
        else:
            self.vision = ViTModel.from_pretrained(vision, add_pooling_layer=False)
            self.text = AutoModel.from_pretrained(text, add_pooling_layer=False)
        dim = self.vision.config.hidden_size
        if self.text.config.hidden_size != dim:
            raise ValueError("Vision and text hidden sizes must match")
        self.image_size, self.blind = image_size, blind
        self.grid = image_size // self.vision.config.patch_size
        # Learnable 2D spatial embedding (row + column) added to patch tokens.
        self.row = nn.Parameter(torch.zeros(self.grid, dim))
        self.col = nn.Parameter(torch.zeros(self.grid, dim))
        nn.init.trunc_normal_(self.row, std=0.02)
        nn.init.trunc_normal_(self.col, std=0.02)
        self.cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(2 * dim, dim), nn.LayerNorm(dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(dim, n_labels))

    def backbone_parameters(self):
        return list(self.vision.parameters()) + list(self.text.parameters())

    def head_parameters(self):
        backbone = {id(p) for p in self.backbone_parameters()}
        return [p for p in self.parameters() if id(p) not in backbone]

    def forward(self, pixel_values, input_ids, attention_mask, label_mask=None):
        vis = self.vision(pixel_values=pixel_values,
                          interpolate_pos_encoding=self.image_size != 224).last_hidden_state
        if self.blind:
            vis = torch.zeros_like(vis)
        cls_img, patches = vis[:, 0], vis[:, 1:]
        spatial = (self.row[:, None, :] + self.col[None, :, :]).reshape(1, -1, patches.size(-1))
        patches = patches + spatial.to(patches.dtype)
        txt = self.text(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        query = txt[:, :1]  # <s> token as the question summary
        grounded, attn = self.cross(query, patches, patches, need_weights=True, average_attn_weights=True)
        grounded = self.norm(grounded[:, 0] + query[:, 0])
        logits = self.head(torch.cat([grounded, cls_img], dim=-1)).float()
        if label_mask is not None:
            logits = logits.masked_fill(~label_mask, NEG)
        return logits, attn[:, 0]  # attn: (batch, n_patches)


def spotlight_loss(attn, target_mask, eps=1e-6):
    """-log of the attention mass that falls inside the target panel (rows with a target only)."""
    has = target_mask.sum(-1) > 0
    if not has.any():
        return attn.new_zeros(())
    mass = (attn[has].float() * target_mask[has].float()).sum(-1)
    return -(mass + eps).log().mean()
