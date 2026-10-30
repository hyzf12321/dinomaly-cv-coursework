from functools import partial

from . import runtime
import torch
from torch import nn
from dinov2.models import vision_transformer as dino
from dinov1.utils import trunc_normal_
from models.uad import ViTill
from models.vision_transformer import Block, bMlp, LinearAttention2, Attention
from optimizers import StableAdamW
from utils import global_cosine, global_cosine_hm_percent, WarmCosineScheduler


def build(cfg, encoder_path=None, random_encoder=False):
    """Same register encoder and initialization as upstream, without auto-downloads."""
    encoder = getattr(dino, 'vit_' + cfg['encoder'])(
        patch_size=14, img_size=518, block_chunks=0, init_values=1e-8,
        num_register_tokens=4, interpolate_antialias=False, interpolate_offset=0.1)
    if encoder_path is not None:
        encoder.load_state_dict(torch.load(str(encoder_path), map_location='cpu'), strict=True)
    elif not random_encoder:
        raise ValueError('Training requires a local pretrained encoder file')
    encoder.requires_grad_(False)
    dim, heads = (768, 12) if cfg['encoder'] == 'base' else (384, 6)
    bottleneck = nn.ModuleList([bMlp(dim, dim * 4, dim, drop=cfg['noise_dropout'])])
    attention = LinearAttention2 if cfg['attention'] == 'linear' else Attention
    decoder = nn.ModuleList([Block(dim=dim, num_heads=heads, mlp_ratio=4., qkv_bias=True,
                                  norm_layer=partial(nn.LayerNorm, eps=1e-8), attn=attention)
                             for _ in range(8)])
    groups = [[i] for i in range(8)] if cfg['supervision'] == 'dense' else [[0, 1, 2, 3], [4, 5, 6, 7]]
    model = ViTill(encoder, bottleneck, decoder, target_layers=list(range(2, 10)),
                  mask_neighbor_size=0, fuse_layer_encoder=groups, fuse_layer_decoder=groups)
    for module in nn.ModuleList([bottleneck, decoder]).modules():
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.01, a=-0.03, b=0.03)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    return model


def parameters(model):
    return list(model.bottleneck.parameters()) + list(model.decoder.parameters())


def optimizer_scheduler(model, cfg):
    optimizer = StableAdamW(parameters(model), lr=cfg['lr'], betas=(0.9, 0.999),
                            weight_decay=cfg['weight_decay'], amsgrad=True, eps=1e-10)
    scheduler = WarmCosineScheduler(optimizer, base_value=cfg['lr'], final_value=cfg['final_lr'],
                                    total_iters=cfg['steps'], warmup_iters=cfg['warmup_steps'])
    return optimizer, scheduler


def loss_value(en, de, cfg, step):
    if cfg['loss'] == 'cosine':
        return global_cosine(en, de)
    return global_cosine_hm_percent(en, de, p=min(0.9 * step / 1000, 0.9), factor=0.1)
