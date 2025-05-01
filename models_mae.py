# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed

from methods import *

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, 
                 is_distill_token=False, is_bootstrapping=False, bootstrap_method='Last_layer',
                 feature_layers=None):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        if is_bootstrapping:
            print(bootstrap_method)
            assert bootstrap_method in ['Last_layer', 'Fixed_layer_fusion', 'Adaptive_layer_fusion', 'Cross_layer_fusion', \
                                        'Gated_fusion_dynamic', 'Cross_layer_self_attention', 'Cross_layer_cross_attention'], \
                    'bootstrap_method must be one of [Last_layer, Fixed_layer_fusion, Adaptive_layer_fusion, Cross_layer_fusion,' \
                    ' Gated_fusion_dynamic, Cross_layer_self_attention and Cross_layer_cross_attention]'
            if bootstrap_method != 'Last_layer':
                assert feature_layers is not None, 'feature_layers must be specified for Hierarchical layers bootstrap'

        self.feature_layers = feature_layers
        self.is_bootstrapping = is_bootstrapping
        self.bootstrap_method = bootstrap_method

        self.is_distill_token = is_distill_token
        if is_distill_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.distill_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        else:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if is_distill_token:
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        else:
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True, distill_token=self.is_distill_token)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True, distill_token=self.is_distill_token)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        if self.is_distill_token:
            torch.nn.init.normal_(self.distill_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, bootstrapping = False, method_class = None):
        # embed patches
        x = self.patch_embed(x)

        if self.is_distill_token:
            # add pos embed w/o cls token
            x = x + self.pos_embed[:, 1:-1, :]

            # masking: length -> length * mask_ratio
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

            # append cls token
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            distill_token = self.distill_token + self.pos_embed[:, -1:, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            distill_tokens = distill_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x, distill_tokens), dim=1)
        else:
            # add pos embed w/o cls token
            x = x + self.pos_embed[:, 1:, :]

            # masking: length -> length * mask_ratio
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

            # append cls token
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        if bootstrapping:
            if self.bootstrap_method == 'Last_layer':
                for blk in self.blocks:
                    x = blk(x)
            elif self.bootstrap_method == 'Fixed_layer_fusion':
                assert method_class is not None, 'method_class must be specified for Fixed_layer_fusion'
                layer_outputs = []
                for index, blk in enumerate(self.blocks):
                    x = blk(x)
                    if (index + 1) in self.feature_layers:
                        layer_outputs.append(x)
                        # features += x
                # x = features / len(self.feature_layers)   # 相当于均匀分配各层之间的权重
                x = method_class(layer_outputs)
            elif self.bootstrap_method == 'Adaptive_layer_fusion':
                assert method_class is not None, 'method_class must be specified for Adaptive_layer_fusion'
                layer_outputs = []
                for index, blk in enumerate(self.blocks):
                    x = blk(x)
                    if (index + 1) in self.feature_layers:
                        layer_outputs.append(x)
                x = method_class(layer_outputs)
            elif self.bootstrap_method == 'Cross_layer_fusion':
                assert method_class is not None, 'method_class must be specified for Cross_layer_fusion'
                # print("Cross_layer_fusion")
                layer_outputs = []
                for index, blk in enumerate(self.blocks):
                    x = blk(x)
                    # print(x.shape)
                    if (index + 1) in self.feature_layers:
                        layer_outputs.append(x)
                x = method_class(layer_outputs)
            elif self.bootstrap_method == 'Gated_fusion_dynamic':
                assert method_class is not None, 'method_class must be specified for Gated_fusion_dynamic'
                # print("Gated_fusion_dynamic")
                layer_outputs = []
                for index, blk in enumerate(self.blocks):
                    x = blk(x)
                    # print(x.shape)
                    if (index + 1) in self.feature_layers:
                        layer_outputs.append(x)
                x = method_class(layer_outputs)
            elif self.bootstrap_method == 'Cross_layer_self_attention':
                assert method_class is not None, 'method_class must be specified for Cross_layer_self_attention'
                layer_outputs = []
                for index, blk in enumerate(self.blocks):
                    x = blk(x)
                    # print(x.shape)
                    if (index + 1) in self.feature_layers:
                        layer_outputs.append(x)
                x = method_class(layer_outputs)
            elif self.bootstrap_method == 'Cross_layer_cross_attention':
                assert method_class is not None, 'method_class must be specified for Cross_layer_cross_attention'
                layer_outputs = []
                for index, blk in enumerate(self.blocks):
                    x = blk(x)
                    # print(x.shape)
                    if (index + 1) in self.feature_layers:
                        layer_outputs.append(x)
                x = method_class(layer_outputs)
            else:
                raise NotImplementedError(f"Unknown bootstrap method: {self.bootstrap_method}")
        else:
            for blk in self.blocks:
                x = blk(x)
        
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        if self.is_distill_token:
            # append mask tokens to sequence
            # 相当于没mask的部分直接使用原图，最后计算loss的时候只计算mask的部分
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 2 - x.shape[1], 1) # +2 for cls and distill token
            x_ = torch.cat([x[:, 1:-1, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_, x[:, -1:, :]], dim=1)  # append cls token
        else:
            # append mask tokens to sequence
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        if self.is_distill_token:
            # remove cls and distill token
            x = x[:, 1:-1, :]
        else:
            # remove cls token
            x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask, last_model=None, method_class=None):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        # print(self.is_bootstrapping, (last_model==None))
        if self.is_bootstrapping and last_model is not None:
            # print("Bootstrapping")
            target, _, _ = last_model.forward_encoder(imgs, mask_ratio=0.0, bootstrapping=True, method_class=method_class)
            if self.is_distill_token:
                target = nn.functional.normalize(target[:, 1:-1, :], dim=-1)
            else:
                target = nn.functional.normalize(target[:, 1:, :], dim=-1)

            pred, _, _ = last_model.forward_encoder(self.unpatchify(pred), mask_ratio=0.0, bootstrapping=True, method_class=method_class)
            if self.is_distill_token:
                pred = nn.functional.normalize(pred[:, 1:-1, :], dim=-1)
            else:
                pred = nn.functional.normalize(pred[:, 1:, :], dim=-1)
                
            # print(pred.shape, target.shape)
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
            return loss
        else:
            # print("Original")
            target = self.patchify(imgs)
            if self.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.e-6)**.5

            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
            return loss

    def forward(self, imgs, mask_ratio=0.75, last_model=None, method_class=None):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask, last_model, method_class)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_tiny_patch4_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        # TODO: 这里embed_dim之后调调
        img_size=32, patch_size=4, embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_deit_tiny_patch4_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        # TODO: 这里embed_dim之后调调
        img_size=32, patch_size=4, embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), is_distill_token=True, **kwargs)
    return model

# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_tiny_patch4 = mae_vit_tiny_patch4_dec512d8b  # decoder: 512 dim, 8 blocks
mae_deit_tiny_patch4 = mae_deit_tiny_patch4_dec512d8b  # decoder: 512 dim, 8 blocks
