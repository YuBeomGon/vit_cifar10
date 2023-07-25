import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union, Tuple
import timm
from timm.layers import DropPath, trunc_normal_, lecun_normal_, PatchDropout
# from timm.models.helpers import named_apply
from functools import partial
import logging
# from collections import OrderedDict

import numpy as np
import math
import logging

_logger = logging.getLogger(__name__)


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    
    def __init__(self, 
                 img_size: Optional[int] = 32,
                 patch_size: int = 8,
                 in_chans: int = 3,
                 embed_dim: int = 192,
                 norm_layer: Optional[Callable] = None,
                 # flatten: bool = True,
                 bias: bool = True,
                 stride: int = 8,
                 roll: bool = False):
        
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.img_size = (img_size, img_size)
        self.grid_size = tuple([ ((s-p) // stride) + 1 for s, p in zip(self.img_size, self.patch_size)])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.roll = roll
        
    def forward(self, x):
        if self.roll:
            h_shift, w_shift = np.random.randint(0,self.patch_size, size=2)
            x = torch.roll(x, shifts=(h_shift, w_shift), dims=(2,3))
        
        B, C, H, W = x.shape
        torch._assert(H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]}).")
        torch._assert(W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]}).")
        
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        x = self.norm(x)
        return x
        

# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.net = nn.Sequential(nn.Linear(in_features, hidden_features),
#                                 act_layer(),
#                                 nn.Linear(hidden_features, out_features),
#                                 nn.Dropout(drop))
        
#     def forward(self, x):
#         x = self.net(x)
#         return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = (bias, bias) 
        drop_probs = (drop, drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, 
                 qkv_bias=False, 
                 qk_norm=False, 
                 v_norm=False, 
                 attn_drop=0, 
                 proj_drop=0, 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads ==0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.v_norm = nn.BatchNorm1d(dim) if v_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        
        self.v_act = nn.GELU() if v_norm else nn.Identity()
                
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        
        if self.v_norm:
            v = v.permute(0,2,1,3).reshape(B,N,C).permute(0,2,1) 
            v = self.v_act(self.v_norm(v))
            v = v.reshape(B, self.num_heads, C //self.num_heads, N).permute(0,1,3,2)
        
        # if self.qk_norm:
        q, k = self.q_norm(q), self.k_norm(k)
            
        attn = ( q @ k.transpose(-2,-1)) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        x = attn @ v
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
# https://arxiv.org/pdf/2103.17239v2.pdf
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, qk_norm=False, v_norm=False, proj_drop=0., attn_drop=0., init_values=None, 
                 drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_layer=Mlp):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, v_norm=v_norm,
                              attn_drop=attn_drop, proj_drop=proj_drop, norm_layer=norm_layer)
        
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drpo_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer 
        pytorch implementation
    """   
    def __init__(self,
                img_size: Union[int, Tuple[int, int]] = 32,
                patch_size: Union[int, Tuple[int, int]] = 8,   
                in_chans: int = 3,
                num_classes: int = 10,  
                embed_dim: int = 192,
                depth: int = 6,
                num_heads: int = 2,
                mlp_ratio: float = 4.,
                qkv_bias: bool = True,
                qk_norm: bool = False,
                v_norm: bool = False,   
                init_values: Optional[float] = None,
                pre_norm: bool = False,
                fc_norm: Optional[bool] = None,
                drop_rate: float = 0.,
                pos_drop_rate: float = 0.,
                patch_drop_rate: float = 0.,
                proj_drop_rate: float = 0.,
                attn_drop_rate: float = 0.,
                drop_path_rate: float = 0.,
                embed_layer: Callable = PatchEmbed,
                norm_layer: Optional[Callable] = None,
                act_layer: Optional[Callable] = None,
                block_fn: Callable = Block,
                mlp_layer: Callable = Mlp,
                checkpoint: str = '',
                patch_stride: int = 8,
                roll: bool = False):    
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            fc_norm: Pre head norm after pool (instead of before)
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
            checkpoint: checkpoint file.
            patch_stride: overraping stride for conv in patch embeding
            roll: rolling in patch embeding
        """        
        
        super().__init__()
        use_fc_norm = fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU    
        
        self.checkpoint = checkpoint
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            stride=patch_stride,
            roll=roll
        )   
        
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        embed_len = num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)   
        
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()     
            
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                v_norm=v_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer
            )
            for i in range(depth)])     
        
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
        
        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()             

        # self.apply(self._init_weights)
        
        # load from pretrained 
        if self.checkpoint:
            self._load_pretrained()

    def _load_pretrained(self):
        print(f'load pretrained from {self.checkpoint}') 
        prev_state_dict = torch.load(self.checkpoint, map_location='cpu')
        # pos emb lenght is not same, need to 2d interpolate
        if prev_state_dict['pos_embed'].shape[1] != self.pos_embed.shape[1]:
            prev_state_dict['pos_embed'] = self.resize_pos_embed(prev_state_dict['pos_embed'], self.pos_embed.shape)
        
        self.load_state_dict(prev_state_dict, strict=False)
        for name, param in self.named_parameters():
            param.requires_grad = True
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)    
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)  
                
    def freeze(self, layers):
        for name, param in self.named_parameters():
            if name.split('.')[0] in layers:
                pass
            else:
                param.requires_grad = False
                
        
    def _pos_embed(self, x):
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
    
    def resize_pos_embed(
            self,
            posemb,
            posemb_new_shape,
            interpolation='bicubic',
            antialias=False,
    ):    
        """ Rescale the grid of position embeddings when loading from state_dict.
        """
        
        _logger.info(f'Resized position embedding: {posemb.shape} to {posemb_new_shape}.')
        
        B, L, C = posemb.shape
        H = W = int(math.sqrt(posemb_new_shape[1] - 1))
        _H = _W = int(math.sqrt(posemb.shape[1] - 1))
        
        cls_posemb, others_posemb = posemb[:,0:1,:], posemb[:,1:,:]
        
        others_posemb = others_posemb.transpose(-1,-2).reshape(B, C, _H, _W)
        
        posemb_new = F.interpolate(others_posemb, (H, W),  mode=interpolation, antialias=antialias)
        posemb_new = posemb_new.reshape(B, C, -1).transpose(-1, -2)
        
        return torch.cat((cls_posemb, posemb_new), dim=1)

    
def vit_small_patch8_32(**kwargs) -> VisionTransformer:
    """ ViT-Tiny (Vit-Ti/8)
    """
    model_args = dict(img_size=32, patch_size=8, embed_dim=384, depth=12, num_heads=4)
    model = VisionTransformer(**dict(model_args, **kwargs))
    
    return model    


# def vit_small_patch8_64(**kwargs) -> VisionTransformer:
#     """ ViT-Tiny (Vit-Ti/8)
#     """
#     model_args = dict(img_size=64, patch_size=8, embed_dim=384, depth=12, num_heads=4)
#     model = VisionTransformer(**dict(model_args, **kwargs))

#     return model    

                 
            
            
        
        
        
            
        
        
    
    
