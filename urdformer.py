import math
from typing import Optional, Tuple, Type

import numpy as np
import PIL
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.ops import box_convert
from vit import vit_b16, vit_l16, vit_s16, vit_scratch16, vit_scratch_base16


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=384, temperature=10):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.scale = 2 * math.pi
        self.fcx = nn.Linear(16, self.num_pos_feats)
        self.fcy = nn.Linear(16, self.num_pos_feats)
        self.fcw = nn.Linear(16, self.num_pos_feats)
        self.fch = nn.Linear(16, self.num_pos_feats)


    def forward(self, x):
        x_embed = x[:, :, 0]
        y_embed = x[:, :, 1]
        w_embed = x[:, :, 2]
        h_embed = x[:, :, 3]

        dim_t = torch.arange(16, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.floor_divide(dim_t, 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_w = w_embed[:, :, None] / dim_t
        pos_h = h_embed[:, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=2).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos_x = self.fcx(pos_x)
        pos_y = self.fcx(pos_y)
        pos_w = self.fcx(pos_w)
        pos_h = self.fcx(pos_h)

        pos = pos_y + pos_x + pos_h + pos_w
        return pos

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x



class URDFormer(pl.LightningModule):
    def __init__(
        self,
        img_size: int = 1024,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        num_relations: int= 2,
        num_roots: int = 5,
        backbone_mode: int = 0,
        mesh_num: int=8,
        part_mesh_num: int=9,
        conn_loss_weight: float = 5.0,
        base_loss_weight: float = 1.0,
        positions_loss_weight: float = 1.0,
        mesh_loss_weight: float = 1.0,
        optimizer_cfg: dict = {"lr": 1e-4, "weight_decay": 0.05, "eps": 1e-8, "betas": (0.9, 0.999)},
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.roi_size = 14
        hidden1 = 64
        e_dim = 512
        self.vit_hidden_size = 384
        self.num_roots = num_roots
        self.num_relations = num_relations

        self.optimizer_cfg = optimizer_cfg
        self.conn_loss_weight = conn_loss_weight
        self.base_loss_weight = base_loss_weight
        self.positions_loss_weight = positions_loss_weight
        self.mesh_loss_weight = mesh_loss_weight
        ############# use pretrained MAE ########
        self.img_backbone, gap_dim = vit_s16(pretrained="backbones/mae_pretrain_hoi_vit_small.pth", img_size=224)

        self.pe_layer = PositionEmbeddingSine(e_dim)

        self.roi_pool = torchvision.ops.RoIAlign(output_size=(self.roi_size, self.roi_size), spatial_scale=1 / 16, sampling_ratio=2)
        self.fc1 = nn.Sequential(
                    nn.Linear(self.vit_hidden_size, 512), # 768 for vit, 1024 for resnet with third-to-last layers
                    nn.LeakyReLU(),
                    nn.Linear(512, 256),
                    nn.LeakyReLU(),
                    nn.Linear(256, hidden1),
                )
        self.fc2 = nn.Linear(self.roi_size*self.roi_size*hidden1, e_dim)
        self.img_fc1 = nn.Sequential(
                    nn.Linear(self.vit_hidden_size, 512),
                    nn.LeakyReLU(),
                    nn.Linear(512, hidden1),
                )
        self.img_fc2 = nn.Sequential(
            nn.Linear(3136, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
        )

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bbox_fc = nn.Sequential(
            nn.Linear(4, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, e_dim),
        )

        self.position_head_x = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 13),
        )
        self.position_head_y = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 13),
        )
        self.position_head_z = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 13),
        )

        self.position_end_x = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 13),
        )

        self.position_end_y = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 13),
        )
        self.position_end_z = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 13),
        )

        self.parent = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 32*num_relations),
        )
        self.child = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 32*num_relations),
        )
        self.embedding = nn.Embedding(num_roots, 512)


        self.mesh_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, part_mesh_num),  #'none', 'drawer', 'doorL', 'doorR', 'handle', 'knob', 'washer_door', 'doorD', 'oven_door', "doorU"
        )

        self.base_mesh_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, mesh_num),
        )

        # Initialize absolute positional embedding with pretrain image size.
        self.norm_layer = norm_layer(e_dim)
        self.norm_layer1 = norm_layer(512)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=e_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            self.blocks.append(block)

        self.init_weights()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)

        self.apply(_init_weights)

        # Special initialization for attention layers in transformer blocks
        for block in self.blocks:
            nn.init.normal_(block.qkv.weight, std=0.02)
            nn.init.normal_(block.attn.in_proj_weight, std=0.02)
            nn.init.normal_(block.attn.out_proj.weight, std=0.02)
            nn.init.constant_(block.attn.in_proj_bias, 0)
            nn.init.constant_(block.attn.out_proj.bias, 0)

    def forward(self, img, bbox, mask, ablation_mode=0):
        img_feature = self.img_backbone(img).view(-1, 14, 14, self.vit_hidden_size).permute(0, 3, 1, 2)
        bbox1 = (bbox * img.shape[-1]).int()
        bbox1[:, :, :, 2] = bbox1[:, :, :,  2] + bbox1[:, :, :,  0]
        bbox1[:, :, :, 3] = bbox1[:, :, :, 3] + bbox1[:, :, :, 1]

        bbox1 = torch.floor_divide(bbox1, img.shape[-1] / 14)
        bbox_list = torch.split(bbox1.squeeze(1), 1, dim=0)
        bbox_list = [tensor.squeeze(0).float() for tensor in bbox_list]

        pooled_features = self.roi_pool(img_feature, bbox_list).view(img_feature.shape[0], -1, img_feature.shape[1], self.roi_size, self.roi_size)
        if ablation_mode>=3:
            pooled_features = mask.squeeze(1).unsqueeze(2)*pooled_features

        masked_features = self.fc1(pooled_features.permute(0, 1, 3, 4, 2))

        # Flatten the pooled features and feed them to a linear layer
        token_features = torch.flatten(masked_features, start_dim=2)
        token_features = self.fc2(token_features)

        # pos_embedding as xyxy of the original bbox
        pos_bbox = self.bbox_fc(bbox.squeeze(1).float())
        x = self.norm_layer(token_features) + self.norm_layer(pos_bbox)

        for blk in self.blocks:
            x = blk(x)

        img_feature = self.img_fc1(self.max_pool(img_feature).permute(0, 2, 3, 1).view(img.shape[0], -1, self.vit_hidden_size))
        img_feature = self.img_fc2(img_feature.view(img.shape[0], -1))
        base_mesh_type = self.base_mesh_head(img_feature)

        x = self.norm_layer(x)
        position_x = self.position_head_x(x)
        position_y = self.position_head_y(x)
        position_z = self.position_head_z(x)

        position_end_x = self.position_end_x(x)
        position_end_y = self.position_end_y(x)
        position_end_z = self.position_end_z(x)

        mesh_type = self.mesh_head(x)

        B, max_mask, feature_size = x.shape

        # predicting relationship
        # concatenate the embedding of roots into x.
        roots_embeddings = self.embedding(torch.arange(self.num_roots, device="cuda").unsqueeze(0).expand(B, -1))
        x = torch.cat([self.norm_layer1(roots_embeddings), self.norm_layer1(x)], dim=1)
        parent_embedding = self.parent(x).reshape(B, max_mask+self.num_roots, self.num_relations, -1).permute([0, 2, 1, 3])
        children_embedding = self.child(x).reshape(B, max_mask+self.num_roots, self.num_relations, -1).permute([0, 2, 1, 3])

        cls_pred = parent_embedding @ torch.transpose(children_embedding, 2, 3) / 32**0.5

        return position_x, position_y, position_z, position_end_x, position_end_y, position_end_z, mesh_type, cls_pred.permute([0, 2, 3, 1]), base_mesh_type

    def training_step(self, batch, batch_idx):
        img, bbox, mask = batch[0]["img"], batch[0]["bbox"], batch[0]["masks"]
        position_x, position_y, position_z, position_end_x, position_end_y, position_end_z, mesh_type, cls_pred, base_mesh_type = self(img, bbox, mask)

        # Compute the losses
        supervision = batch[1]
        positions = supervision["positions"]
        postition_gt_min, position_gt_max = positions
        position_gt_x = postition_gt_min[:, :, 0]
        position_gt_y = postition_gt_min[:, :, 1]
        position_gt_z = postition_gt_min[:, :, 2]
        position_gt_x_end = position_gt_max[:, :, 0]
        position_gt_y_end = position_gt_max[:, :, 1]
        position_gt_z_end = position_gt_max[:, :, 2]
        gt_mesh_type = supervision["mesh_types"]
        gt_connectivity = supervision["connectivity"]
        gt_base_type = supervision["base_type"]

        unpad_masks = batch[2]

        loss_fcn = nn.CrossEntropyLoss(reduction="none")
        loss_scn_reduce = nn.CrossEntropyLoss()
        bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        loss_position_x = loss_fcn(position_x.transpose(1, 2), position_gt_x.long())
        loss_position_y = loss_fcn(position_y.transpose(1, 2), position_gt_y.long())
        loss_position_z = loss_fcn(position_z.transpose(1, 2), position_gt_z.long())

        loss_position_end_x = loss_fcn(position_end_x.transpose(1, 2), position_gt_x_end.long())
        loss_position_end_y = loss_fcn(position_end_y.transpose(1, 2), position_gt_y_end.long())
        loss_position_end_z = loss_fcn(position_end_z.transpose(1, 2), position_gt_z_end.long())

        loss_mesh = loss_fcn(mesh_type.transpose(1, 2), gt_mesh_type.long())

        loss_parent_cls = bce_loss(cls_pred, gt_connectivity.float())

        loss_base = loss_scn_reduce(base_mesh_type, gt_base_type.long())

        # Mask + reduce the losses
        loss_position_x = (loss_position_x * unpad_masks["positions"]).sum() / (unpad_masks["positions"].sum() + 1e-8)
        loss_position_y = (loss_position_y * unpad_masks["positions"]).sum() / (unpad_masks["positions"].sum() + 1e-8)
        loss_position_z = (loss_position_z * unpad_masks["positions"]).sum() / (unpad_masks["positions"].sum() + 1e-8)
        loss_position_end_x = (loss_position_end_x * unpad_masks["positions"]).sum() / (unpad_masks["positions"].sum() + 1e-8)
        loss_position_end_y = (loss_position_end_y * unpad_masks["positions"]).sum() / (unpad_masks["positions"].sum() + 1e-8)
        loss_position_end_z = (loss_position_end_z * unpad_masks["positions"]).sum() / (unpad_masks["positions"].sum() + 1e-8)
        loss_mesh = (loss_mesh * unpad_masks["mesh_types"]).sum() / (unpad_masks["mesh_types"].sum() + 1e-8)
        loss_parent_cls = (loss_parent_cls * unpad_masks["connectivity"]).sum() / (unpad_masks["connectivity"].sum() + 1e-8)

        loss = self.positions_loss_weight * (loss_position_x + loss_position_y + loss_position_z) + \
               self.positions_loss_weight * (loss_position_end_x + loss_position_end_y + loss_position_end_z) + \
               self.mesh_loss_weight * loss_mesh + self.conn_loss_weight * loss_parent_cls + self.base_loss_weight * loss_base

        loss_dict = {
            "loss": loss,
            "loss_position_x": loss_position_x,
            "loss_position_y": loss_position_y,
            "loss_position_z": loss_position_z,
            "loss_position_end_x": loss_position_end_x,
            "loss_position_end_y": loss_position_end_y,
            "loss_position_end_z": loss_position_end_z,
            "loss_mesh": loss_mesh,
            "loss_parent_cls": loss_parent_cls,
            "loss_base": loss_base,
        }
        self.log('train/loss', loss_dict, on_epoch=True, on_step=False, logger=True)

        self.log('train/total_loss', loss, on_epoch=True, on_step=False, logger=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_epoch=True, on_step=False, logger=True)
        return loss_dict

    def validation_step(self, batch, batch_idx):
        loss_dict = self.training_step(batch, batch_idx)
        self.log('val/loss', loss_dict, on_epoch=True, on_step=False, logger=True)

        # Log a single scalar for the scheduler
        self.log('val/total_loss', loss_dict['loss'], on_epoch=True, on_step=False, logger=True)

        return loss_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_cfg)

        constant_epochs = 1
        total_epochs = self.trainer.max_epochs
        cosine_epochs = total_epochs - constant_epochs

        constant_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=constant_epochs)

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-8)

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[constant_scheduler, cosine_scheduler],
            milestones=[constant_epochs]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.

        """
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,  batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

    def forward(self, bbox_features):
        shortcut = bbox_features
        x = self.norm1(bbox_features)
        # self.attention
        B, N, C = bbox_features.shape

        qkv = self.qkv(x).reshape(B, N, 3, -1).permute(2, 0, 1,3)

        q, k, v = qkv.unbind(0)

        x, attn_weights = self.attn(q, k, v)
        x = shortcut+x
        x = x + self.mlp(self.norm2(x))

        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.

        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, N, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, N, -1).permute(0, 2, 1, 3).reshape(B, N, -1)

        x = self.proj(x)
        return x
