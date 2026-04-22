import os
import torch.nn.functional as F
import torch
import torch.distributed as dist
from timm.models.layers import DropPath
from torch import nn


def load_model(
    version: str, adaptor_names: str = None, force_reload: bool = False, **kwargs
):
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if os.path.isfile(version) or "radio" in version:
        model: nn.Module = torch.hub.load(
            "NVlabs/RADIO",
            "radio_model",
            version=version,
            progress=True,
            adaptor_names=adaptor_names,
            force_reload=force_reload,
            skip_validation=True,
            **kwargs,
        )
    elif version.startswith("dinov2"):
        model = torch.hub.load(
            "facebookresearch/dinov2", version, force_reload=force_reload, **kwargs
        )
    else:
        raise ValueError(f"Unsupported model version: {version}")

    return model


class RadioWrapper(nn.Module):
    def __init__(self, model_version: str = "radio_v2.5-l", is_frozen=True):
        super().__init__()
        if not dist.is_initialized() or dist.get_rank() == 0:
            # Pull the model on rank 0 first.
            model = load_model(model_version)
        if dist.is_initialized():
            dist.barrier()
            if dist.get_rank() > 0:
                # Now pull the model from cache on other ranks.
                model = load_model(model_version)
        if is_frozen:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
        if model_version == "radio_v2.5-g":
            architecture = model.model.inner
        else:
            architecture = model.model
        self.patch_size = model.patch_size
        self.input_conditioner = model.input_conditioner
        self.patch_embed = architecture.patch_generator
        self.cls_token = architecture.patch_generator.cls_token.token
        self.blocks = architecture.blocks
        self.embed_dim = architecture.embed_dim

    def activate_mlps(self, active_mlp_indices):
        for idx in active_mlp_indices:
            for name, param in self.blocks[idx].mlp.named_parameters():
                param.requires_grad = True
            print(f"Activated MLP of block {idx}")

    def activate_blocks(self, active_block_indices, drop_path_rate=None):
        for idx in active_block_indices:
            for name, param in self.blocks[idx].named_parameters():
                param.requires_grad = True
            if drop_path_rate is not None:
                self.blocks[idx].drop_path1 = DropPath(drop_path_rate)
                self.blocks[idx].drop_path2 = DropPath(drop_path_rate)
            print(f"Activated block {idx}")


class DINOWrapper(nn.Module):
    def __init__(self, model_version: str = "dinov2_vitb14_reg", is_frozen=True):
        super().__init__()
        if not dist.is_initialized() or dist.get_rank() == 0:
            # Pull the model on rank 0 first.
            model = load_model(model_version)
        if dist.is_initialized():
            dist.barrier()
            if dist.get_rank() > 0:
                # Now pull the model from cache on other ranks.
                model = load_model(model_version)
        if is_frozen:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
        self.vit = model
        self.embed_dim = model.embed_dim
        self.patch_size = model.patch_size

    def convert_14to16(self):
        self.vit.patch_embed = DINOPatchEmbed(self.vit.patch_embed)
        conv_weight = F.interpolate(
            self.vit.patch_embed.proj.weight,
            size=(16, 16),
            mode="bilinear",
            align_corners=False,
        )
        self.vit.patch_embed.proj.weight = nn.Parameter(conv_weight)
        self.patch_size = 16

    def activate_mlps(self, active_mlp_indices):
        for idx in active_mlp_indices:
            for name, param in self.vit.blocks[idx].mlp.named_parameters():
                param.requires_grad = True
            print(f"Activated MLP of block {idx}")

    def activate_blocks(self, active_block_indices):
        for idx in active_block_indices:
            for name, param in self.vit.blocks[idx].named_parameters():
                param.requires_grad = True
            print(f"Activated block {idx}")


class DINOPatchEmbed(nn.Module):
    def __init__(
        self,
        vit,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = vit.embed_dim
        self.flatten_embedding = flatten_embedding
        self.proj = vit.proj
        self.norm = vit.norm

    def forward(self, x):
        _, _, H, W = x.shape
        # patch_H, patch_W = self.patch_size
        # assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        # assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"
        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x
