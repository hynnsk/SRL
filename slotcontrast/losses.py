from typing import Any, Dict, Optional, Tuple

import einops
import torch
from torch import nn

from slotcontrast import modules, utils
import math

@utils.make_build_fn(__name__, "loss")
def build(config, name: str):
    target_transform = None
    if config.get("target_transform"):
        target_transform = modules.build_module(config.get("target_transform"))

    cls = utils.get_class_by_name(__name__, name)
    if cls is not None:
        return cls(
            target_transform=target_transform,
            **utils.config_as_kwargs(config, ("target_transform",)),
        )
    else:
        raise ValueError(f"Unknown loss `{name}`")

class Loss(nn.Module):
    """Base class for loss functions.

    Args:
        video_inputs: If true, assume inputs contain a time dimension.
        patch_inputs: If true, assume inputs have a one-dimensional patch dimension. If false,
            assume inputs have height, width dimensions.
        pred_dims: Dimensions [from, to) of prediction tensor to slice. Useful if only a
            subset of the predictions should be used in the loss, i.e. because the other dimensions
            are used in other losses.
        remove_last_n_frames: Number of frames to remove from the prediction before computing the
            loss. Only valid with video inputs. Useful if the last frame does not have a
            correspoding target.
        target_transform: Transform that can optionally be applied to the target.
    """

    def __init__(
        self,
        pred_key: str,
        target_key: str,
        video_inputs: bool = False,
        patch_inputs: bool = True,
        keep_input_dim: bool = False,
        pred_dims: Optional[Tuple[int, int]] = None,
        remove_last_n_frames: int = 0,
        target_transform: Optional[nn.Module] = None,
        input_key: Optional[str] = None,
    ):
        super().__init__()
        self.pred_path = pred_key.split(".")
        self.target_path = target_key.split(".")
        self.video_inputs = video_inputs
        self.patch_inputs = patch_inputs
        self.keep_input_dim = keep_input_dim
        self.input_key = input_key
        self.n_expected_dims = (
            2 + (1 if patch_inputs or keep_input_dim else 2) + (1 if video_inputs else 0)
        )

        if pred_dims is not None:
            assert len(pred_dims) == 2
            self.pred_dims = slice(pred_dims[0], pred_dims[1])
        else:
            self.pred_dims = None

        self.remove_last_n_frames = remove_last_n_frames
        if remove_last_n_frames > 0 and not video_inputs:
            raise ValueError("`remove_last_n_frames > 0` only valid with `video_inputs==True`")

        self.target_transform = target_transform
        self.to_canonical_dims = self.get_dimension_canonicalizer()

    def get_dimension_canonicalizer(self) -> torch.nn.Module:
        """Return a module which reshapes tensor dimensions to (batch, n_positions, n_dims)."""
        if self.video_inputs:
            if self.patch_inputs:
                pattern = "B F P D -> B (F P) D"
            elif self.keep_input_dim:
                return torch.nn.Identity()
            else:
                pattern = "B F D H W -> B (F H W) D"
        else:
            if self.patch_inputs:
                return torch.nn.Identity()
            else:
                pattern = "B D H W -> B (H W) D"

        return einops.layers.torch.Rearrange(pattern)

    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        target = utils.read_path(outputs, elements=self.target_path, error=False)
        if target is None:
            target = utils.read_path(inputs, elements=self.target_path)

        target = target.detach()

        if self.target_transform:
            with torch.no_grad():
                if self.input_key is not None:
                    target = self.target_transform(target, inputs[self.input_key])
                else:
                    target = self.target_transform(target)

        # Convert to dimension order (batch, positions, dims)
        target = self.to_canonical_dims(target)

        return target

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        prediction = utils.read_path(outputs, elements=self.pred_path)
        if prediction.ndim != self.n_expected_dims:
            raise ValueError(
                f"Prediction has {prediction.ndim} dimensions (and shape {prediction.shape}), but "
                f"expected it to have {self.n_expected_dims} dimensions."
            )

        if self.video_inputs and self.remove_last_n_frames > 0:
            prediction = prediction[:, : -self.remove_last_n_frames]

        # Convert to dimension order (batch, positions, dims)
        prediction = self.to_canonical_dims(prediction)

        if self.pred_dims:
            prediction = prediction[..., self.pred_dims]

        return prediction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement in subclasses")

class TorchLoss(Loss):
    """Wrapper around PyTorch loss functions."""

    def __init__(
        self,
        pred_key: str,
        target_key: str,
        loss: str,
        loss_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)
        loss_kwargs = loss_kwargs if loss_kwargs is not None else {}
        if hasattr(torch.nn, loss):
            self.loss_fn = getattr(torch.nn, loss)(reduction="mean", **loss_kwargs)
        else:
            raise ValueError(f"Loss function torch.nn.{loss} not found")

        # Cross entropy loss wants dimension order (batch, classes, positions)
        self.positions_last = loss == "CrossEntropyLoss"

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.positions_last:
            prediction = prediction.transpose(-2, -1)
            target = target.transpose(-2, -1)

        return self.loss_fn(prediction, target)

class MSELoss(TorchLoss):
    def __init__(self, pred_key: str, target_key: str, **kwargs):
        super().__init__(pred_key, target_key, loss="MSELoss", **kwargs)

class MAELoss(TorchLoss):
    def __init__(self, pred_key: str, target_key: str, **kwargs):
        super().__init__(pred_key, target_key, loss="L1Loss", **kwargs)

class CrossEntropyLoss(TorchLoss):
    def __init__(self, pred_key: str, target_key: str, **kwargs):
        super().__init__(pred_key, target_key, loss="CrossEntropyLoss", **kwargs)

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        features = features.squeeze(0)
        T, B, D = features.size()

        mask = torch.kron(
            torch.eye(B, dtype=torch.bool, device=device),
            torch.ones((T, T), dtype=torch.bool, device=device)
        )
        mask_size = mask.size(0)

        ignore = mask.clone()
        for i in range(B):
            start = i * T
            idx = torch.arange(start, start + T - 1, device=device)
            next_idx = idx + 1
            ignore[idx, next_idx] = False
            ignore[next_idx, idx] = False
        ignore = ~ignore

        mask = mask.float()
        ignore = ignore.float()

        mask = mask * ignore

        features_flat = features.permute(1, 0, 2).reshape(B * T, D)

        anchor_dot_contrast = torch.div(
            torch.matmul(features_flat, features_flat.T),
            self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(mask_size, device=device).view(-1, 1), 0
        )
        mask = mask * logits_mask
        ignore_mask = ignore*logits_mask

        exp_logits = torch.exp(logits) * ignore_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(mask_size).mean()

        return loss

class Slot_Slot_Contrastive_Loss(Loss):
    def __init__(
        self,
        pred_key: str,
        target_key: str,
        temperature: float = 0.1,
        batch_contrast: bool = True,
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)
        self.batch_contrast = batch_contrast
        self.supcon = SupConLoss(temperature=temperature)

    def forward(self, slots, _):
        slots = nn.functional.normalize(slots, p=2.0, dim=-1)
        if self.batch_contrast:
            slots = slots.split(1)
            slots = torch.cat(slots, dim=-2)

        loss = self.supcon(slots)
        return loss

class Slot_Redundancy_Loss(Loss):
    def __init__(
        self,
        pred_key: str,
        target_key: str,
        k: int = 5,
        train_start: int = 0,
        train_end: int = 10000,
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)
        self.k = k

    def choose_idx(self, slots_attn_mask, slots):
        B,T,S,N = slots_attn_mask.shape

        slots_last = slots[:, -1, :, :]
        S = slots_last.size(1)
        slots_norm = torch.nn.functional.normalize(slots_last, dim=-1)
        sim = torch.matmul(slots_norm, slots_norm.transpose(-1, -2))

        tri_i, tri_j = torch.triu_indices(S, S, offset=1, device=slots.device)
        sim_triu_base = sim[:, tri_i, tri_j]

        logN = math.log(N)
        eps = 1e-8
        p = slots_attn_mask.transpose(1, 2)

        ent = (p * torch.log(p + eps)).sum(dim=-1)
        kl_t = ent + logN
        kl_mean = kl_t.mean(dim=-1)

        k = self.k
        chosen_idx_k = torch.full((B, k), -1, dtype=torch.long, device=slots.device)
        banned = torch.zeros(B, S, dtype=torch.bool, device=slots.device)
        for step in range(k):
            valid_pair = (~banned[:, tri_i]) & (~banned[:, tri_j])
            sim_curr = sim_triu_base.masked_fill(~valid_pair, float('-inf'))

            has_any = valid_pair.any(dim=1)
            if has_any.any():
                best_k = sim_curr.argmax(dim=1)
                best_k = torch.where(has_any, best_k, torch.zeros_like(best_k))
                i_idx = tri_i[best_k]
                j_idx = tri_j[best_k]

                kl_i = kl_mean[torch.arange(B, device=slots.device), i_idx]
                kl_j = kl_mean[torch.arange(B, device=slots.device), j_idx]
                choose_i = kl_i <= kl_j
                chosen = torch.where(choose_i, i_idx, j_idx)
                chosen = torch.where(has_any, chosen, torch.full_like(chosen, -1))

                chosen_idx_k[:, step] = chosen
                upd_mask = has_any & (chosen >= 0)
                if upd_mask.any():
                    banned[upd_mask, chosen[upd_mask]] = True
            else:
                break
        return chosen_idx_k

    def forward(self, slots_attn_mask, slots):

        chosen_idx_k = self.choose_idx(slots_attn_mask, slots)

        assert slots_attn_mask.dim() == 4, "slots_attn_mask must be [B,T,S,N]"
        B, T, S, N = slots_attn_mask.shape
        device = slots_attn_mask.device

        k = chosen_idx_k.shape[1]
        idx = chosen_idx_k.to(device)

        valid = (idx >= 0)
        safe_idx = idx.clamp(min=0)
        gather_idx = safe_idx[:, None, :, None]
        gather_idx = gather_idx.expand(B, T, k, N)

        eps = 1e-8
        sel = torch.gather(slots_attn_mask, dim=2, index=gather_idx)

        logN = math.log(N)
        kl = (sel * (torch.log(sel + eps) + logN)).sum(dim=-1)

        valid_btkn = valid[:, None, :].expand(B, T, k)
        kl = torch.where(valid_btkn, kl, torch.zeros_like(kl))

        kl_mean_per_bt = kl.sum(dim=-1) / (valid_btkn.sum(dim=-1).clamp(min=1))
        loss = kl_mean_per_bt.sum() / (B * T)

        return loss


class DECConsLoss(Loss):
    def __init__(
        self,
        pred_key: str,
        target_key: str,
        temperature: float = 0.1,
        base_temperature: float = 0.07,
        topk: int = 8,
        train_start: int = 20000,
        group_frame: int = 2,
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.topk = topk
        self.train_start = train_start
        self.group_frame = group_frame

    def forward(self, feat_trainable, feat_criterion, grp_masks):
        device = feat_trainable.device

        B, T, N, C = feat_trainable.shape
        feat_trainable = nn.functional.normalize(feat_trainable, p=2, dim=-1)
        feat_trainable = feat_trainable.reshape(B, T//self.group_frame, self.group_frame*N, C)

        _, _, _, D = feat_criterion.shape
        feat_criterion = nn.functional.normalize(feat_criterion, p=2, dim=-1)
        feat_criterion = feat_criterion.reshape(B, T//self.group_frame, self.group_frame*N, D)

        BT = B * T
        S = grp_masks.size(-2)

        # semi positive mask
        grp_masks = grp_masks.reshape(B * T, S, N)
        grp_masks = torch.argmax(grp_masks, dim=1)
        grp_masks = grp_masks.reshape(BT, N)
        grp_masks_argmax = grp_masks.reshape(BT // self.group_frame, self.group_frame * N)
        grp_same_slot = grp_masks_argmax.unsqueeze(-1) == grp_masks_argmax.unsqueeze(-2)

        diagonal = torch.eye(N * self.group_frame, dtype=torch.bool, device=device)
        sim_no_diag = grp_same_slot | diagonal
        semi_pos_mask = sim_no_diag.reshape(B * T // self.group_frame, N * self.group_frame, N * self.group_frame)

        # positive mask
        self_idx = torch.arange(self.group_frame * N, device=device, dtype=torch.long)[None, :, None]
        self_idx = self_idx.expand(B*T//self.group_frame, -1, -1)

        pos_mask = torch.zeros(B*T//self.group_frame, self.group_frame * N, self.group_frame * N, device=device, dtype=torch.bool)
        pos_mask.scatter_(dim=-1, index=self_idx, value=True)
        pos_mask = pos_mask.reshape(B * T // self.group_frame, self.group_frame * N, self.group_frame * N)

        # compute logits
        feat_trainable = feat_trainable.reshape(B * T // self.group_frame, self.group_frame * N, C).contiguous()
        feat_criterion = feat_criterion.reshape(B * T // self.group_frame, self.group_frame * N, C).contiguous()
        logits = torch.div(torch.matmul(feat_trainable, feat_criterion.transpose(-1, -2)), self.temperature)

        pos_mask = pos_mask.float()
        semi_pos_mask = semi_pos_mask.float()

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(-1, keepdim=True))

        # ranking contrastive loss
        mean_log_prob_pos = (semi_pos_mask * log_prob).sum(-1) / semi_pos_mask.sum(-1)
        semi_loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        mean_log_prob_pos = (pos_mask * log_prob).sum(-1) / pos_mask.sum(-1)
        pos_loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = (semi_loss + pos_loss).mean() / 2.

        return loss


class ENCConsLoss(Loss):
    def __init__(
        self,
        pred_key: str,
        target_key: str,
        temperature: float = 0.1,
        base_temperature: float = 0.07,
        topk: int = 8,
        train_start: int = 20000,
        group_frame: int = 2,
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.topk = topk
        self.train_start = train_start
        self.group_frame = group_frame

    def forward(self, feat_trainable, feat_criterion, dec_masks):
        device = feat_trainable.device

        B, T, N, C = feat_trainable.shape
        feat_trainable = nn.functional.normalize(feat_trainable, p=2, dim=-1)
        feat_trainable = feat_trainable.reshape(B, T//self.group_frame, self.group_frame*N, C)

        _, _, _, D = feat_criterion.shape
        feat_criterion = nn.functional.normalize(feat_criterion, p=2, dim=-1)
        feat_criterion = feat_criterion.reshape(B, T//self.group_frame, self.group_frame*N, D)

        # semi positive mask
        BT = B * T
        S = dec_masks.size(-2)
        dec_masks = dec_masks.reshape(B * T, S, N)
        dec_masks = torch.argmax(dec_masks, dim=1)
        dec_masks = dec_masks.reshape(BT, N)
        masks_argmax = dec_masks.reshape(BT // self.group_frame, self.group_frame * N)
        semi_pos_mask = masks_argmax.unsqueeze(-1) == masks_argmax.unsqueeze(-2)

        # positive mask
        selfsim = torch.matmul(feat_criterion, feat_criterion.transpose(-1, -2))
        pos_mask = []
        for b in range(B):
            selfsim_b = selfsim[b]
            topk_values, _ = selfsim_b.topk(k=self.topk * self.group_frame, dim=-1)
            topk_value = topk_values[:, :, -1].unsqueeze(-1)
            positive_mask_b = (selfsim_b >= topk_value)
            pos_mask.append(positive_mask_b)

        pos_mask = torch.stack(pos_mask, dim=0)
        pos_mask = pos_mask.reshape(B*T//self.group_frame, self.group_frame*N, self.group_frame*N).contiguous()

        # compute logits
        feat_trainable = feat_trainable.reshape(B * T // self.group_frame, self.group_frame * N, C).contiguous()
        logits = torch.div(torch.matmul(feat_trainable, feat_trainable.transpose(-1, -2)), self.temperature)

        pos_mask = pos_mask.float()
        semi_pos_mask = semi_pos_mask.float()

        # excluding self-contrast cases
        logits_mask = torch.eye(self.group_frame * N, dtype=torch.bool, device=device).unsqueeze(0).repeat(B * T // self.group_frame, 1, 1)  # (B*T//2, 2N, 2N)
        logits_mask = ~logits_mask
        logits_mask = logits_mask.float()
        pos_mask = pos_mask * logits_mask
        semi_pos_mask = semi_pos_mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask

        # ranking contrastive loss
        log_prob = logits - torch.log(exp_logits.sum(-1, keepdim=True))
        mean_log_prob_pos = (pos_mask * log_prob).sum(-1) / (pos_mask.sum(-1) + 1e-8)  # B T N
        pos_loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        log_prob = logits - torch.log(exp_logits.sum(-1, keepdim=True))
        mean_log_prob_pos = (semi_pos_mask * log_prob).sum(-1) / (semi_pos_mask.sum(-1) + 1e-8)  # B T N
        semi_loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = (semi_loss + pos_loss).mean() / 2.

        return loss
