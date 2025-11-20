import torch
import torch.nn as nn

from typing import Self


class CellTypeLoss(nn.Module):
    def __init__(
        self,
        weight_mean,
        loss_fct_mean=nn.MSELoss(reduction="none"),
        loss_fct_deviation=nn.MSELoss(reduction="none"),
        normalize_by_mean=True,
    ):
        super().__init__()
        assert 0 <= weight_mean <= 1, "weight_mean must be between 0 and 1"
        self.loss_fct_mean = loss_fct_mean
        self.loss_fct_deviation = loss_fct_deviation
        self.weight_mean = weight_mean
        self.weight_deviation = 1 - weight_mean
        self.normalize_by_mean = normalize_by_mean

    def forward(
        self, cls_targets, cls_preds, cls_mask, dataset_mean, dataset_deviation
    ):
        # mean across cell types
        cls_targets_mean = (cls_targets * cls_mask).sum(dim=1) / cls_mask.sum(dim=1)
        cls_targets_mean = cls_targets_mean.reshape(cls_targets_mean.shape[0], 1)
        cls_preds_mean = (cls_preds * cls_mask).sum(dim=1) / cls_mask.sum(dim=1)
        cls_preds_mean = cls_preds_mean.reshape(cls_preds_mean.shape[0], 1)

        # normalize by mean
        if self.normalize_by_mean:
            cls_targets_deviation = (cls_targets - cls_targets_mean) / cls_targets_mean
            cls_preds_deviation = (cls_preds - cls_preds_mean) / cls_preds_mean
        else:
            cls_targets_deviation = cls_targets - cls_targets_mean
            cls_preds_deviation = cls_preds - cls_preds_mean

        # loss
        cls_loss_mean = (
            self.loss_fct_mean(cls_preds_mean, cls_targets_mean) * cls_mask
        ).sum() / cls_mask.sum()
        cls_loss_deviation = (
            self.loss_fct_deviation(cls_preds_deviation, cls_targets_deviation)
            * cls_mask
        ).sum() / cls_mask.sum()
        full_loss = (
            self.weight_mean * cls_loss_mean
            + self.weight_deviation * cls_loss_deviation
        )

        return full_loss, cls_loss_mean, cls_loss_deviation


class ExpressionCountsLoss(nn.Module):
    def __init__(
        self: Self,
        weight: torch.Tensor | float = 1.0,
        fct_loss_fn: torch.nn.Module | None = nn.MSELoss(reduction="none"),
        cell_type_specific_loss_fn: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.weight: torch.nn.Buffer = torch.nn.Buffer(
            torch.asarray(weight, dtype=torch.float32)
        )

        self.fct_loss_fn = fct_loss_fn
        #self.cell_type_specific_loss_fn = cell_type_specific_loss_fn

    def forward(
        self: Self,
        logits: torch.Tensor,
        labels: torch.LongTensor | None,
        labels_mask: torch.BoolTensor | None,
    ):
        # Loss
        loss = None
        cls_loss = None
        other_loss = None
        labels_reshaped = None
        labels_mask_reshaped = None

        #print(f"{logits.shape=}, {labels.shape=}, {labels_mask.shape=}")

        loss = None
        if labels is not None:
            B, seq_len, N = labels.shape
            # labels, labels_mask: (B, seq_len, N)
            # Нужно:   (B*N, seq_len, 1)
            # 1) permute(0,2,1) -> (B, N, seq_len)
            # 2) reshape -> (B*N, seq_len)
            # 3) unsqueeze -> (B*N, seq_len, 1)
            # расширяем labels и labels_mask, добавляя "пустой" токен для desc_token

            #pad = torch.zeros((labels.size(0), 1, labels.size(2)), device=labels.device, dtype=labels.dtype)
            #labels = torch.cat([pad, labels], dim=1)
            #print(labels.shape)

            #pad_mask = torch.ones((labels_mask.size(0), 1, labels_mask.size(2)), device=labels_mask.device, dtype=labels_mask.dtype)
            #labels_mask = torch.cat([pad_mask, labels_mask], dim=1)
            #print(labels_mask.shape)

            labels_reshaped = labels.permute(0, 2, 1).reshape(B*N, seq_len, 1).to(logits.device)
            labels_mask_reshaped = labels_mask.permute(0, 2, 1).reshape(B*N, seq_len, 1).to(logits.device)

            # loss
            # Cчитаем общий лосс
            unreduced_loss = self.fct_loss_fn(logits, labels_reshaped)  # (B*N, seq_len, 1)

            if labels_mask_reshaped.sum() > 0:
                # Разделяем маску на последний токен (CLS) и остальные токены
                cls_mask = labels_mask_reshaped[:, -1:, :]  # (B*N, 1, 1)
                other_mask = labels_mask_reshaped[:, :-1, :]  # (B*N, seq_len-1, 1)

                # Считаем лосс для CLS (последний токен)
                cls_loss = None
                if cls_mask.sum() > 0:
                    cls_loss = (unreduced_loss[:, -1:, :] * cls_mask).sum() / cls_mask.sum()

                # Считаем лосс для остальных токенов (все, кроме последнего)
                other_loss = None
                if other_mask.sum() > 0:
                    other_loss = (unreduced_loss[:, :-1, :] * other_mask).sum() / other_mask.sum()

                # Объединяем лоссы
                if cls_loss is not None and other_loss is not None:
                    loss = cls_loss + self.weight * other_loss
                elif cls_loss is not None:
                    loss = cls_loss
                elif other_loss is not None:
                    loss = self.weight * other_loss

        return dict(
            loss=loss,
            cls_loss=cls_loss,
            other_loss=other_loss,
            labels_reshaped=labels_reshaped,
            labels_mask_reshaped=labels_mask_reshaped,
        )
