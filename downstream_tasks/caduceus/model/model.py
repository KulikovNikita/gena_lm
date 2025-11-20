import torch
import torch.nn as nn


from dataclasses import dataclass

from typing import Optional

from transformers import AutoModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput

import numpy

torch.serialization.add_safe_globals(
    [
        numpy.core.multiarray.scalar,
        numpy._core.multiarray.scalar,
        numpy._core.multiarray._reconstruct,
        numpy.ndarray,
        numpy.dtype,
        numpy.dtypes.UInt32DType,
    ]
)


@dataclass
class ExpressionCountsModelOutput(TokenClassifierOutput):
    labels_reshaped: Optional[torch.FloatTensor] = None
    labels_mask_reshaped: Optional[torch.FloatTensor] = None
    cls_loss: Optional[torch.FloatTensor] = None
    other_loss: Optional[torch.FloatTensor] = None
    mean_loss: Optional[torch.FloatTensor] = None
    deviation_loss: Optional[torch.FloatTensor] = None


class ExpressionCountsModel(BertPreTrainedModel):
    """
    Размерности:
      - input_ids: (B, seq_len)
      - attention_mask: (B, seq_len)
      - token_type_ids: (B, seq_len) [опционально]
      - desc_vectors: (B, N, hidden_size)
      - labels: (B, seq_len, N) -> приводим к (B*N, seq_len, 1)
      - labels_mask: (B, seq_len, N) -> приводим к (B*N, seq_len, 1)

    Шаги:
      1) Прогоняем через GENA -> (B, seq_len, hidden_size)
      2) Расширяем выход -> (B, N, seq_len, hidden_size)
      3) Прогоняем desc_vectors через MLP
      4) Складываем desc_vectors с CLS
      5) Превращаем (B, N, seq_len, hidden_size) -> (B*N, seq_len, hidden_size)
      6) Прогоняем через Encoder
      7) classifier -> (B*N, seq_len, 1)
      8) Меняем labels и labels_mask -> (B*N, seq_len, 1), считаем loss
    """

    def __init__(
        self,
        config,
        losses=None,
        activation=nn.Identity(),
        hidden_size_desc=768,
        hidden_ff=1024,
        num_encoder_layers=3,
        nhead=8,
        weight=1.0,
        text_model=None,
        hf_model_name: str = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
    ):
        print(config)
        config.initializer_range = 0.02
        super().__init__(config)
        self.config = config
        self.hidden_size = config.d_model
        self.hidden_size_desc = hidden_size_desc

        self.caduceus = AutoModel.from_pretrained(hf_model_name, trust_remote_code=True)

        if text_model is not None:
            self.desc_fc = text_model
        else:
            # 2) MLP для desc_vectors
            self.desc_fc = nn.Sequential(
                nn.Linear(self.hidden_size_desc, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )

        # 3) Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_ff,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # 4) Classifier
        self.classifier = nn.Linear(self.hidden_size, 1)

        # 5) Loss
        self.activation = activation
        self.weight = weight
        self.losses = losses

        self.post_init()

    def forward(
        self,
        input_ids=None,
        labels_mask=None,
        inputs_embeds=None,
        labels=None,
        output_hidden_states=None,
        return_dict=None,
        desc_vectors=None,
        dataset_mean=None,
        dataset_deviation=None,
        **kwargs,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Прогоняем через GENA
        caduceus_outputs = self.caduceus(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        # Notaton:
        # B - batch size
        # N - number of cell types (a.k.a experiment descriptors)
        # seq_len - sequence length (number of tokens in the input sequence)
        # hidden_size - hidden size

        # (B, seq_len, hidden_size)
        sequence_output = caduceus_outputs.last_hidden_state
        B, seq_len, hidden_size = sequence_output.shape  # (B, seq_len, hidden_size)

        # Assuming that desc_vectors.shape -> (B, N, hidden_size_desc), where N - number of cell types (a.k.a experiment descriptors)
        N = desc_vectors.shape[1]

        # Расширяем выход
        # (B, seq_len, hidden_size) -> (B, N, seq_len, hidden_size)
        seq_out_expanded = sequence_output.unsqueeze(1).expand(
            -1, N, -1, -1
        )  # B, N, seq_len, hidden_size

        # Прогоняем desc_vectors через MLP
        # (B, N, hidden_size) -> (B*N, hidden_size)
        desc_vectors_2d = desc_vectors.reshape(B * N, desc_vectors.shape[-1])
        desc_fc_output = self.desc_fc(desc_vectors_2d)  # (B*N, hidden_size)
        # (B*N, hidden_size) -> (B, N, hidden_size)
        desc_fc_output = desc_fc_output.reshape(B, N, hidden_size)

        # Складываем desc_vectors с CLS
        # CLS-токен — seq_out_expanded[:, :, 0, :]  (B, N, hidden_size)
        seq_out_expanded = seq_out_expanded.contiguous()
        seq_out_expanded[:, :, 0, :] = (
            seq_out_expanded[:, :, 0, :].clone() + desc_fc_output
        )

        # (B, N, seq_len, hidden_size) -> (B*N, seq_len, hidden_size)
        seq_out_flat = seq_out_expanded.reshape(B * N, seq_len, hidden_size)

        # Прогоняем через Encoder
        encoder_output = self.transformer_encoder(
            seq_out_flat
        )  # (B*N, seq_len, hidden_size)

        # Classifier -> (B*N, seq_len, 1)
        logits = self.classifier(encoder_output)
        logits = self.activation(logits)

        # Loss
        losses = dict()
        if self.losses:
            losses = self.losses(
                logits=logits,
                labels=labels,
                labels_mask=labels_mask,
            )

        if not return_dict:
            return (losses["loss"], logits)

        output = ExpressionCountsModelOutput(
            logits=logits,
            hidden_states=sequence_output,
            loss=losses.get("loss", None),
            cls_loss=losses.get("cls_loss", None),
            mean_loss=losses.get("mean_loss", None),
            other_loss=losses.get("other_loss", None),
            deviation_loss=losses.get("deviation_loss", None),
            labels_reshaped=losses.get("labels_reshaped", None),
            labels_mask_reshaped=losses.get("labels_mask_reshaped", None),
        )

        return output
