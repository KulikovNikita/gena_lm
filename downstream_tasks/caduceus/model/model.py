import torch
import torch.nn as nn


from dataclasses import dataclass

from typing import Optional

from transformers import AutoModel, PreTrainedModel
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


class CrossLayer(torch.nn.Module):
    def __init__(self, hidden_size, desc_size, num_heads = 1, activation = torch.nn.Identity()):
        super().__init__()

        half_size = hidden_size // 2
        self.attn_fwd = torch.nn.MultiheadAttention(
            embed_dim = half_size,
            num_heads = num_heads,
            kdim = desc_size,
            vdim = desc_size,
            batch_first = True,
            bias = False,
        )
        self.attn_bck = torch.nn.MultiheadAttention(
            embed_dim = half_size,
            num_heads = num_heads,
            kdim = desc_size,
            vdim = desc_size,
            batch_first = True,
            bias = False,
        )

        self.fwd_out = torch.nn.Linear(hidden_size, half_size, bias = False)
        self.bck_out = torch.nn.Linear(hidden_size, half_size, bias = False)

        self.norm = torch.nn.RMSNorm(half_size)
        self.activ = activation

    def forward(self, hidden_states, description) -> torch.Tensor:
        def flip(x):
            return x.flip(dims=(-2, -1))

        hidden_size = hidden_states.shape[-1]
        fwd_hidden = hidden_states[..., :(hidden_size // 2)]
        bck_hidden = flip(hidden_states[..., (hidden_size // 2):])
        attn_fwd, _ = self.attn_fwd(fwd_hidden, description, description)
        attn_bck, _ = self.attn_bck(bck_hidden, description, description) 
        attn_out = self.activ(torch.cat((attn_fwd, attn_bck), dim = -1))

        fwd_out = self.norm(self.fwd_out(attn_out) + fwd_hidden)
        bck_out = self.norm(self.bck_out(attn_out) + bck_hidden)
        return torch.cat((fwd_out, flip(bck_out)), dim = -1)

class FusedLayer(torch.nn.Module):
    def __init__(self, caduceus, xattention):
        super().__init__()

        self.caduceus = caduceus
        self.xattention = xattention

    def forward(self, hidden_states, desc_vectors, residual = None):
        hidden_states, residual = self.caduceus(
            hidden_states, residual, inference_params=None
        )
        hidden_states = self.xattention(
            hidden_states, desc_vectors
        )
        return (hidden_states, residual)

class CaduceusExpressionCountsModel(PreTrainedModel):
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
        feature_count = 16,
        nhead=8,
        weight=1.0,
        hf_model_name: str = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
    ):
        config.initializer_range = 0.02

        super().__init__(config)
        self.config = config
        self.hidden_size = config.d_model
        self.hidden_size_desc = hidden_size_desc

        caduceus = AutoModel.from_pretrained(hf_model_name, trust_remote_code=True)

        self.embeddings = caduceus.backbone.embeddings

        self.norm_f = caduceus.backbone.norm_f

        raw_layers = caduceus.backbone.layers
        cross_layers = torch.nn.ModuleList([
            CrossLayer(
                hidden_size = self.hidden_size,
                desc_size = hidden_size_desc,
                activation = activation,
                num_heads = nhead,
            )
            for _ in range(len(raw_layers))
        ])
        self.layers = torch.nn.ModuleList([
            FusedLayer(raw_layer, cross_layer)
            for raw_layer, cross_layer in zip(raw_layers, cross_layers)
        ])

        self.feature_count = feature_count
        self.feature_weights: torch.nn.Parameter = torch.nn.Parameter(
            torch.randn(self.feature_count, self.hidden_size_desc)
        )
        torch.nn.init.orthogonal_(self.feature_weights)

        # 4) Classifier
        self.classifier = nn.Linear(self.hidden_size, 1)

        # 5) Loss
        self.activation = activation
        self.weight = weight
        self.losses = losses

        self.post_init()

    def pooling(self, hidden_states) -> torch.Tensor:
        return torch.mean(hidden_states, dim = -2)

    def forward_batch(self, hidden_states, desc_vectors) -> torch.Tensor:
        # hidden_states: B*N, seq_len, hidden_size
        # desc_vectors: B*N, desc_len, desc_size
        residual = None
        for layer in self.layers:
            # TODO: Add support for gradient checkpointing
            hidden_states, residual = layer(
                hidden_states, desc_vectors, residual
            )

        return hidden_states

    def forward_model(self, 
                      desc_vectors,
                      input_ids = None,
                      inputs_embeds = None, 
                      **kwargs):
        # desc_vectors: batch_size, desc_len, desc_size
        # hidden_states: batch_size, seq_len, hidden_size
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(input_ids)

        h_batch_size, seq_len, h_hidden_size = hidden_states.shape
        d_batch_size, desc_len, d_hidden_size = desc_vectors.shape
        assert h_batch_size == d_batch_size

        all_hidden_states = hidden_states.unsqueeze(1).expand(-1, desc_len, -1, -1) \
            .reshape(h_batch_size * desc_len, seq_len, h_hidden_size)
        all_desc_states = desc_vectors.unsqueeze(2).expand(-1, -1, self.feature_count, -1) \
            .reshape(d_batch_size * desc_len, self.feature_count, d_hidden_size)
        all_desc_states = all_desc_states + self.feature_weights.unsqueeze(0) \
            .expand(d_batch_size * desc_len, self.feature_count, d_hidden_size)

        all_hidden_states = self.forward_batch(all_hidden_states, all_desc_states)
        logits = self.classifier(all_hidden_states)
        logits = self.activation(logits)
        return (all_hidden_states, logits)

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

        hidden_states, logits = self.forward_model(
            desc_vectors, input_ids, inputs_embeds
        )

        losses = dict()
        if self.losses:
            losses = self.losses(
                logits=logits,
                labels=labels,
                labels_mask=labels_mask,
            )

        #print(losses)

        if not return_dict:
            return (losses["loss"], logits)

        output = ExpressionCountsModelOutput(
            logits=logits,
            hidden_states=hidden_states,
            loss=losses.get("loss", None),
            cls_loss=losses.get("cls_loss", None),
            mean_loss=losses.get("mean_loss", None),
            other_loss=losses.get("other_loss", None),
            deviation_loss=losses.get("deviation_loss", None),
            labels_reshaped=losses.get("labels_reshaped", None),
            labels_mask_reshaped=losses.get("labels_mask_reshaped", None),
        )

        return output
