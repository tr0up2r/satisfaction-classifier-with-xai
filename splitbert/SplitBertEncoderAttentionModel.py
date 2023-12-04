import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from transformers import modeling_outputs
from typing import Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModel
from utils import PositionalEncoding
from utils import get_batch_embeddings
from utils import make_masks


class SplitBertEncoderAttentionModel(nn.Module):
    def __init__(self, num_labels, embedding_size, max_len, max_post_len, max_comment_len, device, target,
                 encoder_mask_mode, attention_mask_mode, softmax=False):
        super().__init__()
        self.num_labels = num_labels
        self.embedding_size = embedding_size
        self.max_len = max_len
        self.max_post_len = max_post_len
        self.max_comment_len = max_comment_len
        self.target = target
        self.device = device
        self.encoder_mask_mode = encoder_mask_mode
        self.attention_mask_mode = attention_mask_mode
        self.softmax = softmax

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.sbert = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.fc_layer = nn.Linear(self.embedding_size, self.embedding_size)
        self.fc_layer_for_pc = nn.Linear(768, self.embedding_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=8)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.mhead_attention = nn.MultiheadAttention(embed_dim=self.embedding_size, num_heads=8)
        self.pe = PositionalEncoding(self.embedding_size, max_len=self.max_len)
        self.layer_norm = nn.LayerNorm(self.embedding_size)

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

        self.classifier1 = nn.Linear(self.embedding_size * (2 if target == 'post_comment' else 3), self.embedding_size)
        self.classifier2 = nn.Linear(self.embedding_size, self.num_labels)
        self.attn_classifier1 = nn.Linear(self.embedding_size * (self.max_post_len + self.max_comment_len),
                                          (self.embedding_size * (self.max_post_len + self.max_comment_len)) // 2)
        self.attn_classifier2 = nn.Linear((self.embedding_size * (self.max_post_len + self.max_comment_len)) // 2,
                                          self.embedding_size)
        self.mean_attn_layer = nn.Linear(self.embedding_size * 2, self.embedding_size)

    def forward(
            self,
            labels: Optional[torch.tensor] = None,
            input_ids1: Optional[torch.tensor] = None,
            input_ids2: Optional[torch.tensor] = None,
            attention_mask1: Optional[torch.tensor] = None,
            attention_mask2: Optional[torch.tensor] = None,
            sentence_count1: Optional[torch.tensor] = None,
            sentence_count2: Optional[torch.tensor] = None,
    ) -> Union[Tuple[torch.Tensor], modeling_outputs.SequenceClassifierOutput]:

        input_ids_list = [input_ids1, input_ids2]
        attention_mask_list = [attention_mask1, attention_mask2]
        sentence_count_list = [sentence_count1, sentence_count2]
        current_batch_size = len(input_ids1)
        batch_embeddings_list = []

        for input_ids, attention_mask, sentence_count, max_count in zip(input_ids_list, attention_mask_list,
                                                                                  sentence_count_list,
                                                                                  [self.max_post_len,
                                                                                   self.max_comment_len]):
            model = self.sbert
            is_all = False
            max_sentence = max_count

            batch_embeddings = torch.empty(size=(current_batch_size, 1, max_sentence, self.embedding_size),
                                           requires_grad=True).to(self.device)
            batch_embeddings = get_batch_embeddings(model, self.fc_layer_for_pc, self.embedding_size, input_ids,
                                                    attention_mask, sentence_count, max_sentence, batch_embeddings,
                                                    is_all, self.device)
            batch_embeddings_list.append(batch_embeddings)

        batch_embeddings = torch.cat((batch_embeddings_list[0], batch_embeddings_list[1]), dim=2)
        hidden_states = batch_embeddings.clone().detach()
        encoder_outputs = torch.empty(size=(current_batch_size, self.embedding_size * 2), requires_grad=True).to(
            self.device)
        outputs_list = []

        for i in range(len(batch_embeddings)):
            non_zero_rows = batch_embeddings[i][0][batch_embeddings[i][0].sum(dim=1) != 0]
            zero_rows = torch.zeros((batch_embeddings[i][0].shape[0] - non_zero_rows.shape[0], self.embedding_size),
                                    dtype=torch.int, device=self.device)
            batch_embeddings[i] = torch.cat([non_zero_rows, zero_rows])

        i = 0
        for embeddings, p_count, c_count in zip(batch_embeddings, sentence_count_list[0], sentence_count_list[1]):
            embeddings = embeddings.swapaxes(0, 1)
            outputs_list.append(embeddings)
            src_mask, src_key_padding_mask = make_masks(self.max_post_len, [p_count, c_count], self.device,
                                                        self.max_post_len, self.max_comment_len,
                                                        "concat_all")
            if self.encoder_mask_mode:
                encoder_output = self.encoder(embeddings, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            else:
                encoder_output = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)
            encoder_outputs[i][:self.embedding_size] = torch.mean(encoder_output[:p_count+c_count], dim=0).squeeze(0)

            if self.attention_mask_mode:
                attention = self.mhead_attention(encoder_output, encoder_output, encoder_output, attn_mask=src_mask,
                                                 key_padding_mask=src_key_padding_mask)[0]
            else:
                attention = self.mhead_attention(encoder_output, encoder_output, encoder_output,
                                                 key_padding_mask=src_key_padding_mask)[0]

            # mul mask - diagonal masking
            attention = attention.swapaxes(0, 2)
            mask = torch.tensor([1] * (p_count + c_count) +
                                [0] * (self.max_post_len + self.max_comment_len - (p_count + c_count))).to(
                self.device)
            attention = attention.mul(mask).swapaxes(0, 2)

            attention = torch.flatten(attention)
            attention = self.attn_classifier1(attention)
            attention = self.attn_classifier2(attention)
            encoder_outputs[i][self.embedding_size:] = attention
            i += 1
        encoder_outputs = self.mean_attn_layer(encoder_outputs)
        logits = self.classifier2(encoder_outputs)
        if self.softmax:
            logits = F.softmax(logits, dim=1)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.squeeze(), labels.squeeze())

        return modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states
        )
