# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from typing import Dict, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader

from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.nlp.data import TextNormalizationDataset
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.models.text_normalization.modules import EncoderRNN
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.collections.nlp.parts.utils_funcs import tensor2list
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import NeuralType
from nemo.utils import logging

__all__ = ['TextNormalizationModel']


class TextNormalizationModel(NLPModel):
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return None

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return None

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self._tokenizer_context = self._setup_tokenizer(cfg.tokenizer_context)
        self._tokenizer_encoder = self._setup_tokenizer(cfg.tokenizer_encoder)
        self._tokenizer_decoder = self._setup_tokenizer(cfg.tokenizer_decoder)
        super().__init__(cfg=cfg, trainer=trainer)
        self.teacher_forcing = True
        self.context_encoder = EncoderRNN(
            input_size=cfg.context.embedding_size,
            hidden_size=cfg.context.hidden_size,
            num_layer=cfg.context.num_layers,
            dropout=cfg.context.dropout,
        )
        self.tagger_decoder = nn.GRU(
            input_size=cfg.tagger.embedding_size,
            hidden_size=cfg.tagger.hidden_size,
            num_layers=cfg.tagger.num_layers,
            dropout=cfg.tagger.dropout,
            batch_first=True,
        )
        self.seq_encoder = EncoderRNN(
            input_size=cfg.context.embedding_size, hidden_size=cfg.context.hidden_size, num_layers=1
        )
        self.seq_decoder = EncoderRNN(
            input_size=cfg.context.embedding_size, hidden_size=cfg.context.hidden_size, num_layers=1
        )

    # @typecheck()
    def forward(
        self,
        context_ids,
        tag_ids,
        len_context,
        input_ids,
        len_input,
        output_ids,
        len_output,
        l_context_ids,
        r_context_ids,
    ):
        batch_size = len(context_ids)

        # context_outputs, context_hidden = self.encoder_sent(input_seqs=context_ids, input_lens=len_context, hidden=None)
        # max_seq_length = context_ids.shape[1]
        # tagger_hidden = self.tagger_output_emb(context_hidden[0] + context_hidden[1]).unsqueeze(0)
        # all_tagger_outputs = torch.zeros((batch_size, max_seq_length, self._cfg.tagger.num_classes), device=self._device)
        # for i in range(max_seq_length):
        #     context_in = self.tagger_output_emb(context_outputs[i]).unsqueeze(0)
        #     tagger_outputs, tagger_hidden = self.tagger(input=context_in, hidden=tagger_hidden)  # tagger outputs [B, H]
        #     logits = self.tagger_output_layer(tagger_outputs)
        #     all_tagger_outputs[:, i, :] = logits

        # seq_enc_outputs, seq_enc_hidden = self.seq2seq_encoder(input_seqs=input_ids, input_lens=len_input, hidden=None)
        # max_target_length = output_ids.shape[1]

        # all_decoder_outputs = torch.zeros((batch_size, max_target_length, self._tokenizer_sent.vocab_size), device=self._device)
        # decoder_hidden = context_hidden.view(1, batch_size, -1)
        # decoder_input = output_ids[:, 0]
        # for i in range(max_target_length):
        #     # word_input torch.Size([5])    decoder_hidden=torch.Size([1, 5, 25])
        #     if self.teacher_forcing:
        #         decoder_input = output_ids[:, i]
        #     l_context = context_outputs[l_context_ids, torch.arange(end=batch_size, dtype=torch.long)]
        #     r_context = context_outputs[r_context_ids, torch.arange(end=batch_size, dtype=torch.long)]
        #     decoder_output, decoder_hidden = self.seq2seq_decoder(word_input=decoder_input, last_hidden=decoder_hidden, encoder_outputs=context_outputs, l_context=l_context, r_context=r_context, src_len=len_input)
        #     all_decoder_outputs[:, 0, :] = decoder_output
        #     topv, topi = decoder_output.topk(1)
        #     decoder_input = topi.squeeze().detach()  # detach from history as input
        # return all_tagger_outputs, all_decoder_outputs

    def _setup_tokenizer(self, cfg: DictConfig):
        """Instantiates tokenizer based on config and registers tokenizer artifacts.

           If model is being restored from .nemo file then the tokenizer.vocab_file will
           be used (if it exists).

           Otherwise, we will use the vocab file provided in the config (if it exists).

           Finally, if no vocab file is given (this happens frequently when using HF),
           we will attempt to extract the vocab from the tokenizer object and then register it.

        Args:
            cfg (DictConfig): Tokenizer config
        """
        vocab_file = None
        if cfg.vocab_file:
            vocab_file = self.register_artifact(config_path='tokenizer.vocab_file', src=cfg.vocab_file)
        tokenizer = get_tokenizer(
            tokenizer_name=cfg.tokenizer_name,
            vocab_file=vocab_file,
            special_tokens=OmegaConf.to_container(cfg.special_tokens) if cfg.special_tokens else None,
            tokenizer_model=self.register_artifact(config_path='tokenizer.tokenizer_model', src=cfg.tokenizer_model),
        )

        if vocab_file is None:
            # when there is no vocab file we try to get the vocab from the tokenizer and register it
            self._register_vocab_from_tokenizer(vocab_file_config_path='tokenizer.vocab_file', cfg=cfg)
        return tokenizer

    def training_step(self, batch, batch_idx):
        (
            sent_ids,
            tag_ids,
            sent_lens,
            unnormalized_ids,
            char_lens_input,
            normalized_ids,
            char_lens_output,
            l_context_ids,
            r_context_ids,
        ) = batch
        # bs, max_seq_length = sent_ids.shape
        # _, max_target_length = normalized_ids.shape
        self.forward(
            sent_ids,
            tag_ids,
            sent_lens,
            unnormalized_ids,
            char_lens_input,
            normalized_ids,
            char_lens_output,
            l_context_ids,
            r_context_ids,
        )
        # tagger_loss_mask = torch.arange(max_seq_length).to(self._device).expand(bs, max_seq_length) < sent_lens.unsqueeze(1)
        # decoder_loss_mask = torch.arange(max_target_length).to(self._device).expand(bs, max_target_length) < sent_lens.unsqueeze(1)
        # tagger_loss = self.tagger_loss(logits=tagger_logits, labels=tag_ids, loss_mask=tagger_loss_mask)
        # decoder_loss = self.seq2seq_loss(logits=decoder_logits, labels=normalized_ids, loss_mask=decoder_loss_mask)
        # train_loss = tagger_loss + decoder_loss

        # tensorboard_logs = {} #{'train_loss': train_loss, 'lr': self._optimizer.param_groups[0]['lr']}
        # return {'loss': train_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        if self.trainer.testing:
            prefix = 'test'
        else:
            prefix = 'val'

        (
            sent_ids,
            tag_ids,
            sent_lens,
            unnormalized_ids,
            char_lens_input,
            normalized_ids,
            char_lens_output,
            l_context_ids,
            r_context_ids,
        ) = batch
        bs, max_seq_length = sent_ids.shape
        # _, max_target_length = normalized_ids.shape
        # tagger_logits, decoder_logits = self.forward(sent_ids, tag_ids, sent_lens, unnormalized_ids, char_lens_input, normalized_ids, char_lens_output, l_context_ids, r_context_ids)
        # tagger_loss_mask = torch.arange(max_seq_length).to(self._device).expand(bs, max_seq_length) < sent_lens.unsqueeze(1)
        # decoder_loss_mask = torch.arange(max_target_length).to(self._device).expand(bs, max_target_length) < sent_lens.unsqueeze(1)
        # tagger_loss = self.tagger_loss(logits=tagger_logits, labels=tag_ids, loss_mask=tagger_loss_mask)
        # decoder_loss = self.seq2seq_loss(logits=decoder_logits, labels=normalized_ids, loss_mask=decoder_loss_mask)
        # train_loss = tagger_loss + decoder_loss

        # tensorboard_logs = {} #{'train_loss': train_loss, 'lr': self._optimizer.param_groups[0]['lr']}
        # return {'val_loss': train_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    # def validation_epoch_end(self, outputs):
    #     if self.trainer.testing:
    #         prefix = 'test'
    #     else:
    #         prefix = 'val'

    #     avg_loss = torch.stack([x[f'{prefix}_loss'] for x in outputs]).mean()
    #     self.log(f'{prefix}_loss', avg_loss)

    # def test_epoch_end(self, outputs):
    #     return self.validation_epoch_end(outputs)

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        if not train_data_config or not train_data_config.file:
            logging.info(
                f"Dataloader config or file_path for the train is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        if not val_data_config or not val_data_config.file:
            logging.info(
                f"Dataloader config or file_path for the validation is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        if not test_data_config or test_data_config.file is None:
            logging.info(
                f"Dataloader config or file_path for the test is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config)

    def _setup_dataloader_from_config(self, cfg: DictConfig):
        input_file = cfg.file
        if not os.path.exists(input_file):
            raise FileNotFoundError(
                f'{input_file} not found! The data should be be stored in TAB-separated files \n\
                "validation_ds.file" and "train_ds.file" for train and evaluation respectively. \n\
                Each line of the files contains text sequences, where words are separated with spaces. \n\
                The label of the example is separated with TAB at the end of each line. \n\
                Each line of the files should follow the format: \n\
                [WORD][SPACE][WORD][SPACE][WORD][...][TAB][LABEL]'
            )

        dataset = TextNormalizationDataset(
            input_file=input_file,
            tokenizer_context=self._tokenizer_context,
            tokenizer_encoder=self._tokenizer_encoder,
            tokenizer_decoder=self._tokenizer_decoder,
            num_samples=cfg.get("num_samples", -1),
            use_cache=self._cfg.dataset.use_cache,
        )

        dl = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.get("num_workers", 0),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
            collate_fn=dataset.collate_fn,
        )
        return dl

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []
        return result