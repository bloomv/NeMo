# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import csv
import json
import os
import copy
from collections import OrderedDict as od
from datetime import datetime
from typing import List

import numpy as np
import math
import torch
import wget
from omegaconf import OmegaConf
import soundfile as sf
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.metrics.wer_bpe import WERBPE
from nemo.collections.asr.models import ClusteringDiarizer, EncDecCTCModel, EncDecCTCModelBPE, EncDecRNNTBPEModel
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR, AudioFeatureIterator
from nemo.collections.asr.parts.utils.speaker_utils import audio_rttm_map as get_audio_rttm_map
from nemo.collections.asr.parts.utils.speaker_utils import (
    get_DER,
    labels_to_pyannote_object,
    rttm_to_labels,
    write_rttm2manifest,
)
from nemo.utils import logging

__all__ = ['ASR_DIAR_OFFLINE']

NONE_LIST = ['None', 'none', 'null', '']


def dump_json_to_file(file_path, riva_dict):
    """Write json file from the riva_dict dictionary.
    """
    with open(file_path, "w") as outfile:
        json.dump(riva_dict, outfile, indent=4)


def write_txt(w_path, val):
    """Write text file from the string input.
    """
    with open(w_path, "w") as output:
        output.write(val + '\n')
    return None


def get_uniq_id_from_audio_path(audio_file_path):
    """Get the unique ID from the audio file path
    """
    return '.'.join(os.path.basename(audio_file_path).split('.')[:-1])


def get_file_lists(file_list_path):
    """Read file paths from the given list
    """
    out_path_list = []
    if not file_list_path or (file_list_path in NONE_LIST):
        raise ValueError("file_list_path is not provided.")
    else:
        with open(file_list_path, 'r') as path2file:
            for _file in path2file.readlines():
                out_path_list.append(_file.strip())

    return out_path_list

class WERBPE_TS(WERBPE):
    """
    This is WER class that is modified for generating word_timestamps with logits.
    The functions in WER class is modified to save the word_timestamps whenever character
    is being saved into a list. Please refer to the definition of WER class for
    more information.
    """
    def __init__(
        self,
        tokenizer: TokenizerSpec,
        batch_dim_index=0,
        use_cer=False,
        ctc_decode=True,
        log_prediction=True,
        dist_sync_on_step=False,
    ):

        super().__init__(tokenizer, batch_dim_index, use_cer, ctc_decode, log_prediction, dist_sync_on_step)
    
    def ctc_decoder_predictions_tensor_with_ts(
        self, time_stride, predictions: torch.Tensor, predictions_len: torch.Tensor = None
    ) -> List[str]:
        hypotheses, timestamps, word_timestamps = [], [], []
        # Drop predictions to CPU
        prediction_cpu_tensor = predictions.long().cpu()
        # iterate over batch
        self.time_stride = time_stride
        for ind in range(prediction_cpu_tensor.shape[self.batch_dim_index]):
            prediction = prediction_cpu_tensor[ind].detach().numpy().tolist()
            if predictions_len is not None:
                prediction = prediction[: predictions_len[ind]]
            # CTC decoding procedure
            decoded_prediction, char_ts, timestamp_list = [], [], []
            previous = self.blank_id
            for pdx, p in enumerate(prediction):
                if (p != previous or previous == self.blank_id) and p != self.blank_id:
                    decoded_prediction.append(p)
                    char_ts.append(round(pdx*self.time_stride, 2))
                    timestamp_list.append(round(pdx*self.time_stride, 2))

                previous = p

            hypothesis = self.decode_tokens_to_str_with_ts(decoded_prediction)
            word_ts = self.get_ts_from_decoded_prediction(decoded_prediction, hypothesis, char_ts)

            hypotheses.append(hypothesis)
            timestamps.append(timestamp_list)
            word_timestamps.append(word_ts)
        return hypotheses, timestamps, word_timestamps

    def decode_tokens_to_str_with_ts(self, tokens: List[int]) -> str:
        hypothesis = self.tokenizer.ids_to_text(tokens)
        return hypothesis

    def decode_ids_to_tokens_with_ts(self, tokens: List[int]) -> List[str]:
        token_list = self.tokenizer.ids_to_tokens(tokens)
        return token_list

    def get_ts_from_decoded_prediction(self, decoded_prediction, hypothesis, char_ts):
        decoded_char_list = self.tokenizer.ids_to_tokens(decoded_prediction)
        stt_idx, end_idx = 0, len(decoded_char_list)-1 
        space = '▁'
        word_ts = []
        word_open_flag = False
        for idx, ch in enumerate(decoded_char_list):
            if idx != end_idx and (space == ch and space in decoded_char_list[idx+1]):
                continue

            if (idx == stt_idx or space == decoded_char_list[idx-1] or (space in ch and len(ch) > 1)) and (ch != space):
                _stt = char_ts[idx]
                word_open_flag = True
            
            if word_open_flag and ch != space and (idx == end_idx or space in decoded_char_list[idx+1]):
                _end = round(char_ts[idx] + self.time_stride, 2)
                word_open_flag = False
                word_ts.append([_stt, _end])
        try:
            assert len(hypothesis.split()) == len(word_ts), "Hypothesis does not match word time stamp."
        except:
            import ipdb
            ipdb.set_trace()
        return word_ts

class WER_TS(WER):
    """
    This is WER class that is modified for generating timestamps with logits.
    The functions in WER class is modified to save the timestamps whenever character
    is being saved into a list. Please refer to the definition of WER class for
    more information.
    """

    def __init__(
        self,
        vocabulary,
        batch_dim_index=0,
        use_cer=False,
        ctc_decode=True,
        log_prediction=True,
        dist_sync_on_step=False,
    ):
        super().__init__(vocabulary, batch_dim_index, use_cer, ctc_decode, log_prediction, dist_sync_on_step)

    def decode_tokens_to_str_with_ts(self, tokens: List[int], timestamps: List[int]) -> str:
        """
        Accepts frame-level tokens and timestamp list and collects the timestamps for
        start and end of the each word.
        """
        hypothesis_list, timestamp_list = self.decode_ids_to_tokens_with_ts(tokens, timestamps)
        hypothesis = ''.join(self.decode_ids_to_tokens(tokens))
        return hypothesis, timestamp_list

    def decode_ids_to_tokens_with_ts(self, tokens: List[int], timestamps: List[int]) -> List[str]:
        token_list, timestamp_list = [], []
        for i, c in enumerate(tokens):
            if c != self.blank_id:
                token_list.append(self.labels_map[c])
                timestamp_list.append(timestamps[i])
        return token_list, timestamp_list

    def ctc_decoder_predictions_tensor_with_ts(
        self, predictions: torch.Tensor, predictions_len: torch.Tensor = None,
    ) -> List[str]:
        """
        A shortened version of the original function ctc_decoder_predictions_tensor().
        Replaced decode_tokens_to_str() function with decode_tokens_to_str_with_ts().
        """
        hypotheses, timestamps = [], []
        prediction_cpu_tensor = predictions.long().cpu()
        for ind in range(prediction_cpu_tensor.shape[self.batch_dim_index]):
            prediction = prediction_cpu_tensor[ind].detach().numpy().tolist()
            if predictions_len is not None:
                prediction = prediction[: predictions_len[ind]]

            # CTC decoding procedure with timestamps
            decoded_prediction, decoded_timing_list = [], []
            previous = self.blank_id
            for pdx, p in enumerate(prediction):
                if (p != previous or previous == self.blank_id) and p != self.blank_id:
                    decoded_prediction.append(p)
                    decoded_timing_list.append(pdx)
                previous = p

            text, timestamp_list = self.decode_tokens_to_str_with_ts(decoded_prediction, decoded_timing_list)
            hypotheses.append(text)
            timestamps.append(timestamp_list)

        return hypotheses, timestamps


def get_wer_feat_logit(audio_file_list, asr, frame_len, tokens_per_chunk, delay, model_stride_in_secs, device):
    # Create a preprocessor to convert audio samples into raw features,
    # Normalization will be done per buffer in frame_bufferer
    # Do not normalize whatever the model's preprocessor setting is
    hyps = []
    tokens_list = []
    sample_list = []
    # with open(mfst, "r") as mfst_f:
    for idx, audio_file_path in enumerate(audio_file_list):
        asr.reset()
        # row = json.loads(l.strip())
        samples = asr.read_audio_file_and_return(audio_file_path, delay, model_stride_in_secs)
        logging.info(f"[{idx+1}/{len(audio_file_list)}] FrameBatchASR: {audio_file_path}")
        hyp, tokens = asr.transcribe_with_ts(tokens_per_chunk, delay)
        hyps.append(hyp)
        tokens_list.append(tokens)
        sample_list.append(samples)
        # refs.append(row['text'])
    return hyps, tokens_list, sample_list

def get_samples(audio_file, target_sr=16000):
    with sf.SoundFile(audio_file, 'r') as f:
        dtype = 'int16'
        sample_rate = f.samplerate
        samples = f.read(dtype=dtype)
        if sample_rate != target_sr:
            samples = librosa.core.resample(samples, sample_rate, target_sr)
        samples = samples.astype('float32') / 32768
        samples = samples.transpose()
    return samples

class FrameBatchASR_Logits(FrameBatchASR):
    """
    class for streaming frame-based ASR use reset() method to reset FrameASR's
    state call transcribe(frame) to do ASR on contiguous signal's frames
    """

    def __init__(self, asr_model, frame_len=1.6, total_buffer=4.0, batch_size=4):
    	super().__init__(asr_model, frame_len, total_buffer, batch_size)

    def read_audio_file_and_return(self, audio_filepath: str, delay, model_stride_in_secs):
        samples = get_samples(audio_filepath)
        sample_length = len(samples)
        samples = np.pad(samples, (0, int(delay * model_stride_in_secs * self.asr_model._cfg.sample_rate)))
        frame_reader = AudioFeatureIterator(samples, self.frame_len, self.raw_preprocessor, self.asr_model.device)
        self.set_frame_reader(frame_reader)
        return samples

    def transcribe_with_ts(
        self, tokens_per_chunk: int, delay: int,
    ):
        self.infer_logits()
        self.unmerged = []
        for pred in self.all_preds:
            decoded = pred.tolist()
            self.unmerged += decoded[len(decoded) - 1 - delay : len(decoded) - 1 - delay + tokens_per_chunk]
        return self.greedy_merge(self.unmerged), self.unmerged

class ASR_DIAR_OFFLINE(object):
    """
    A Class designed for performing ASR and diarization together.

    """

    def __init__(self, params):
        self.params = params
        self.nonspeech_threshold = self.params['threshold']
        self.root_path = None
        self.run_ASR = None
        self.frame_VAD = {}

    def set_asr_model(self, ASR_model_name):
        if 'QuartzNet' in ASR_model_name:
            self.run_ASR = self.run_ASR_QuartzNet_CTC
            # self.get_speech_labels_list = self.get_speech_labels_list_QuartzNet_CTC
            asr_model = EncDecCTCModel.from_pretrained(model_name=ASR_model_name, strict=False)
            self.params['offset'] = -0.18
            self.model_stride_in_secs = 0.02
            self.asr_delay_sec = -1*self.params['offset']
        
        elif 'conformer_ctc' in ASR_model_name:
            self.run_ASR = self.run_ASR_BPE_CTC
            asr_model = EncDecCTCModelBPE.from_pretrained(model_name=ASR_model_name, strict=False)
            self.model_stride_in_secs = 0.04
            self.asr_delay_sec = 0.0
            self.params['offset'] = 0
            self.chunk_len_in_sec = 1.6
            self.total_buffer_in_secs = 4
        
        elif 'citrinet' in ASR_model_name:
            self.run_ASR = self.run_ASR_BPE_CTC
            asr_model = EncDecCTCModelBPE.from_pretrained(model_name=ASR_model_name, strict=False)
            self.model_stride_in_secs = 0.08
            self.asr_delay_sec = 0.0
            self.params['offset'] = 0
            self.chunk_len_in_sec = 1.6
            self.total_buffer_in_secs = 4
        
        elif 'conformer_transducer' in ASR_model_name or 'contextnet' in ASR_model_name:
            self.run_ASR = self.run_ASR_BPE_RNNT
            self.get_speech_labels_list = self.save_SAD_labels_list
            asr_model = EncDecRNNTBPEModel.from_pretrained(model_name=ASR_model_name, strict=False)
            self.model_stride_in_secs = 0.04
            self.asr_delay_sec = 0.0
            self.params['offset'] = 0
            self.chunk_len_in_sec = 1.6
            self.total_buffer_in_secs = 4
        
        else:
            raise ValueError(f"ASR model name not found: {self.params['ASR_model_name']}")
        self.params['time_stride'] = self.model_stride_in_secs
        self.asr_batch_size = 16

        return asr_model

    def create_directories(self, output_path):
        """Creates directories for transcribing with diarization.
        """
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            assert os.path.exists(output_path)
            ROOT = output_path
        else:
            ROOT = os.path.join(os.getcwd(), 'asr_with_diar')
        self.oracle_vad_dir = os.path.join(ROOT, 'oracle_vad')
        self.json_result_dir = os.path.join(ROOT, 'json_result')
        self.trans_with_spks_dir = os.path.join(ROOT, 'transcript_with_speaker_labels')
        self.audacity_label_dir = os.path.join(ROOT, 'audacity_label')

        self.root_path = ROOT
        os.makedirs(self.root_path, exist_ok=True)
        os.makedirs(self.oracle_vad_dir, exist_ok=True)
        os.makedirs(self.json_result_dir, exist_ok=True)
        os.makedirs(self.trans_with_spks_dir, exist_ok=True)
        os.makedirs(self.audacity_label_dir, exist_ok=True)

        data_dir = os.path.join(ROOT, 'data')
        os.makedirs(data_dir, exist_ok=True)

    def run_ASR_QuartzNet_CTC(self, _asr_model, audio_file_list):
        """
        Run an ASR model and collect logit, timestamps and text output

        Args:
            _asr_model (class):
                The loaded NeMo ASR model.
            audio_file_list (list):
                The list of audio file paths.
        """
        words_list, word_ts_list = [], [] 

        # A part of decoder instance
        wer_ts = WER_TS(
            vocabulary=_asr_model.decoder.vocabulary,
            batch_dim_index=0,
            use_cer=_asr_model._cfg.get('use_cer', False),
            ctc_decode=True,
            dist_sync_on_step=True,
            log_prediction=_asr_model._cfg.get("log_prediction", False),
        )

        with torch.cuda.amp.autocast():
            transcript_logits_list = _asr_model.transcribe(audio_file_list, batch_size=1, logprobs=True)
            for logit_np in transcript_logits_list:
                log_prob = torch.from_numpy(logit_np)
                logits_len = torch.from_numpy(np.array([log_prob.shape[0]]))
                greedy_predictions = log_prob.argmax(dim=-1, keepdim=False).unsqueeze(0)
                text, char_ts = wer_ts.ctc_decoder_predictions_tensor_with_ts(
                    greedy_predictions, predictions_len=logits_len
                )
                _trans, char_ts_in_feature_frame_idx = self.clean_trans_and_TS(text[0], char_ts[0])
                _spaces_in_sec, _trans_words = self._get_spaces(_trans, char_ts_in_feature_frame_idx, self.params['time_stride'])
                word_ts = self.get_word_ts_from_spaces(char_ts_in_feature_frame_idx, _spaces_in_sec, end_stamp=logit_np.shape[0])
                assert len(_trans_words) == len(word_ts), "Words and word-timestamp list length does not match."
                words_list.append(_trans_words)
                word_ts_list.append(word_ts)
        return words_list, word_ts_list

    def run_ASR_BPE_RNNT(self, _asr_model, audio_file_list):
        raise NotImplementedError

    def run_ASR_BPE_CTC(self, _asr_model, audio_file_list):
        """
        Not implemented Yet
        """
        torch.manual_seed(0)
        words_list, word_ts_list = [], []

        onset_delay = math.ceil(((self.total_buffer_in_secs - self.chunk_len_in_sec)/2)/self.model_stride_in_secs)+1

        tokens_per_chunk = math.ceil(self.chunk_len_in_sec / self.model_stride_in_secs)
        mid_delay = math.ceil((self.chunk_len_in_sec + (self.total_buffer_in_secs - self.chunk_len_in_sec) / 2) / self.model_stride_in_secs)

        wer_ts = WERBPE_TS(
            tokenizer=_asr_model.tokenizer,
            batch_dim_index=0,
            use_cer=_asr_model._cfg.get('use_cer', False),
            ctc_decode=True,
            dist_sync_on_step=True,
            log_prediction=_asr_model._cfg.get("log_prediction", False),
        )
        frame_asr = FrameBatchASR_Logits(
                asr_model=_asr_model, frame_len=self.chunk_len_in_sec, total_buffer=self.total_buffer_in_secs, batch_size=self.asr_batch_size,
        )

        with torch.cuda.amp.autocast():
            print("Performing ASR...") 
            hyps, tokens_list, sample_list= get_wer_feat_logit(
                audio_file_list,
                frame_asr,
                self.chunk_len_in_sec,
                tokens_per_chunk,
                mid_delay,
                self.model_stride_in_secs,
                _asr_model.device)
            
            for k, greedy_predictions_list in enumerate(tokens_list):
                logits_len = torch.from_numpy(np.array([len(greedy_predictions_list)]))
                greedy_predictions_list = greedy_predictions_list[onset_delay: -mid_delay]
                greedy_predictions = torch.from_numpy(np.array(greedy_predictions_list)).unsqueeze(0)
                text, char_ts, word_ts = wer_ts.ctc_decoder_predictions_tensor_with_ts(
                    self.model_stride_in_secs, greedy_predictions, predictions_len=logits_len
                )
                _trans_words, word_ts = text[0].split(), word_ts[0]
                words_list.append(_trans_words)
                word_ts_list.append(word_ts)
                assert len(_trans_words) == len(word_ts)
        return words_list, word_ts_list
    
    def apply_delay(self, word_ts):
        """
        Compensate the constant delay in the decoder output.

        Arg:
            word_ts (List): The list contains the timestamps of the word sequences.
        """
        for p in range(len(word_ts)):
            word_ts[p] = [max(round(word_ts[p][0]-self.asr_delay_sec,2), 0), round(word_ts[p][1]-self.asr_delay_sec,2)]
        return word_ts

    def save_SAD_labels_list(self, word_ts_list, audio_file_list):
        """
        Get non_speech labels from logit output. The logit output is obtained from
        run_ASR() function.

        Args:
            word_ts_list (list):
                The list that contains word timestamps.
            audio_file_list (list):
                The list of audio file paths.
        """
        for i, word_timestamps in enumerate(word_ts_list):
            
            # if not self.params['external_oracle_vad']:
            if self.params['asr_based_vad']:
                speech_labels_float = self._get_speech_labels_from_decoded_prediction(word_timestamps)
                print("speech_labels_float:", speech_labels_float)
                speech_labels = self.get_str_speech_labels(speech_labels_float)
                self.write_VAD_rttm_from_speech_labels(self.root_path, audio_file_list[i], speech_labels)
    
    def _get_speech_labels_from_decoded_prediction(self, input_word_ts):
        speech_labels = []
        word_ts = copy.deepcopy(input_word_ts)
        if word_ts == []:
            return speech_labels
        else:
            count = len(word_ts)-1
            while count > 0:
                if len(word_ts) > 1: 
                    if word_ts[count][0] - word_ts[count-1][1] <= self.nonspeech_threshold:
                        trangeB = word_ts.pop(count)
                        trangeA = word_ts.pop(count-1)
                        word_ts.insert(count-1, [trangeA[0], trangeB[1]])
                count -= 1
        return word_ts 
    
    def get_word_ts_from_spaces(self, char_ts, _spaces_in_sec, end_stamp):
        start_stamp_in_sec = char_ts[0]*self.params['time_stride'] - self.asr_delay_sec
        end_stamp_in_sec = end_stamp * self.params['time_stride'] - self.asr_delay_sec
        word_timetamps_middle = [[_spaces_in_sec[k][1]-self.asr_delay_sec, _spaces_in_sec[k + 1][0]-self.asr_delay_sec] for k in range(len(_spaces_in_sec) - 1)]
        word_timestamps = (
            [[start_stamp_in_sec, _spaces_in_sec[0][0]-self.asr_delay_sec]] + word_timetamps_middle + [[_spaces_in_sec[-1][1]-self.asr_delay_sec, end_stamp_in_sec]]
        )
        return word_timestamps
    
    @staticmethod
    def _get_silence_timestamps(probs, symbol_idx, state_symbol):
        """
        Get timestamps for blanks or spaces (for CTC decoder).

        Args:
            symbol_idx: (int)
                symbol index of blank or space in the ASR decoder.
            state_symbol: (str)
                The string that indicates the current state.
        """
        spaces = []
        idx_state = 0
        state = ''

        if np.argmax(probs[0]) == symbol_idx:
            state = state_symbol

        for idx in range(1, probs.shape[0]):
            current_char_idx = np.argmax(probs[idx])
            if state == state_symbol and current_char_idx != 0 and current_char_idx != symbol_idx:
                spaces.append([idx_state, idx - 1])
                state = ''
            if state == '':
                if current_char_idx == symbol_idx:
                    state = state_symbol
                    idx_state = idx

        if state == state_symbol:
            spaces.append([idx_state, len(probs) - 1])

        return spaces

    def run_diarization(self, 
                        audio_file_list, 
                        words_and_timestamps, 
                        # oracle_vad_manifest=None,
                        oracle_manifest=None,
                        oracle_num_speakers=None, 
                        pretrained_speaker_model=None,
                        pretrained_vad_model=None):
        """
        Run diarization process using the given VAD timestamp (oracle_manifest).

        Args:
            audio_file_list (list):
                The list of audio file paths.
            word_and_timestamps (list):
                The list contains words and word-timestamps.
            oracle_manifest (str):
                json file path which contains timestamp of VAD output.
                if None, we use word-timestamps for VAD and segmentation.
            oracle_num_speakers (int):
                Oracle number of speakers. If None, the number of speakers is estimated.
            pretrained_speaker_model (str):
                NeMo model file path for speaker embedding extractor model.
        """

        if oracle_num_speakers != None:
            if oracle_num_speakers.isnumeric():
                oracle_num_speakers = int(oracle_num_speakers)
            elif oracle_num_speakers in NONE_LIST:
                oracle_num_speakers = None

        data_dir = os.path.join(self.root_path, 'data')

        MODEL_CONFIG = os.path.join(data_dir, 'speaker_diarization.yaml')
        if not os.path.exists(MODEL_CONFIG):
            MODEL_CONFIG = wget.download(self.params['diar_config_url'], data_dir)

        config = OmegaConf.load(MODEL_CONFIG)
        if oracle_manifest == 'asr_based_vad':
            # Use ASR-based VAD for diarization.
            self.save_SAD_labels_list(words_and_timestamps, audio_file_list)
            oracle_manifest = self.write_VAD_rttm(self.oracle_vad_dir, audio_file_list)

        elif oracle_manifest == 'system_vad':
            logging.info(f"Using the provided system VAD model for diarization: {pretrained_vad_model}")
            config.diarizer.vad.model_path = pretrained_vad_model
        else:
            config.diarizer.speaker_embeddings.oracle_vad_manifest = oracle_manifest

        output_dir = os.path.join(self.root_path, 'oracle_vad')
        config.diarizer.paths2audio_files = audio_file_list
        config.diarizer.out_dir = output_dir  # Directory to store intermediate files and prediction outputs
        if pretrained_speaker_model:
            config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model 
        config.diarizer.speaker_embeddings.oracle_vad_manifest = oracle_manifest
        config.diarizer.oracle_num_speakers = oracle_num_speakers
        config.diarizer.speaker_embeddings.shift_length_in_sec = self.params['shift_length_in_sec']
        config.diarizer.speaker_embeddings.window_length_in_sec = self.params['window_length_in_sec']

        oracle_model = ClusteringDiarizer(cfg=config)
        oracle_model.diarize()
        if oracle_manifest == 'system_vad':
            self.get_frame_level_SAD(oracle_model, audio_file_list)        
        diar_labels = self.get_diarization_labels(audio_file_list)

        return diar_labels


    def get_frame_level_SAD(self, oracle_model, audio_file_list):
        """
        Read frame-level SAD.
        Args:
            oracle_model (ClusteringDiarizer):
                ClusteringDiarizer instance.
            audio_file_path (List):
                The list contains file paths for audio files.
        """
        for k, audio_file_path in enumerate(audio_file_list):
            uniq_id = get_uniq_id_from_audio_path(audio_file_path)
            vad_pred_diar = oracle_model.vad_pred_dir
            frame_vad = os.path.join(vad_pred_diar, uniq_id + '.median')
            frame_vad_list = get_file_lists(frame_vad)
            frame_vad_float_list = [ float(x) for x in frame_vad_list] 
            self.frame_VAD[uniq_id] = frame_vad_float_list 



    def get_diarization_labels(self, audio_file_list):
        """
        Save the diarization labels into a list.

        Arg:
            audio_file_list (list):
                The list of audio file paths.
        """
        diar_labels = []
        for k, audio_file_path in enumerate(audio_file_list):
            uniq_id = get_uniq_id_from_audio_path(audio_file_path)
            pred_rttm = os.path.join(self.oracle_vad_dir, 'pred_rttms', uniq_id + '.rttm')
            pred_labels = rttm_to_labels(pred_rttm)
            diar_labels.append(pred_labels)
            est_n_spk = self.get_num_of_spk_from_labels(pred_labels)
            logging.info(f"Estimated n_spk [{uniq_id}]: {est_n_spk}")

        return diar_labels

    def eval_diarization(self, audio_file_list, diar_labels, ref_rttm_file_list):
        """
        Evaluate the predicted speaker labels (pred_rttm) using ref_rttm_file_list.
        DER and speaker counting accuracy are calculated.

        Args:
            audio_file_list (list):
                The list of audio file paths.
            ref_rttm_file_list (list):
                The list of refrence rttm paths.
        """
        ref_labels_list = []
        all_hypotheses, all_references = [], []
        DER_result_dict = {}
        count_correct_spk_counting = 0

        audio_rttm_map = get_audio_rttm_map(audio_file_list, ref_rttm_file_list)
        for k, audio_file_path in enumerate(audio_file_list):
            uniq_id = get_uniq_id_from_audio_path(audio_file_path)
            rttm_file = audio_rttm_map[uniq_id]['rttm_path']
            if os.path.exists(rttm_file):
                ref_labels = rttm_to_labels(rttm_file)
                ref_labels_list.append(ref_labels)
                reference = labels_to_pyannote_object(ref_labels)
                all_references.append(reference)
            else:
                raise ValueError("No reference RTTM file provided.")

            pred_labels = diar_labels[k]

            est_n_spk = self.get_num_of_spk_from_labels(pred_labels)
            ref_n_spk = self.get_num_of_spk_from_labels(ref_labels)
            hypothesis = labels_to_pyannote_object(pred_labels)
            all_hypotheses.append(hypothesis)
            DER, CER, FA, MISS, mapping = get_DER([reference], [hypothesis])
            DER_result_dict[uniq_id] = {
                "DER": DER,
                "CER": CER,
                "FA": FA,
                "MISS": MISS,
                "n_spk": est_n_spk,
                "mapping": mapping[0],
                "spk_counting": (est_n_spk == ref_n_spk),
            }
            count_correct_spk_counting += int(est_n_spk == ref_n_spk)

        DER, CER, FA, MISS, mapping = get_DER(all_references, all_hypotheses)
        logging.info(
            "Cumulative results of all the files:  \n FA: {:.4f}\t MISS {:.4f}\t\
                Diarization ER: {:.4f}\t, Confusion ER:{:.4f}".format(
                FA, MISS, DER, CER
            )
        )
        DER_result_dict['total'] = {
            "DER": DER,
            "CER": CER,
            "FA": FA,
            "MISS": MISS,
            "spk_counting_acc": count_correct_spk_counting / len(audio_file_list),
        }
        return ref_labels_list, DER_result_dict

    def closest_silence_start(self, vad_index_word_end, vad_frames, params, offset=10):
        c = vad_index_word_end + offset
        while c < len(vad_frames):
            if vad_frames[c] <  params['SAD_threshold_for_word_ts']:
                break
            else:
                c += 1
        c = min(len(vad_frames)-1, c)
        return round(c/100.0, 2)


    def get_enhanced_word_ts_list(self, audio_file_list, word_ts_list, params):
        enhanced_word_ts_list = []
        for idx, word_ts_seq_list in enumerate(word_ts_list):
            uniq_id = get_uniq_id_from_audio_path(audio_file_list[idx])
            N = len(word_ts_seq_list)
            enhanced_word_ts_buffer = []
            for k, word_ts in enumerate(word_ts_seq_list):
                if k < N-1:
                    # Case 1 where word length is shorter than minimum word length
                    vad_index_word_end = int(100*word_ts[1])
                    closest_sil_stt = self.closest_silence_start(vad_index_word_end, self.frame_VAD[uniq_id], params)
                    vad_est_len = round(closest_sil_stt - word_ts[0], 2)
                    word_len = round(word_ts[1] - word_ts[0], 2)
                    len_to_next_word = round(word_ts_seq_list[k+1][0] - word_ts[0] - 0.01,2)
                    print(f"vad_est_len: {vad_est_len} word_len:{word_len} len_to_next_word:{len_to_next_word}") 
                    fixed_word_len = min(vad_est_len, len_to_next_word) 
                    fixed_word_len = max(min(params['max_word_ts_length_in_sec'], fixed_word_len), word_len)
                    # else:
                    print(f"fixed_word_len: {fixed_word_len}")
                    # if fixed_word_len < 0.05:
                        # import ipdb; ipdb.set_trace()
                    enhanced_word_ts_buffer.append([word_ts[0], word_ts[0]+fixed_word_len])
            enhanced_word_ts_list.append(enhanced_word_ts_buffer)
        return enhanced_word_ts_list


    def write_json_and_transcript(
        self, audio_file_list, diar_labels, word_list, word_ts_list,
    ):
        """
        Matches the diarization result with ASR output.
        The words and timestamps for the corresponding words are matched
        in the for loop.

        Args:
            audio_file_list (list):
                The list that contains audio file paths.
            diar_labels (list):
                Diarization output labels in str.
            word_list (list):
                The list of words from ASR inference.
            word_ts_list (list):
                Contains word_ts_stt_end lists.
                word_ts_stt_end = [stt, end]
                    stt: Start of the word in sec.
                    end: End of the word in sec.

        """
        total_riva_dict = {}
        if self.params['fix_word_ts_with_SAD']:
            word_ts_list = self.get_enhanced_word_ts_list(audio_file_list, word_ts_list, self.params)
        for k, audio_file_path in enumerate(audio_file_list):
            uniq_id = get_uniq_id_from_audio_path(audio_file_path)
            labels = diar_labels[k]
            audacity_label_words = []
            n_spk = self.get_num_of_spk_from_labels(labels)
            string_out = ''
            riva_dict = od(
                {
                    'status': 'Success',
                    'session_id': uniq_id,
                    'transcription': ' '.join(word_list[k]),
                    'speaker_count': n_spk,
                    'words': [],
                }
            )

            start_point, end_point, speaker = labels[0].split()
            words = word_list[k]
            
            logging.info(f"Creating results for Session: {uniq_id} n_spk: {n_spk} ")
            string_out = self.print_time(string_out, speaker, start_point, end_point, self.params)

            word_pos, idx = 0, 0
            for j, word_ts_stt_end in enumerate(word_ts_list[k]):

                word_pos = (word_ts_stt_end[0] + word_ts_stt_end[1])/2
                if word_pos < float(end_point):
                    string_out = self.print_word(string_out, words[j], self.params)
                else:
                    idx += 1
                    idx = min(idx, len(labels) - 1)
                    start_point, end_point, speaker = labels[idx].split()
                    string_out = self.print_time(string_out, speaker, start_point, end_point, self.params)
                    string_out = self.print_word(string_out, words[j], self.params)

                stt_sec, end_sec = round(word_ts_stt_end[0],2), round(word_ts_stt_end[1], 2)
                riva_dict = self.add_json_to_dict(riva_dict, words[j], stt_sec, end_sec, speaker)

                total_riva_dict[uniq_id] = riva_dict
                audacity_label_words = self.get_audacity_label(
                    words[j], stt_sec, end_sec, speaker, audacity_label_words
                )

            self.write_and_log(uniq_id, riva_dict, string_out, audacity_label_words)
            

        return total_riva_dict

    def get_WDER(self, audio_file_list, total_riva_dict, DER_result_dict, ref_labels_list):
        """
        Calculate Word-level Diarization Error Rate (WDER). WDER is calculated by
        counting the the wrongly diarized words and divided by the total number of words
        recognized by the ASR model.

        Args:
            total_riva_dict: (dict)
                The dictionary that stores riva_dict(dict)indexed by uniq_id variable.
            DER_result_dict: (dict)
                The dictionary that stores DER, FA, Miss, CER, mapping, the estimated
                number of speakers and speaker counting accuracy.
            audio_file_list: (list)
                The list that contains audio file paths.
            ref_labels_list: (list)
                The list that contains the ground truth speaker labels for each segment.
        """
        wder_dict = {}
        grand_total_word_count, grand_correct_word_count = 0, 0
        for k, audio_file_path in enumerate(audio_file_list):

            labels = ref_labels_list[k]
            uniq_id = get_uniq_id_from_audio_path(audio_file_path)
            mapping_dict = DER_result_dict[uniq_id]['mapping']
            words_list = total_riva_dict[uniq_id]['words']

            idx, correct_word_count = 0, 0
            total_word_count = len(words_list)
            ref_label_list = [[float(x.split()[0]), float(x.split()[1])] for x in labels]
            ref_label_array = np.array(ref_label_list)

            for wdict in words_list:
                speaker_label = wdict['speaker_label']
                if speaker_label in mapping_dict:
                    est_spk_label = mapping_dict[speaker_label]
                else:
                    continue
                start_point, end_point, ref_spk_label = labels[idx].split()
                word_range = np.array([wdict['start_time'], wdict['end_time']])
                word_range_tile = np.tile(word_range, (ref_label_array.shape[0], 1))
                ovl_bool = self.isOverlapArray(ref_label_array, word_range_tile)
                if np.any(ovl_bool) == False:
                    continue

                ovl_length = self.getOverlapRangeArray(ref_label_array, word_range_tile)

                if self.params['lenient_overlap_WDER']:
                    ovl_length_list = list(ovl_length[ovl_bool])
                    max_ovl_sub_idx = np.where(ovl_length_list == np.max(ovl_length_list))[0]
                    max_ovl_idx = np.where(ovl_bool == True)[0][max_ovl_sub_idx]
                    ref_spk_labels = [x.split()[-1] for x in list(np.array(labels)[max_ovl_idx])]
                    if est_spk_label in ref_spk_labels:
                        correct_word_count += 1
                else:
                    max_ovl_sub_idx = np.argmax(ovl_length[ovl_bool])
                    max_ovl_idx = np.where(ovl_bool == True)[0][max_ovl_sub_idx]
                    _, _, ref_spk_label = labels[max_ovl_idx].split()
                    correct_word_count += int(est_spk_label == ref_spk_label)

            wder = 1 - (correct_word_count / total_word_count)
            grand_total_word_count += total_word_count
            grand_correct_word_count += correct_word_count

            wder_dict[uniq_id] = wder

        wder_dict['total'] = 1 - (grand_correct_word_count / grand_total_word_count)
        print("Total WDER: ", wder_dict['total'])

        return wder_dict

    def get_str_speech_labels(self, speech_labels_float):
        speech_labels = []
        for start, end in speech_labels_float:
            speech_labels.append("{:.3f} {:.3f} speech".format(start, end))
        return speech_labels
        
    def write_result_in_csv(self, args, WDER_dict, DER_result_dict, effective_WDER):
        """
        This function is for development use.
        Saves the diariazation result into a csv file.
        """
        row = [
            args.threshold,
            WDER_dict['total'],
            DER_result_dict['total']['DER'],
            DER_result_dict['total']['FA'],
            DER_result_dict['total']['MISS'],
            DER_result_dict['total']['CER'],
            DER_result_dict['total']['spk_counting_acc'],
            effective_WDER,
        ]

        with open(os.path.join(self.root_path, args.csv), 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(row)

    @staticmethod
    def _get_spaces(trans, char_ts, time_stride):
        """
        Collect the space symboles with list of words.

        Args:
            trans (list):
                The list of character output (str).
            timestamps (list):
                The list of timestamps (int) for each character.
        """
        assert (len(trans) > 0) and (len(char_ts) > 0), "Transcript and char_ts length should not be 0."
        assert len(trans) == len(char_ts), "Transcript and timestamp lengths do not match."

        spaces_in_sec, word_list = [], []
        stt_idx = 0
        for k, s in enumerate(trans):
            if s == ' ':
                spaces_in_sec.append([round(char_ts[k]*time_stride, 2), round((char_ts[k+1]-1)*time_stride,2)])
                word_list.append(trans[stt_idx:k])
                stt_idx = k + 1
        if len(trans) > stt_idx and trans[stt_idx] != ' ':
            word_list.append(trans[stt_idx:])

        return spaces_in_sec, word_list

    def write_and_log(self, uniq_id, riva_dict, string_out, audacity_label_words):
        """Writes output files and display logging messages.
        """
        ROOT = self.root_path
        logging.info(f"Writing {ROOT}/json_result/{uniq_id}.json")
        dump_json_to_file(f'{ROOT}/json_result/{uniq_id}.json', riva_dict)

        logging.info(f"Writing {ROOT}/transcript_with_speaker_labels/{uniq_id}.txt")
        write_txt(f'{ROOT}/transcript_with_speaker_labels/{uniq_id}.txt', string_out.strip())

        logging.info(f"Writing {ROOT}/audacity_label/{uniq_id}.w.label")
        write_txt(f'{ROOT}/audacity_label/{uniq_id}.w.label', '\n'.join(audacity_label_words))

    @staticmethod
    def clean_trans_and_TS(trans, char_ts):
        """
        Removes the spaces in the beginning and the end.
        The char_ts need to be changed and synced accordingly.

        Args:
            trans (list):
                The list of character output (str).
            char_ts (list):
                The list of timestamps (int) for each character.
        """
        assert (len(trans) > 0) and (len(char_ts) > 0)
        assert len(trans) == len(char_ts)

        trans = trans.lstrip()
        diff_L = len(char_ts) - len(trans)
        char_ts = char_ts[diff_L:]

        trans = trans.rstrip()
        diff_R = len(char_ts) - len(trans)
        if diff_R > 0:
            char_ts = char_ts[: -1 * diff_R]
        return trans, char_ts

    @staticmethod
    def write_VAD_rttm_from_speech_labels(ROOT, AUDIO_FILENAME, speech_labels):
        """Writes a VAD rttm file from speech_labels list.
        """
        uniq_id = get_uniq_id_from_audio_path(AUDIO_FILENAME)
        oracle_vad_dir = os.path.join(ROOT, 'oracle_vad')
        with open(f'{oracle_vad_dir}/{uniq_id}.rttm', 'w') as f:
            for spl in speech_labels:
                start, end, speaker = spl.split()
                start, end = float(start), float(end)
                f.write("SPEAKER {} 1 {:.3f} {:.3f} <NA> <NA> speech <NA>\n".format(uniq_id, start, end - start))

    @staticmethod
    def write_VAD_rttm(oracle_vad_dir, audio_file_list, reference_rttmfile_list_path=None):
        """
        Writes VAD files to the oracle_vad_dir folder.

        Args:
            oracle_vad_dir (str):
                The path of oracle VAD folder.
            audio_file_list (list):
                The list of audio file paths.
        """
        if not reference_rttmfile_list_path:
            reference_rttmfile_list_path = []
            for path_name in audio_file_list:
                uniq_id = get_uniq_id_from_audio_path(path_name)
                reference_rttmfile_list_path.append(f'{oracle_vad_dir}/{uniq_id}.rttm')

        oracle_manifest = os.path.join(oracle_vad_dir, 'oracle_manifest.json')

        write_rttm2manifest(
            paths2audio_files=audio_file_list, paths2rttm_files=reference_rttmfile_list_path, manifest_file=oracle_manifest
        )
        return oracle_manifest

    @staticmethod
    def threshold_non_speech(source_list, params):
        return list(filter(lambda x: x[1] - x[0] > params['threshold'], source_list))

    @staticmethod
    def get_effective_WDER(DER_result_dict, WDER_dict):
        return 1 - (
            (1 - (DER_result_dict['total']['FA'] + DER_result_dict['total']['MISS'])) * (1 - WDER_dict['total'])
        )

    @staticmethod
    def isOverlapArray(rangeA, rangeB):
        startA, endA = rangeA[:, 0], rangeA[:, 1]
        startB, endB = rangeB[:, 0], rangeB[:, 1]
        return (endA > startB) & (endB > startA)

    @staticmethod
    def getOverlapRangeArray(rangeA, rangeB):
        left = np.max(np.vstack((rangeA[:, 0], rangeB[:, 0])), axis=0)
        right = np.min(np.vstack((rangeA[:, 1], rangeB[:, 1])), axis=0)
        return right - left

    @staticmethod
    def get_audacity_label(word, stt_sec, end_sec, speaker, audacity_label_words):
        spk = speaker.split('_')[-1]
        audacity_label_words.append(f'{stt_sec}\t{end_sec}\t[{spk}] {word}')
        return audacity_label_words

    @staticmethod
    def print_time(string_out, speaker, start_point, end_point, params):
        datetime_offset = 16 * 3600
        if float(start_point) > 3600:
            time_str = "%H:%M:%S.%f"
        else:
            time_str = "%M:%S.%f"
        start_point_str = datetime.fromtimestamp(float(start_point) - datetime_offset).strftime(time_str)[:-4]
        end_point_str = datetime.fromtimestamp(float(end_point) - datetime_offset).strftime(time_str)[:-4]
        strd = "\n[{} - {}] {}: ".format(start_point_str, end_point_str, speaker)
        if params['print_transcript']:
            print(strd, end=" ")
        return string_out + strd

    @staticmethod
    def print_word(string_out, word, params):
        word = word.strip()
        if params['print_transcript']:
            print(word, end=" ")
        return string_out + word + " "

    @staticmethod
    def softmax(logits):
        e = np.exp(logits - np.max(logits))
        return e / e.sum(axis=-1).reshape([logits.shape[0], 1])

    @staticmethod
    def get_num_of_spk_from_labels(labels):
        spk_set = [x.split(' ')[-1].strip() for x in labels]
        return len(set(spk_set))

    @staticmethod
    def add_json_to_dict(riva_dict, word, stt, end, speaker):
        riva_dict['words'].append({'word': word, 'start_time': stt, 'end_time': end, 'speaker_label': speaker})
        return riva_dict