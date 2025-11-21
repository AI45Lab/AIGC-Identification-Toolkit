import math
from dataclasses import dataclass, field
from typing import Union, Iterable, List, Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, PreTrainedModel, PreTrainedTokenizer
import torch.nn.functional as F

from .base_processor import WmProcessorBase
from ...src.utils.random_utils import np_temp_random
from ...src.utils.hash_fn import Hash1
from .message_models.lm_message_model import LMMessageModel, TextTooShortError
import logging
logging.basicConfig(filename='example.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')
torch.set_printoptions(threshold=10000)

HfTokenizer = Union[PreTrainedTokenizerFast, PreTrainedTokenizer]



def convert_input_ids_to_key(input_ids):
    if len(input_ids.shape) == 2:
        assert input_ids.shape[0] == 1
        input_ids = input_ids[0]
    return tuple(input_ids.tolist())


@dataclass
class SeqState:
    input_ids: torch.LongTensor = None
    log_probs: torch.FloatTensor = None
    next_lm_predictions: torch.FloatTensor = None

    def key(self):
        return convert_input_ids_to_key(self.input_ids)


@dataclass
class SeqStateDict:
    message_model: LMMessageModel
    messages: torch.LongTensor
    seq_states: Iterable[SeqState] = field(default_factory=list)

    def __post_init__(self):
        self.seq_states_dict = {seq_state.key(): seq_state for seq_state in self.seq_states}

    def find(self, input_ids):
        if len(input_ids.shape) == 2:
            assert input_ids.shape[0] == 1
            input_ids = input_ids[0]
        key = convert_input_ids_to_key(input_ids)
        return self.seq_states_dict.get(key, None)

    def clear(self):
        self.seq_states_dict.clear()

    def refresh(self):
        if len(self.seq_states_dict) == 0:
            return
        cur_length = max([len(seq_state.input_ids) for seq_state in self.seq_states_dict.values()])
        self.seq_states_dict = {key: seq_state for key, seq_state in self.seq_states_dict.items()
                                if len(seq_state.input_ids) >= cur_length - 1}

    def update(self, input_ids, next_lm_predictions):
        if self.messages.device != input_ids.device:
            self.messages = self.messages.to(input_ids.device)
        if len(input_ids.shape) == 2:
            assert input_ids.shape[0] == 1
            input_ids = input_ids[0]
        key = convert_input_ids_to_key(input_ids[:-1])
        seq_state = self.seq_states_dict.get(key, SeqState())
        if seq_state.next_lm_predictions is not None:
            try:

                new_log_probs = self.message_model.cal_log_Ps(
                    input_ids.unsqueeze(0)[:, :-1],
                    messages=self.messages,
                    x_cur=input_ids[-1].reshape(1, 1),
                    lm_predictions=seq_state.next_lm_predictions)
                new_log_probs = new_log_probs.squeeze()

                log_probs = seq_state.log_probs + new_log_probs if seq_state.log_probs is not None \
                    else new_log_probs
            except TextTooShortError:
                log_probs = None
        else:
            log_probs = None
        new_seq_state = SeqState(input_ids=input_ids, log_probs=log_probs,
                                 next_lm_predictions=next_lm_predictions)
        new_key = new_seq_state.key()
        self.seq_states_dict[new_key] = new_seq_state

    def __repr__(self):
        return (f"SeqStateDict(\n"
                f"  message_model={self.message_model},\n"
                f"  messages={self.messages},\n"
                f"  seq_states_dict={self.seq_states_dict}\n"
                f")")


class WmProcessorMessageModel(WmProcessorBase):
    def __init__(self, message, message_model: LMMessageModel, tokenizer, encode_ratio=10.,
                 seed=42, strategy='vanilla', max_confidence_lbd=0.2, top_k=1000):
        super().__init__(seed=seed)
        self.message = message
        if not isinstance(self.message, torch.Tensor):
            self.message = torch.tensor(self.message)
        self.message_model: LMMessageModel = message_model
        self.message_code_len = self.message_model.message_code_len
        self.encode_ratio = encode_ratio
        self.encode_len = int(self.message_code_len * self.encode_ratio)
        self.start_length = 0
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.lm_prefix_len = self.message_model.lm_prefix_len
        if hasattr(self.message_model, 'tokenizer'):
            assert self.message_model.tokenizer == tokenizer
        self.strategy = strategy
        if self.strategy in ['max_confidence', 'max_confidence_updated']:
            self.seq_state_dict = SeqStateDict(seq_states=[], message_model=self.message_model,
                                               messages=torch.arange(2 ** self.message_code_len))
            self.max_confidence_lbd = max_confidence_lbd

    def set_random_state(self):
        pass

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        original_scores = scores.clone()
        if self.message.device != input_ids.device:
            self.message = self.message.to(input_ids.device)
        try:
            input_ids = input_ids[:, self.start_length:] # s_{k-1}
            if input_ids.shape[1] < self.lm_prefix_len:
                raise TextTooShortError
            message_len = input_ids.shape[1] - self.lm_prefix_len
            cur_message = self.message[message_len // self.encode_len]# which part of message is being processed
            cur_message = cur_message.reshape(-1)
            if message_len % self.encode_len == 0:
                if self.strategy in ['max_confidence', 'max_confidence_updated']:
                    self.seq_state_dict.clear()

            topk_scores, topk_indices = torch.topk(scores, scores.shape[1], dim=1)
            

            input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            public_lm_probs = self.message_model.get_lm_predictions(input_texts)

     

            log_Ps = self.message_model.cal_log_Ps(input_ids, x_cur=topk_indices,
                                                   messages=cur_message,
                                                   lm_predictions=public_lm_probs) # compute watermark_logits
            

            log_Ps = log_Ps[:, 0, :]
            scores = scores.to(log_Ps.dtype)
            scores.scatter_(1, topk_indices, log_Ps, reduce='add') # model_logits + watermark_logits


            max_cur = torch.argmax(scores, dim=1)
            self.message_model.seed_cacher.add_cur(max_cur.tolist())

        except TextTooShortError:
            pass
        if self.strategy in ['max_confidence', 'max_confidence_updated']:
            self.seq_state_dict.refresh()

        return scores

    @property
    def decode_messages(self):
        if not hasattr(self, '_decode_messages'):
            self._decode_messages = torch.arange(2 ** self.message_code_len,
                                                 device=self.message_model.device)
        return self._decode_messages

    def decode_with_input_ids(self, input_ids, messages=None, batch_size=16, disable_tqdm=False,
                              non_analyze=False):
        print("begin extracting watermark")
        if messages is None:
            messages = self.decode_messages
        else:
            if not isinstance(messages, torch.Tensor):
                messages = torch.tensor(messages, device=input_ids.device)
        all_log_Ps = []
        assert input_ids.shape[0] == 1
        for i in tqdm(range(0, input_ids.shape[1] - self.lm_prefix_len - 1, batch_size),
                      disable=disable_tqdm):
            batch_input_ids = []
            # for j in range(i, min(i + batch_size, input_ids.shape[1] - self.lm_prefix_len - 1)):
            #     batch_input_ids.append(input_ids[0, :j + self.lm_prefix_len + 1].tolist())
            # x_prefix_ids = [x[:-1] for x in batch_input_ids]
            # x_prefix_texts = self.message_model.lm_tokenizer.batch_decode(x_prefix_ids,
            #                                                               skip_special_tokens=True)
            # x_prefix_texts, batch_input_ids_bools = self.message_model.filter(x_prefix_texts)
            # x_cur = torch.tensor(
            #     [x[-1] for i, x in enumerate(batch_input_ids) if batch_input_ids_bools[i]],
            #     device=input_ids.device).reshape(
            #     -1, 1)
            # self.message_model.seed_cacher.add_cur(x_cur.flatten().tolist())
            # lm_predictions = self.message_model.get_lm_predictions(x_prefix_texts)
            # x_prefix_ids = torch.LongTensor([x[-self.message_model.lm_prefix_len:] for i, x in
            #                                  enumerate(batch_input_ids) if
            #                                  batch_input_ids_bools[i]]).to(input_ids.device)
            # log_Ps = self.message_model.cal_log_Ps(x_prefix_ids, x_cur, messages=messages,
            #                                        lm_predictions=lm_predictions)
            # all_log_Ps.append(log_Ps)
            for j in range(i, min(i + batch_size, input_ids.shape[1] - self.lm_prefix_len - 1)):
                batch_input_ids.append(input_ids[:, j:j + self.lm_prefix_len + 1])
            batch_input_ids = torch.cat(batch_input_ids, dim=0)
            x_prefix = batch_input_ids[:, :-1]
            x_cur = batch_input_ids[:, -1:]
            x_prefix_texts = self.message_model.lm_tokenizer.batch_decode(x_prefix.tolist(),
                                                                          skip_special_tokens=True)
            lm_predictions = self.message_model.get_lm_predictions(x_prefix_texts,
                                                                   input_lm_token_ids=True)
            log_Ps = self.message_model.cal_log_Ps(x_prefix, x_cur, messages=messages,
                                                   lm_predictions=lm_predictions)
            self.message_model.seed_cacher.add_cur(x_cur.flatten().tolist())
            all_log_Ps.append(log_Ps)

        all_log_Ps = torch.cat(all_log_Ps, dim=0)
        all_log_Ps = all_log_Ps.squeeze()
        # print(all_log_Ps)
        decoded_messages = []
        decoded_confidences = []
        for i in range(0, all_log_Ps.shape[0], self.encode_len):
            # top_values, top_indices = torch.topk((all_log_Ps[i:i + self.encode_len] > 0).sum(0), 2)
            #
            # decoded_message = messages[top_indices[0]]
            # decoded_confidence = int(top_values[0])
            # relative_decoded_confidence = int(top_values[0]) - int(top_values[1])
            # decoded_messages.append(decoded_message)
            # decoded_confidences.append((decoded_confidence,relative_decoded_confidence))
            if not non_analyze:
                # print(all_log_Ps[i:i + self.encode_len])
                nums = (all_log_Ps[i:i + self.encode_len] > 0).sum(0)
                # print(nums)
                # logging.info(nums)

                max_values, max_indices = torch.max(nums, 0)
                top_values, top_indices = torch.topk(nums, 2)
                # top4_values, top4_indices = torch.topk(nums, 4)
                # values = [messages[i] for i in top_indices]
                # print(values,top_values)
                # print(max_indices,top_indices)
                decoded_message = messages[max_indices]
                # print(decoded_message)
                decoded_confidence = int(max_values)
                decoded_probs = float(torch.softmax(nums.float(), dim=-1)[max_indices])
                relative_decoded_confidence = int(max_values) - int(top_values[1])
                decoded_messages.append(decoded_message)
                decoded_confidences.append(
                    (decoded_confidence, relative_decoded_confidence, decoded_probs))
            else:
                nums = (all_log_Ps[i:i + self.encode_len] > 0).sum(0)
                decoded_probs = torch.softmax(nums.float(), dim=-1)
                decoded_prob, decoded_message = decoded_probs.max(0)
                decoded_message = int(decoded_message)
                decoded_messages.append(decoded_message)
                decoded_prob = float(decoded_prob)
                decoded_confidences.append(
                    (-1, -1, decoded_prob))
        decoded_messages = [int(_) for _ in decoded_messages]
        if non_analyze:
            return decoded_messages, (None, decoded_confidences)
        else:
            return decoded_messages, (all_log_Ps.cpu(), decoded_confidences)

    def decode(self, text, messages=None, batch_size=16, disable_tqdm=False, non_analyze=False):
        input_ids = self.message_model.lm_tokenizer(text, return_tensors='pt')['input_ids'].to(
            self.message_model.device)
        return self.decode_with_input_ids(input_ids, messages, batch_size, disable_tqdm,
                                          non_analyze=non_analyze)
