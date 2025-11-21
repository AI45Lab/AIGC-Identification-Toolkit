import warnings
from typing import Union, Optional, List

import numpy as np
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer
import torch
import torch.nn.functional as F
import random

from .base_message_model_fast import BaseMessageModelFast, TextTooShortError
from ....src.utils.hash_fn import Hash1, random_shuffle
from ....src.utils.random_utils import np_temp_random

HfTokenizer = Union[PreTrainedTokenizerFast, PreTrainedTokenizer]
import logging
logging.basicConfig(filename='example.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')
torch.set_printoptions(threshold=np.inf)
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


import hashlib
def calculate_entropy(tensor):
    # Ensure all probabilities are non-zero to avoid log(0)
    tensor = tensor[tensor > 0]
    return -torch.sum(tensor * torch.log(tensor))

def find_best_window_per_row(shuffled_lm_probs, window_size, threshold=0.3):
    adequate_windows = []
    for row in shuffled_lm_probs:
        adequate_window = None

        for i in range(len(row) - window_size + 1):
            window_sum = torch.sum(row[i:i + window_size])

            if window_sum > threshold:
                adequate_window = (i, i + window_size - 1)
                break  # 找到第一个满足条件的窗口后立即停止搜索

        adequate_windows.append(adequate_window)
    
    return adequate_windows

def hash_seed(seed: int, max_value: int) -> int:
    hash_object = hashlib.sha256(str(seed).encode())
    hash_value = int(hash_object.hexdigest(), 16)
    return hash_value % max_value
# def hash_seed(seed: int, max_value: int) -> int:
#     key = seed
#     key = (~key) + (key << 21)
#     key = key ^ (key >> 24)
#     key = (key + (key << 3)) + (key << 8)
#     key = key ^ (key >> 14)
#     key = (key + (key << 2)) + (key << 4)
#     key = key ^ (key >> 28)
#     key = key + (key << 31)

#     key = (key + (key << 3)) + (key << 8)
#     key = key ^ (key >> 14)
#     key = (key + (key << 2)) + (key << 4)
#     key = key ^ (key >> 28)
    
#     return key % max_value

class LMPredictions:
    def __init__(self, public_lm_probs: torch.Tensor, delta, top_k: int = -1, cache_A=True,shifts = [21, 24, 3, 8, 14, 2, 4, 28, 31, 3, 8, 14, 2, 4, 28]):
        '''
        :param public_lm_probs: torch.FloatTensor of shape [batcs_size, vocab_size]
        '''
        assert public_lm_probs.dim() == 2

        self.public_lm_probs = public_lm_probs
        self.top_k = top_k
        if self.top_k > 0:
            raise NotImplementedError
        self.vocab_size = public_lm_probs.shape[-1]
        self.cache_A = cache_A
        self.delta = delta
        self.is_in_As_for_message_count = 0
        self.shifts = shifts

    @property
    def topk_indices(self) -> torch.Tensor:
        if self.top_k == -1:
            raise ValueError("topk is not set")
        if hasattr(self, "_topk_indices"):
            return self._topk_indices
        self._topk_indices = torch.topk(self.public_lm_probs, self.top_k, dim=-1).indices
        return self._topk_indices

    @property
    def topk_public_lm_probs(self) -> torch.Tensor:
        if self.top_k == -1:
            raise ValueError("topk is not set")
        if hasattr(self, "_topk_public_lm_probs"):
            return self._topk_public_lm_probs
        self._topk_public_lm_probs = torch.torch.gather(self.public_lm_probs, -1, self.topk_indices)
        self._topk_public_lm_probs = self._topk_public_lm_probs / self._topk_public_lm_probs.sum(-1,
                                                                                                 keepdim=True)
        return self._topk_public_lm_probs


    def P_message_single_1(self, seeds: torch.Tensor) -> torch.Tensor:
        '''
        :param batch_idx: int
        :param seeds: torch.LongTensor of shape [bsz]

        :return: P_message, torch.FloatTensor of shape [bsz, vocab_size]
        '''
        single_lm_probs = self.topk_public_lm_probs if self.top_k > 0 \
            else self.public_lm_probs

        # print(seeds.shape)
        assert seeds.dim() == 1 and seeds.numel() == single_lm_probs.shape[0]

        shuffle_idxs = random_shuffle(key=seeds,
                                      n=self.top_k if self.top_k > 0 else self.vocab_size,
                                      device=self.public_lm_probs.device)
        # print(shuffle_idxs.shape)
        # shuffle_idxs = custom_shuffle(key=seeds,
        #                               n=self.top_k if self.top_k > 0 else self.vocab_size,
        #                               device=self.public_lm_probs.device)


        if shuffle_idxs.dim() == 1:
            shuffle_idxs = shuffle_idxs.unsqueeze(0)
        shuffled_lm_probs = single_lm_probs.gather(-1, shuffle_idxs)

        # mask = (shuffled_lm_probs.cumsum(-1) < 0.5).to(shuffled_lm_probs.dtype)
        # mask_2 = (shuffled_lm_probs.cumsum(-1) >= 0.5).to(shuffled_lm_probs.dtype)
        # 随机列表
        # noise = -random.choice([0.5, 1.5, 2.5, 3.5])
        # shuffled_lm_probs_with_noise = -(shuffled_lm_probs.cumsum(-1) < 0.5).to(shuffled_lm_probs.dtype) + noise * mask-(shuffled_lm_probs.cumsum(-1) >= 0.5).to(shuffled_lm_probs.dtype) + noise * mask_2
    #     shuffled_lm_probs_with_noise = (
    # (shuffled_lm_probs.cumsum(-1) < 0.5).to(shuffled_lm_probs.dtype) * (-random.choice([1.5, 2.5, 3.5, 4.5, 5.5]))
    # + (shuffled_lm_probs.cumsum(-1) >= 0.5).to(shuffled_lm_probs.dtype) * (random.choice([ 1.5, 2.5, 3.5, 4.5, 5.5])))
       
        #均匀分布
        # noise = np.random.uniform(low=-5, high=-0)
        # shuffled_lm_probs_with_noise = (shuffled_lm_probs.cumsum(-1) < 0.5).to(shuffled_lm_probs.dtype) * noise
        
        #指数分布
        # noise = -np.random.exponential(scale=1.5)
        # shuffled_lm_probs_with_noise = (shuffled_lm_probs.cumsum(-1) < 0.9).to(shuffled_lm_probs.dtype) * noise
        

        # p_message = single_lm_probs.scatter(dim=-1, index=shuffle_idxs, src=shuffled_lm_probs_with_noise)
        shuffled_lm_probs_black_mask = (shuffled_lm_probs.cumsum(-1) >0.3).to(shuffled_lm_probs.dtype) * (self.delta)
        # for i in range((shuffled_lm_probs.cumsum(-1) >0.9).shape[0]):
        
        #     print((shuffled_lm_probs.cumsum(-1) >0.9)[i].sum())
        # # TO DO here we change single_lm_probs in-place!
        # print("下一个")
        p_message = single_lm_probs.scatter_(dim=-1, index=shuffle_idxs,
                                             src=shuffled_lm_probs_black_mask)

        return p_message
    
    def P_message_single_wrong(self, seeds: torch.Tensor) -> torch.Tensor:
        '''
        :param seeds: torch.LongTensor of shape [bsz]

        :return: P_message, torch.FloatTensor of shape [bsz, vocab_size]
        '''
        single_lm_probs = self.topk_public_lm_probs if self.top_k > 0 else self.public_lm_probs

        assert seeds.dim() == 1 and seeds.numel() == single_lm_probs.shape[0]
        
        # print(seeds.shape)

        bsz, vocab_size = single_lm_probs.shape

        top_k = 10
        # Step 1: 选出前1000个概率最大的单词的索引
        topk_probs, topk_indice = torch.topk(single_lm_probs, k=top_k, dim=-1)
        # print(topk_indices)
        topk_indices = topk_indice.flip(-1)

        # Step 2: 创建掩码数组
        mask = torch.zeros((bsz, vocab_size), device=single_lm_probs.device, dtype=torch.bool)
        mask.scatter_(1, topk_indices, True)

        # Step 3: 根据seed通过哈希函数单向确定起始位置
        max_start_position = vocab_size - top_k
        # start_positions = torch.tensor([hash_seed(seed.item(), max_start_position) for seed in seeds], device=single_lm_probs.device)
        start_positions = torch.zeros(bsz, device=single_lm_probs.device, dtype=torch.long)

        for i in range(bsz):
            torch.manual_seed(seeds[i].item())  # 使用每个 seed 来设置随机种子
            start_positions[i] = torch.randint(0, max_start_position, (1,), device=single_lm_probs.device)
        # print(start_positions)
        # Step 4: 创建新的shuffle_idxs数组
        shuffle_idxs = torch.zeros((bsz, vocab_size), device=single_lm_probs.device, dtype=torch.long)
        
        # shuffled_lm_probs_black_mask = torch.zeros((bsz, vocab_size), device=single_lm_probs.device, dtype=torch.bool)


        for i in range(bsz):
            # 获取前10个单词的索引
            topk_range = torch.arange(start_positions[i], start_positions[i] + top_k, device=single_lm_probs.device) % vocab_size
            # print(topk_range)
            shuffle_idxs[i, topk_range] = topk_indices[i]
            # print(shuffle_idxs[i, topk_range])
            # shuffled_lm_probs_black_mask[i, topk_range] = True
            # 获取剩余部分的索引
            remaining_indices = torch.arange(vocab_size, device=single_lm_probs.device)
            remaining_indices = remaining_indices[~mask[i]]
            seeds_i=seeds[i].unsqueeze(0)
            # print(seeds_i.shape)
            # print(remaining_indices.size(0))
            # 使用random_shuffle函数打乱其余的索引
            shuffled_remaining_indices = remaining_indices[random_shuffle(seeds_i, remaining_indices.size(0), single_lm_probs.device)]
            
            # 填充剩余部分的索引
            remaining_range = torch.cat((torch.arange(0, start_positions[i], device=single_lm_probs.device),
                                        torch.arange(start_positions[i] + top_k, vocab_size, device=single_lm_probs.device) % vocab_size))
            # print(remaining_range.shape,shuffled_remaining_indices.shape)
            shuffle_idxs[i, remaining_range] = shuffled_remaining_indices

        # Step 5: 给这10个单词加上偏置
        shuffled_lm_probs = single_lm_probs.gather(-1, shuffle_idxs)
        
        shuffled_lm_probs_black_mask = torch.zeros_like(shuffled_lm_probs, dtype=shuffled_lm_probs.dtype,device=single_lm_probs.device)
        # for i in range(bsz):
        #     # topk_range = torch.arange(start_positions[i], start_positions[i] + 10, device=single_lm_probs.device) % vocab_size
        #     cumulative_probs = shuffled_lm_probs[i, :start_positions[i]+top_k].cumsum(dim=-1)
        #     # print(cumulative_probs)
        #     mask_0_7 = (cumulative_probs < 0.5).to(shuffled_lm_probs.dtype)

        #     shuffled_lm_probs_black_mask[i, :start_positions[i] +top_k] = mask_0_7*(-1.5)
        for i in range(bsz):
            topk_range = torch.arange(0, start_positions[i] + top_k-1, device=single_lm_probs.device) % vocab_size
            shuffled_lm_probs_black_mask[i, topk_range] = -1.5  # 直接给这1000个词的位置加上负偏置
        # print(shuffled_lm_probs_black_mask)
        # shuffled_lm_probs_black_mask = shuffled_lm_probs_black_mask*1.5
        # shuffled_lm_probs_black_mask = (shuffled_lm_probs.cumsum(-1) <0.5).to(shuffled_lm_probs.dtype) * (-2.5)
        # print(shuffled_lm_probs_black_mask_1.shape,shuffled_lm_probs_black_mask.shape)

        # Step 6: 返回处理后的概率分布
        p_message = single_lm_probs.scatter(-1, shuffle_idxs, shuffled_lm_probs_black_mask)

        return p_message
        
    def P_message_single(self, seeds: torch.Tensor) -> torch.Tensor:
        '''
        :param batch_idx: int
        :param seeds: torch.LongTensor of shape [bsz]

        :return: P_message, torch.FloatTensor of shape [bsz, vocab_size]
        '''
        single_lm_probs = self.topk_public_lm_probs if self.top_k > 0 \
            else self.public_lm_probs

        # print(seeds.shape)
        assert seeds.dim() == 1 and seeds.numel() == single_lm_probs.shape[0]

        shuffle_idxs = random_shuffle(key=seeds,
                                      n=self.top_k if self.top_k > 0 else self.vocab_size,
                                      device=self.public_lm_probs.device,shifts=self.shifts)
        # print(shuffle_idxs.shape)
        # shuffle_idxs = custom_shuffle(key=seeds,
        #                               n=self.top_k if self.top_k > 0 else self.vocab_size,
        #                               device=self.public_lm_probs.device)

        if shuffle_idxs.dim() == 1:
            shuffle_idxs = shuffle_idxs.unsqueeze(0)
        shuffled_lm_probs = single_lm_probs.gather(-1, shuffle_idxs)
        # unshuffle_idxs = torch.argsort(shuffle_idxs, dim=-1)

        shuffled_lm_probs_black_mask = ((shuffled_lm_probs.cumsum(-1) > 0.9)).to(shuffled_lm_probs.dtype) * (self.delta)
        
        
        # mask = ((shuffled_lm_probs.cumsum(-1) > 0.9)).to(shuffled_lm_probs.dtype)
        # filtered_probs = shuffled_lm_probs * mask.to(shuffled_lm_probs.dtype)
        # entropy_sum = calculate_entropy(filtered_probs)
        # print(entropy_sum)
        # alpha = 0.5
        
        # # TO DO here we change single_lm_probs in-place!
        # p_message = single_lm_probs.scatter_(dim=-1, index=shuffle_idxs,
        #                                      src=shuffled_lm_probs_black_mask)
        # Create masks for the three intervals
        # mask_1 = (shuffled_lm_probs.cumsum(-1) <= alpha).to(shuffled_lm_probs.dtype)
        # mask_2 = ((shuffled_lm_probs.cumsum(-1) > alpha) & (shuffled_lm_probs <= (1 - alpha))).to(shuffled_lm_probs.dtype)
        # mask_3 = (shuffled_lm_probs.cumsum(-1) > (1 - alpha)).to(shuffled_lm_probs.dtype)

        # Apply the transformations to calculate the bias
        # bias = mask_1 * (-1.5) + mask_2 * (0) + mask_3 * 0

        # Scatter the bias back to the original tensor
        p_message = torch.zeros_like(single_lm_probs).scatter(dim=-1, index=shuffle_idxs, src=shuffled_lm_probs_black_mask)

        return p_message


    def is_in_As_for_message_wrong(self, batch_idx: int, seeds: torch.Tensor,
                             x_cur_idx: int) -> \
            Union[List[torch.Tensor], torch.Tensor]:
        '''
        :param batch_idx: int
        :param seeds: torch.LongTensor of shape [num_seeds]
        :param x_cur_idx: int or None

        :return: [x_cur_id in A for A in As], torch.FloatTensor of shape [vocab_size]
        '''
        # self.is_in_As_for_message_count += 1
        # print(self.is_in_As_for_message_count)
        assert seeds.dim() == 1
        # print(seeds.shape)
        single_lm_probs = self.topk_public_lm_probs[batch_idx:batch_idx + 1] if self.top_k > 0 \
            else self.public_lm_probs[batch_idx:batch_idx + 1]
        
        # print(single_lm_probs)
        # print(torch.topk(single_lm_probs, k=10, dim=-1))
        # shuffle_idxs = random_shuffle(key=seeds,
        #                               n=self.top_k if self.top_k > 0 else self.vocab_size,
        #                               device=self.public_lm_probs.device,)
        # # print(shuffle_idxs.shape)
        # if shuffle_idxs.dim() == 1:
        #     shuffle_idxs = shuffle_idxs.unsqueeze(0)
        # print(shuffle_idxs.shape)
        
        
        bsz = seeds.shape[0]
        vocab_size = single_lm_probs.shape[1]
        
        # print(bsz,vocab_size)

        top_k = 10
        # Step 1: 选出前10个概率最大的单词的索引  topk的索引可以尝试逆向
        topk_indices = torch.topk(single_lm_probs, k=top_k, dim=-1).indices
        # print(topk_indices.shape)

        # Step 2: 创建掩码数组
        mask = torch.zeros((1, vocab_size), device=single_lm_probs.device, dtype=torch.bool)
        mask.scatter_(1, topk_indices, True)
        mask = mask.expand(bsz, -1)
        # print(mask)

        # Step 3: 根据seed通过哈希函数单向确定起始位置
        max_start_position = vocab_size - top_k
        # start_positions = torch.tensor([hash_seed(seed.item(), max_start_position) for seed in seeds], device=single_lm_probs.device)
        start_positions = torch.zeros(bsz, device=single_lm_probs.device, dtype=torch.long)

        for i in range(bsz):
            torch.manual_seed(seeds[i].item())  # 使用每个 seed 来设置随机种子
            start_positions[i] = torch.randint(0, max_start_position, (1,), device=single_lm_probs.device)
            print(start_positions[i])
        # Step 4: 创建新的shuffle_idxs数组
        shuffle_idxs = torch.zeros((bsz, vocab_size), device=single_lm_probs.device, dtype=torch.long)

        for i in range(bsz):
            # 获取前10个单词的索引
            topk_range = torch.arange(start_positions[i], start_positions[i] + top_k, device=single_lm_probs.device) % vocab_size
            shuffle_idxs[i, topk_range] = topk_indices[0]

            # 获取剩余部分的索引
            remaining_indices = torch.arange(vocab_size, device=single_lm_probs.device)
            remaining_indices = remaining_indices[~mask[i]]
            seeds_i=seeds[i].unsqueeze(0)
            # print(remaining_indices.size(0))
            # 使用random_shuffle函数打乱其余的索引
            shuffled_remaining_indices = remaining_indices[random_shuffle(seeds_i, remaining_indices.size(0), single_lm_probs.device)]
            
            # 填充剩余部分的索引
            remaining_range = torch.cat((torch.arange(0, start_positions[i], device=single_lm_probs.device),
                                        torch.arange(start_positions[i] + top_k, vocab_size, device=single_lm_probs.device) % vocab_size))
            # print(remaining_range.shape,shuffled_remaining_indices.shape)
            shuffle_idxs[i, remaining_range] = shuffled_remaining_indices
        

       
        # print(single_lm_probs[0][12345])
        single_lm_probs = single_lm_probs.expand(shuffle_idxs.shape[0], -1)
        shuffled_lm_probs = single_lm_probs.gather(-1, shuffle_idxs)
        # print(x_cur_idx)
        eq_mask = torch.eq(shuffle_idxs, x_cur_idx)
        left_mask = 1 - eq_mask.cumsum(-1) + eq_mask 
        # print(left_mask.shape)
        # print(eq_mask.cumsum(-1))
        # count_ones = torch.sum(left_mask[0] == 1).item()
        # print(count_ones)
        # batch_size, seq_length = left_mask.shape
        # new_mask = torch.zeros_like(left_mask)

        # # # 找到每一行最后一个 1 的位置
        # last_one_pos = (left_mask.cumsum(dim=1) == left_mask.sum(dim=1, keepdim=True)).max(dim=1)[1]

        # # 设置掩码：从最后一个 1 往前 1000 个位置为 1，其余为 0
        # for i in range(batch_size):
        #     end_pos = last_one_pos[i].item()
        #     start_pos = max(0, end_pos - 9999)  # 确保起始位置不小于 0
        #     new_mask[i, start_pos:end_pos + 1] = 1
            # print(new_mask[i].sum())
        
        # print((shuffled_lm_probs * left_mask).sum(-1))
        isinAs = ((shuffled_lm_probs * left_mask).sum(-1) > 0.5)
        # print(isinAs)
        # print(isinAs.sum(dim=-1))
        # print(shuffled_lm_probs[0][new_mask[0]==1])
        # print(len(isinAs))
        # print(isinAs)
            

        # left_mask = 1 - eq_mask.cumsum(-1)
        # isinAs = (shuffled_lm_probs * left_mask).sum(-1) < 0.5
        return isinAs



    def is_in_As_for_message(self, batch_idx: int, seeds: torch.Tensor,
                             x_cur_idx: int) -> \
            Union[List[torch.Tensor], torch.Tensor]:
        '''
        :param batch_idx: int
        :param seeds: torch.LongTensor of shape [num_seeds]
        :param x_cur_idx: int or None

        :return: [x_cur_id in A for A in As], torch.FloatTensor of shape [vocab_size]
        '''
        assert seeds.dim() == 1

        shuffle_idxs = random_shuffle(key=seeds,
                                      n=self.top_k if self.top_k > 0 else self.vocab_size,
                                      device=self.public_lm_probs.device,shifts = self.shifts)
        # print(shuffle_idxs.shape)
        if shuffle_idxs.dim() == 1:
            shuffle_idxs = shuffle_idxs.unsqueeze(0)

        single_lm_probs = self.topk_public_lm_probs[batch_idx:batch_idx + 1] if self.top_k > 0 \
            else self.public_lm_probs[batch_idx:batch_idx + 1]
        # print(single_lm_probs[0][12345])
        single_lm_probs = single_lm_probs.expand(shuffle_idxs.shape[0], -1)
        shuffled_lm_probs = single_lm_probs.gather(-1, shuffle_idxs)
        # print(x_cur_idx)
        eq_mask = torch.eq(shuffle_idxs, x_cur_idx)
        left_mask = 1 - eq_mask.cumsum(-1) + eq_mask 
        # print(eq_mask.cumsum(-1))
        # print(left_mask)
        # print((shuffled_lm_probs * left_mask).sum(-1))

        isinAs = (shuffled_lm_probs * left_mask).sum(-1) >= 0.9
        # print(len(isinAs))

        # left_mask = 1 - eq_mask.cumsum(-1)
        # isinAs = (shuffled_lm_probs * left_mask).sum(-1) < 0.5
        return isinAs

    def clear_cache(self):
        if hasattr(self, "_A_for_message_dict"):
            del self._A_for_message_dict


class SeedCacher:
    def __init__(self):
        self._cache = []
        self._curs = []

    def add(self, seeds):
        self._cache.extend(seeds)

    def add_cur(self,cur):
        self._curs.extend(cur)

    def clear(self):
        self._cache = []
        self._curs = []

    def data(self):
        return self._cache, self._curs


class LMMessageModel(BaseMessageModelFast):
    def __init__(self, tokenizer: HfTokenizer, lm_model, lm_tokenizer: HfTokenizer,
                 delta, lm_prefix_len, seed=42, lm_topk=1000, message_code_len=10,
                 random_permutation_num=100, hash_prefix_len=1, hash_fn=Hash1,shifts = [21, 24, 3, 8, 14, 2, 4, 28, 31, 3, 8, 14, 2, 4, 28]):
        super().__init__(seed, message_code_len)
        self.tokenizer = tokenizer
        self.lm_model = lm_model
        self.lm_tokenizer = lm_tokenizer
        if self.lm_tokenizer.pad_token is None:
            self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token
        self.delta = delta
        self.lm_prefix_len = lm_prefix_len
        self.message_code_len = message_code_len
        self.lm_top_k = lm_topk
        if self.lm_top_k > 0:
            raise NotImplementedError
        self.hash_prefix_len = hash_prefix_len
        self.hash_fn = hash_fn
        self.shifts = shifts
        self.random_permutation_num = random_permutation_num
        self.tokenizer_id_2_lm_tokenizer_id_list = self.construct_tokenizer_id_2_lm_tokenizer_id_list()
        self.random_permutation_idxs = torch.arange(self.random_permutation_num).unsqueeze(0)
        self.max_vocab_size = 100000
        self.seed_cacher = SeedCacher()

    @property
    def random_delta(self):
        return self.delta * 1.

    @property
    def device(self):
        return self.lm_model.device

    def set_random_state(self):
        pass

    def construct_tokenizer_id_2_lm_tokenizer_id_list(self):
        tokenizer_id_2_lm_tokenizer_id_list = torch.zeros(len(self.tokenizer.vocab),
                                                          dtype=torch.long)
        for v, i in self.tokenizer.vocab.items():
            tokenizer_id_2_lm_tokenizer_id_list[i] = self.lm_tokenizer.convert_tokens_to_ids(v)
        # device = self.lm_model.module.device if isinstance(self.lm_model, torch.nn.DataParallel) else self.lm_model.device
        # return tokenizer_id_2_lm_tokenizer_id_list.to(device)
        return tokenizer_id_2_lm_tokenizer_id_list.to(self.lm_model.device)

    def check_x_prefix_x_cur_messages(self, x_prefix: Optional[torch.Tensor] = None,
                                      x_cur: Optional[torch.Tensor] = None,
                                      messages: Optional[torch.Tensor] = None):
        if x_prefix is not None:
            assert x_prefix.dim() == 2
        if x_cur is not None:
            assert x_cur.dim() == 2
        if messages is not None:
            assert messages.dim() == 1
        if x_prefix is not None and x_cur is not None:
            assert x_prefix.shape[0] == x_cur.shape[0]

    def filter(self, texts):
        warnings.warn("filter is Deprecated.")

        filtered_bools = torch.full((len(texts),), False, dtype=torch.bool, device=self.device)
        filtered_texts = []
        for i, text in enumerate(texts):
            tokens = self.lm_tokenizer.tokenize(text, add_special_tokens=False)
            if len(tokens) < self.lm_prefix_len:
                filtered_bools[i] = False
                continue
            else:
                filtered_bools[i] = True
                filtered_texts.append(text)
        return filtered_texts, filtered_bools

    def get_lm_predictions(self, strings, return_log_softmax=False, input_lm_token_ids = False):
        ori_pad_side = self.lm_tokenizer.padding_side
        self.lm_tokenizer.padding_side = 'left'
        encodings = self.lm_tokenizer(strings, return_tensors="pt", padding=True,
                                      add_special_tokens=False)
        self.lm_tokenizer.padding_side = ori_pad_side
        if encodings['input_ids'].shape[1] < self.lm_prefix_len:
            # raise TextTooShortError
            warnings.warn("TextTooShortError")
        encodings['input_ids'] = encodings['input_ids'][:, -self.lm_prefix_len:]
        encodings['attention_mask'] = encodings['attention_mask'][:, -self.lm_prefix_len:]
        encodings = encodings.to(self.lm_model.device)
        # encodings = {key: value.to('cuda') for key, value in encodings.items()}
        
        # Filter out unsupported parameters for LLaMA models
        filtered_encodings = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }
        
        with torch.no_grad():
            logits = self.lm_model(**filtered_encodings).logits
        final_pos = (encodings['input_ids'] != self.lm_tokenizer.pad_token_id).sum(dim=1) - 1
        logits = logits[torch.arange(logits.shape[0]), final_pos]
        if return_log_softmax:
            logit_or_probs = torch.log_softmax(logits, dim=-1)
        else:
            logit_or_probs = torch.softmax(logits, dim=-1)

        return logit_or_probs

    def get_x_prefix_seed_element(self, x_prefix, transform: bool):
        self.check_x_prefix_x_cur_messages(x_prefix=x_prefix)
        if x_prefix.shape[1] < self.hash_prefix_len:
            raise TextTooShortError
        x_prefix = x_prefix[:, -self.hash_prefix_len:]

        if transform:
            x_prefix = self.tokenizer_id_2_lm_tokenizer_id_list[x_prefix]
        seed_elements = [tuple(_.tolist()) for _ in x_prefix]

        return seed_elements


    def get_x_prefix_seeds(self, x_prefix, transform: bool):
        '''
        :param x_prefix: [batch_size, seq_len]

        :return: seeds: list of [batch_size]
        '''
        self.check_x_prefix_x_cur_messages(x_prefix=x_prefix)
        seed_elements = self.get_x_prefix_seed_element(x_prefix, transform=transform)
        seeds = torch.tensor([hash(seed_element + (self.seed,)) for seed_element in seed_elements],
                             device=x_prefix.device)

        self.seed_cacher.add(seeds.tolist())

        return seeds

    def get_x_prefix_A_seeds(self, x_prefix_seeds: torch.Tensor,
                             random_permutation_idxs: torch.Tensor = None):
        '''
        :param x_prefix: [batch_size, seq_len]
        :param x_prefix_seeds: [batch_size]
        :param random_permutation_idxs: [batch_size, random_permutation_num] or [1, random_permutation_num]

        :return: seeds: list of [batch_size, self.random_permutation_num]
        '''


        assert x_prefix_seeds.dim() == 1
        if random_permutation_idxs is not None:
            assert random_permutation_idxs.dim() == 2
            assert random_permutation_idxs.shape[0] == 1 or x_prefix_seeds.shape[0] == \
                   random_permutation_idxs.shape[0]

        if random_permutation_idxs is None:
            self.random_permutation_idxs = self.random_permutation_idxs.to(x_prefix_seeds.device)
            random_permutation_idxs = self.random_permutation_idxs
        else:
            random_permutation_idxs = random_permutation_idxs.to(x_prefix_seeds.device)
        seeds = self.hash_fn(
            x_prefix_seeds.unsqueeze(1) * self.random_permutation_num + random_permutation_idxs,self.shifts)
        return seeds

    def get_x_prefix_and_message_seeds(self, messages: torch.Tensor,
                                       x_prefix_seeds: torch.Tensor):
        '''
        :param message_or_messages: [message_num]

        :return: [batch_size, message_num]
        '''
        seeds = (x_prefix_seeds % (2 ** (32 - self.message_code_len))).unsqueeze(1)   
        seeds = seeds * (2 ** self.message_code_len) + messages.unsqueeze(0)
        seeds = self.hash_fn(seeds, self.shifts)
        # print(seeds.shape)
        return seeds

    def get_x_prefix_and_message_and_x_cur_seeds(self, x_prefix: torch.Tensor,
                                                 messages: torch.Tensor, x_cur: torch.Tensor,
                                                 x_prefix_seeds: torch.Tensor,
                                                 x_prefix_and_message_seeds: torch.Tensor = None):
        '''
        :param x_prefix: [batch_size, seq_len]
        :param message_or_messages: [message_num]
        :param x_cur: [batch_size, vocab_size]

        return: [batch_size, message_num, vocab_size]
        '''
        assert x_cur.dim() == 2 and x_prefix.dim() == 2
        if x_prefix_and_message_seeds is None:
            x_prefix_and_message_seeds = self.get_x_prefix_and_message_seeds(messages,
                                                                             x_prefix_seeds)
        x_prefix_and_message_and_x_cur_seeds = self.hash_fn(x_prefix_and_message_seeds.unsqueeze(
            -1) * self.max_vocab_size + x_cur.unsqueeze(1))
        return x_prefix_and_message_and_x_cur_seeds

    def cal_log_P_with_single_message(self, x_prefix: torch.LongTensor, x_cur: torch.Tensor,
                                      messages: torch.Tensor,
                                      lm_predictions: Optional[torch.Tensor],
                                      transform_x_cur: bool = True) -> torch.Tensor:
        '''
        :param x_prefix: [batch, seq_len]
        :param x_cur: [batch, vocab_size]
        :param messages: [message_num]
        :param lm_predictions: [batch, all_vocab_size]

        :return: log of P of size [batch, messages, vocab_size]
        '''
        assert messages.numel() == 1
        self.check_x_prefix_x_cur_messages(x_prefix=x_prefix, x_cur=x_cur, messages=messages)

        if transform_x_cur:
            x_cur = self.tokenizer_id_2_lm_tokenizer_id_list[x_cur]

        x_prefix_seeds = self.get_x_prefix_seeds(x_prefix, transform=True)
        # print(x_prefix_seeds.shape)
        # x_prefix_seeds = torch.tensor([5768399737913331976],device=x_prefix.device)
        
        x_prefix_and_message_seeds = self.get_x_prefix_and_message_seeds(
            messages,
            x_prefix_seeds=x_prefix_seeds)
        
        # print(x_prefix_and_message_seeds)


        lm_predictions = LMPredictions(lm_predictions, delta=self.delta, top_k=self.lm_top_k,
                                       cache_A=False, shifts=self.shifts)
        x_prefix_and_message_idxs = x_prefix_and_message_seeds % self.random_permutation_num
        # print(x_prefix_and_message_idxs.shape)
        x_prefix_and_A_seeds = self.get_x_prefix_A_seeds(x_prefix_seeds=x_prefix_seeds,
                                                         random_permutation_idxs=x_prefix_and_message_idxs)

        # print(x_prefix_and_A_seeds)
        # x_prefix_and_message_and_x_cur_seeds = self.get_x_prefix_and_message_and_x_cur_seeds(
        #     x_prefix, messages, x_cur, x_prefix_seeds=x_prefix_seeds,
        #     x_prefix_and_message_seeds=x_prefix_and_message_seeds)
        # x_prefix_and_message_and_x_cur_idxs = x_prefix_and_message_and_x_cur_seeds % 2

        if self.lm_top_k > 0:
            # log_Ps = x_prefix_and_message_and_x_cur_idxs.float() * self.random_delta
            raise NotImplementedError
        else:
            assert x_prefix_and_A_seeds.shape[1] == 1
            all_log_Ps = lm_predictions.P_message_single(seeds=x_prefix_and_A_seeds[:, 0])
            # print(all_log_Ps[1, :10])
            log_Ps = all_log_Ps.gather(dim=1, index=x_cur)
            log_Ps = log_Ps.unsqueeze(1)
            # log_Ps = torch.zeros((x_prefix.shape[0], 1, x_cur.shape[1]), device=x_cur.device)
            # for batch_idx in range(x_prefix.shape[0]):
            #     As = lm_predictions.As_for_message(batch_idx=batch_idx,
            #                                        seeds=x_prefix_and_A_seeds[batch_idx])
            #     is_inA = torch.isin(x_cur[batch_idx], As[0])
            #     log_Ps[batch_idx, 0, is_inA] = self.delta

        return log_Ps

    def cal_log_P_with_single_x_cur(self, x_prefix: torch.LongTensor, x_cur: torch.Tensor,
                                    messages: torch.Tensor,
                                    lm_predictions: Optional[torch.Tensor]):
        '''
        :param x_prefix: [batch, seq_len]
        :param x_cur: [batch, vocab_size]
        :param messages: [message_num] tensor([   0,    1,    2,  ..., 1021, 1022, 1023], device='cuda:0')
        :param lm_predictions: [batch, all_vocab_size]

        :return: log of P of size [batch, messages, vocab_size]
        '''
        assert x_cur.shape[1] == 1
        # print("该阶段message",messages)
        x_prefix_seeds = self.get_x_prefix_seeds(x_prefix, transform=False)
        # x_prefix_seeds = torch.tensor([5768399737913331976],device=x_prefix.device)


        lm_predictions = LMPredictions(lm_predictions, delta=self.delta, top_k=self.lm_top_k,  #self.lm_top_k=-1
                                       cache_A=False, shifts=self.shifts)
        x_prefix_and_A_seeds = self.get_x_prefix_A_seeds(x_prefix_seeds=x_prefix_seeds)
        x_prefix_and_message_seeds = self.get_x_prefix_and_message_seeds(messages,
                                                                         x_prefix_seeds=x_prefix_seeds)
        x_prefix_and_message_idxs = x_prefix_and_message_seeds % self.random_permutation_num
        log_Ps = []
        for batch_idx in range(x_prefix.shape[0]):
            if self.lm_top_k < 0:
                x_cur_idx = int(x_cur[batch_idx, 0])
            else:
                x_cur_idx = \
                    torch.where(lm_predictions.topk_indices[batch_idx] == x_cur[batch_idx, 0])[0]
                if len(x_cur_idx) > 0:
                    x_cur_idx = int(x_cur_idx[0])
                else:
                    x_cur_idx = None
            if x_cur_idx is not None:
                in_A_s = lm_predictions.is_in_As_for_message(batch_idx=batch_idx,
                                                             seeds=x_prefix_and_A_seeds[batch_idx],
                                                             x_cur_idx=x_cur_idx)
                # print(in_A_s)
                # print((x_prefix_and_A_seeds[batch_idx]>0).sum())
                is_in_certain_A = torch.gather(in_A_s, 0, x_prefix_and_message_idxs[batch_idx])
                # print(is_in_certain_A.sum())
                single_log_Ps = is_in_certain_A.float() * 0.01  #torch.Size([1024])
                # print(single_log_Ps.shape)
                single_log_Ps = single_log_Ps.reshape(1, -1, 1)   #torch.Size([1, 1024, 1])
                # print(single_log_Ps.shape)

                # print('in top_k')
            else:
                raise NotImplementedError
                x_prefix_and_message_and_x_cur_seeds = self.get_x_prefix_and_message_and_x_cur_seeds(
                    x_prefix[batch_idx:batch_idx + 1], messages,
                    x_cur[batch_idx:batch_idx + 1],
                    x_prefix_seeds=x_prefix_seeds[batch_idx:batch_idx + 1])
                x_prefix_and_message_and_x_cur_idxs = x_prefix_and_message_and_x_cur_seeds % 2
                single_log_Ps = x_prefix_and_message_and_x_cur_idxs.float() * self.random_delta
                # print(single_log_Ps.shape)
            log_Ps.append(single_log_Ps)
        log_Ps = torch.cat(log_Ps, dim=0)
        return log_Ps

    def cal_log_Ps(self, x_prefix: torch.LongTensor, x_cur: torch.Tensor,
                   messages: torch.Tensor,
                   lm_predictions: torch.Tensor):
        '''
        :param x_prefix: [batch, seq_len]
        :param x_cur: [batch, vocab_size]
        :param messages: [message_num]
        :param lm_predictions: [batch, all_vocab_size]

        :return: log of P of size [batch, messages, vocab_size]
        '''

        # check input shape
        self.check_x_prefix_x_cur_messages(x_prefix, x_cur, messages)
        assert x_prefix.shape[0] == lm_predictions.shape[0]

        if messages.numel() == 1:
            # cal P(x_prefix, x_cur, message) for all possible x_cur
            log_Ps = self.cal_log_P_with_single_message(x_prefix, x_cur,
                                                        messages,
                                                        lm_predictions)
            # print("函数1,%s" % x_cur.shape[1])

        elif x_cur.shape[1] == 1:  ###in max_confidence phase
            # print(messages)
            log_Ps = self.cal_log_P_with_single_x_cur(x_prefix, x_cur, messages, lm_predictions)
            # print('函数2,%s'% x_cur.shape[1])
        else:
            raise NotImplementedError

        return log_Ps
