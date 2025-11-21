import torch
from transformers import LogitsProcessor

class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty >= 0):
            raise ValueError(f"`penalty` has to be a non-negasitive float, but is {penalty}")

        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        score = torch.gather(scores, 1, input_ids)

        score = score - self.penalty

        scores.scatter_(1, input_ids, score)
        return scores
