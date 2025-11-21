import os
import numpy as np
from scipy.optimize import minimize
from scipy.stats import binom
from tqdm import tqdm


def logit_mean(decode):
    watermark_gt = np.array([0, 1] * 48)
    decoded = np.mean(decode, axis=0)
    bitacc = np.sum((watermark_gt)==(decoded >= 0)) / len(watermark_gt)
    return bitacc >= 67/96


def logit_g_median(decode):
    watermark_gt = np.array([0, 1] * 48)
    def objective(median):
        return np.sum(np.linalg.norm(decode - median, axis=1))
    initial_guess = np.mean(decode, axis=0)
    decoded = minimize(objective, initial_guess, method='Powell', tol=1e-5)
    bitacc = np.sum((watermark_gt) == (decoded.x >= 0)) / len(watermark_gt)
    return bitacc >= 67/96


def bit_majority(decode):
    watermark_gt = np.array([0, 1] * 48)
    decode = (decode >= 0)
    decoded = np.sum(decode, axis=0) > (len(decode) / 2)
    bitacc = np.sum((watermark_gt) == decoded) / len(watermark_gt)
    return bitacc >= 67/96


def bit_g_median(decode):
    watermark_gt = np.array([0, 1] * 48)
    decode[decode >= 0] = 1
    decode[decode < 0] = 0
    decoded = np.median(decode, axis=0)
    bitacc = np.sum((watermark_gt) == decoded >= 0) / len(watermark_gt)
    return bitacc >= 67/96


def bitacc_mean(decode):
    watermark_gt = np.array([0, 1] * 48)
    watermark_gt = np.tile(watermark_gt, (decode.shape[0], 1))
    bitacc = np.mean((watermark_gt) == (decode >= 0))
    return bitacc >= 67/96


def bitacc_median(decode):
    watermark_gt = np.array([0, 1] * 48)
    bitacc = np.mean((decode >= 0) == watermark_gt, axis=1)
    median_bitacc = np.median(bitacc)
    return median_bitacc >= 67/96


def detection_threshold(decode):
    watermark_gt = np.array([0, 1] * 48)
    bitacc = np.mean((decode >= 0) == watermark_gt, axis=1)
    detected_frames = np.sum(bitacc >= 67/96)

    n = decode.shape[0]
    p = 0.0000661
    threshold = 10e-4
    for i in range(n + 1):
        if binom.sf(i - 1, n, p) <= threshold:
            k = i
            break

    return detected_frames >= k


def detection_majority(decode):
    watermark_gt = np.array([0, 1] * 48)
    bitacc = np.mean((decode >= 0) == watermark_gt, axis=1)
    detected_frames = np.sum(bitacc >= 67 / 96)

    return detected_frames > (decode.shape[0] // 2)