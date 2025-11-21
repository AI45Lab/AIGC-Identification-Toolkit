from foolbox.attacks.blended_noise import LinearSearchBlendedUniformNoiseAttack
from foolbox.attacks.base import MinimizationAttack, get_criterion
import sys
import torch_dct
from attack_utils import *
import time

global device


# initialize an adversarial example with uniform noise
def get_x_adv(x_o: torch.Tensor, label: torch.Tensor, model) -> torch.Tensor:
    criterion = get_criterion(label)
    init_attack: MinimizationAttack = LinearSearchBlendedUniformNoiseAttack(steps=100)
    x_adv = init_attack.run(model, x_o, criterion)
    return x_adv


# coompute the difference
def get_difference(x_o: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
    difference = x_adv - x_o
    if torch.norm(difference, p=2) == 0:
        raise ('difference is zero vector!')
        return difference
    return difference


def rotate_in_2d(x_o2x_adv: torch.Tensor, direction: torch.Tensor, theta: float = np.pi / 8) -> torch.Tensor:
    alpha = torch.sum(x_o2x_adv * direction) / torch.sum(x_o2x_adv * x_o2x_adv)
    orthogonal = direction - alpha * x_o2x_adv
    direction_theta = x_o2x_adv * np.cos(theta) + torch.norm(x_o2x_adv, p=2) / torch.norm(orthogonal,
                                                                                          p=2) * orthogonal * np.sin(
        theta)
    direction_theta = direction_theta / torch.norm(direction_theta) * torch.norm(x_o2x_adv)
    return direction_theta


# obtain the mask in the low frequency
def get_orthogonal_1d_in_subspace(args,x_o2x_adv: torch.Tensor, n, ratio_size_mask=0.3, if_left=1) -> torch.Tensor:
    random.seed(time.time())
    zero_mask = torch.zeros(size=[args.H, args.W], device=device)
    size_mask = int(args.W * ratio_size_mask)
    if if_left:
        zero_mask[:size_mask, :size_mask] = 1

    else:
        zero_mask[-size_mask:, -size_mask:] = 1

    to_choose = torch.where(zero_mask == 1)
    x = to_choose[0]
    y = to_choose[1]

    masks_per_frame = []
    for _ in range(x_o2x_adv.shape[1]//3):
        # --- mask1 ---
        select = np.random.choice(len(x), size=n, replace=False)
        mask1 = torch.zeros_like(zero_mask)
        mask1[x[select], y[select]] = 1
        # --- mask2 ---
        select = np.random.choice(len(x), size=n, replace=False)
        mask2 = torch.zeros_like(zero_mask)
        mask2[x[select], y[select]] = 1
        # --- mask3 ---
        select = np.random.choice(len(x), size=n, replace=False)
        mask3 = torch.zeros_like(zero_mask)
        mask3[x[select], y[select]] = 1

        masks_per_frame.append(torch.stack([mask1, mask2, mask3], dim=0))  # (3,H,W)
    mask = torch.cat(masks_per_frame, dim=0).unsqueeze(0)

    mask *= torch.randn_like(mask, device=device)
    direction = rotate_in_2d(x_o2x_adv, mask, theta=np.pi / 2)
    return direction / torch.norm(direction, p=2) * torch.norm(x_o2x_adv, p=2), mask


# compute the best adversarial example in the surface
def get_x_hat_in_2d(args, x_o: torch.Tensor, x_adv: torch.Tensor, axis_unit1: torch.Tensor, axis_unit2: torch.Tensor,
                    net: torch.nn.Module, queries, original_label, max_iter=2,plus_learning_rate=0.01,minus_learning_rate=0.0005,half_range=0.1, init_alpha = np.pi/2):
    if not hasattr(get_x_hat_in_2d, 'alpha'):
        get_x_hat_in_2d.alpha = init_alpha
    upper = np.pi / 2 + half_range
    lower = np.pi / 2 - half_range

    d = torch.norm(x_adv - x_o, p=2)

    theta = max(np.pi - 2 * get_x_hat_in_2d.alpha, 0) + min(np.pi / 16, get_x_hat_in_2d.alpha / 2)
    x_hat = torch_dct.idct_2d(x_adv)
    right_theta = np.pi - get_x_hat_in_2d.alpha
    x = x_o + d * (axis_unit1 * np.cos(theta) + axis_unit2 * np.sin(theta)) / np.sin(get_x_hat_in_2d.alpha) * np.sin(
        get_x_hat_in_2d.alpha + theta)
    x = torch_dct.idct_2d(x)
    get_x_hat_in_2d.total += 1
    get_x_hat_in_2d.clamp += torch.sum(x > 1) + torch.sum(x < 0)
    x = torch.clamp(x, 0, 1)
    label = net(x, args.aggregation)
    queries += 1
    if label != original_label:
        x_hat = x
        left_theta = theta
        flag = 1
    else:

        get_x_hat_in_2d.alpha -= minus_learning_rate
        get_x_hat_in_2d.alpha = max(lower, get_x_hat_in_2d.alpha)
        theta = max(theta, np.pi - 2 * get_x_hat_in_2d.alpha + np.pi / 64)

        x = x_o + d * (axis_unit1 * np.cos(theta) - axis_unit2 * np.sin(theta)) / np.sin(
            get_x_hat_in_2d.alpha) * np.sin(
            get_x_hat_in_2d.alpha + theta)  # * mask
        x = torch_dct.idct_2d(x)
        get_x_hat_in_2d.total += 1
        get_x_hat_in_2d.clamp += torch.sum(x > 1) + torch.sum(x < 0)
        x = torch.clamp(x, 0, 1)
        label = net(x, args.aggregation)
        queries += 1
        if label != original_label:
            x_hat = x
            left_theta = theta
            flag = -1
        else:
            get_x_hat_in_2d.alpha -= minus_learning_rate
            get_x_hat_in_2d.alpha = max(get_x_hat_in_2d.alpha, lower)
            return x_hat, queries, False

    # binary search for beta
    theta = (left_theta + right_theta) / 2
    for i in range(max_iter):
        x = x_o + d * (axis_unit1 * np.cos(theta) + flag * axis_unit2 * np.sin(theta)) / np.sin(
            get_x_hat_in_2d.alpha) * np.sin(
            get_x_hat_in_2d.alpha + theta)
        x = torch_dct.idct_2d(x)
        get_x_hat_in_2d.total += 1
        get_x_hat_in_2d.clamp += torch.sum(x > 1) + torch.sum(x < 0)
        x = torch.clamp(x, 0, 1)
        label = net(x, args.aggregation)
        queries += 1
        if label != original_label:
            left_theta = theta
            x_hat = x
            get_x_hat_in_2d.alpha += plus_learning_rate
            return x_hat, queries, True
        else:

            get_x_hat_in_2d.alpha -= minus_learning_rate
            get_x_hat_in_2d.alpha = max(lower, get_x_hat_in_2d.alpha)
            theta = max(theta, np.pi - 2 * get_x_hat_in_2d.alpha + np.pi / 64)

            flag = -flag
            x = x_o + d * (axis_unit1 * np.cos(theta) + flag * axis_unit2 * np.sin(theta)) / np.sin(
                get_x_hat_in_2d.alpha) * np.sin(
                get_x_hat_in_2d.alpha + theta)
            x = torch_dct.idct_2d(x)
            get_x_hat_in_2d.total += 1
            get_x_hat_in_2d.clamp += torch.sum(x > 1) + torch.sum(x < 0)
            x = torch.clamp(x, 0, 1)
            label = net(x, args.aggregation)
            queries += 1
            if label != original_label:
                left_theta = theta
                x_hat = x
                get_x_hat_in_2d.alpha += plus_learning_rate
                get_x_hat_in_2d.alpha = min(upper, get_x_hat_in_2d.alpha)
                return x_hat, queries, True
            else:
                get_x_hat_in_2d.alpha -= minus_learning_rate
                get_x_hat_in_2d.alpha = max(lower, get_x_hat_in_2d.alpha)
                left_theta = max(np.pi - 2 * get_x_hat_in_2d.alpha, 0) + min(np.pi / 16, get_x_hat_in_2d.alpha / 2)
                right_theta = theta
        theta = (left_theta + right_theta) / 2
    get_x_hat_in_2d.alpha += plus_learning_rate
    get_x_hat_in_2d.alpha = min(upper, get_x_hat_in_2d.alpha)
    return x_hat, queries, True


def get_x_hat_arbitary(args,x_o: torch.Tensor, net: torch.nn.Module, original_label, init_x=None,dim_num=5):
    if net(x_o, args.aggregation) != original_label:
        return x_o, 1001, [[0, 0.], [1001, 0.]]
    x_adv = init_x
    x_hat = x_adv
    queries = 0.
    save = 0
    l2 = []
    linf = []

    while queries < args.max_queries :

        x_o2x_adv = torch_dct.dct_2d(get_difference(x_o, x_adv))
        axis_unit1 = x_o2x_adv / torch.norm(x_o2x_adv)
        direction, mask = get_orthogonal_1d_in_subspace(args,x_o2x_adv, dim_num, args.ratio_mask, args.dim_num)
        axis_unit2 = direction / torch.norm(direction)
        x_hat, queries, changed = get_x_hat_in_2d(args, torch_dct.dct_2d(x_o), torch_dct.dct_2d(x_adv), axis_unit1,
                                                  axis_unit2, net, queries, original_label, max_iter=args.max_iter_num_in_2d,plus_learning_rate=args.plus_learning_rate,minus_learning_rate=args.minus_learning_rate,half_range=args.half_range, init_alpha=args.init_alpha)
        x_adv = x_hat

        if queries >= save:
            l2.append(torch.norm(x_adv-x_o, p=2).item())
            linf.append(torch.norm(x_hat-x_o, p=float('inf')).item())
            save += 50
        if queries >= args.max_queries:
            break
    return x_hat, queries, l2, linf


class TA:
    def __init__(self, model, input_device):
        self.net = model
        global device
        device = input_device

    def attack_video(self, args, inputs):
        get_x_hat_in_2d.alpha = np.pi / 2
        get_x_hat_in_2d.total = 0
        get_x_hat_in_2d.clamp = 0

        def random_init(x, model, original_label, max_trials=500, sigma=0.1):
            ### Removal
            if args.type == 'removal':
                seed = x
                for _ in range(max_trials):
                    noise = torch.randn_like(x) * sigma
                    seed = torch.clamp(seed + noise, 0, 1)
                    preds = model(seed, args.aggregation)
                    print(preds, original_label)
                    if preds != original_label:
                        return seed
                raise RuntimeError("failed to find adversarial seed")

            ### Forgery
            elif args.type == 'forgery':
                ### VideoSeal
                if args.model == 'videoseal':
                    x = x.reshape(x.shape[1] // 3, 3, x.shape[2], x.shape[3])
                    seed = torch.rand_like(x)
                    seed = model.embed(seed, is_video=True)["imgs_w"]
                    seed = seed.reshape(1, seed.shape[0] * seed.shape[1], seed.shape[2], seed.shape[3])

                ### StegaStamp
                elif args.model == 'stegastamp':
                    import StegaStamp
                    x = x.reshape(x.shape[1] // 3, 3, x.shape[2], x.shape[3])
                    encoder = model.StegaStampEncoder(secret_size=32, image_size=512)
                    encoder.load_state_dict(torch.load('./model_adversarial/encoder.pth'))
                    encoder = encoder.to(device)
                    seed = torch.rand_like(x)
                    watermark = torch.tensor([1,0]*16).float().to(device)
                    m = watermark.repeat(seed.shape[0], 1)
                    seed = encoder((m, seed)) + seed
                    seed = seed.reshape(1, seed.shape[0] * seed.shape[1], seed.shape[2], seed.shape[3])

                ### REVMark
                elif args.model == 'revmark':
                    from REVMark import Encoder, framenorm
                    x = x.reshape(x.shape[1] // 24, 3, 8, x.shape[2], x.shape[3])
                    encoder = Encoder(96, [8, 128, 128]).to(device).eval()
                    encoder.load_state_dict(torch.load('checkpoints/Encoder.pth'))
                    encoder.tasblock.enable = True
                    watermark = torch.tensor([1, 0] * 48).float().to(device)
                    cover = torch.rand_like(x) * 2 - 1
                    m = watermark.repeat(cover.shape[0], 1)
                    r = encoder(cover, m)
                    seed = (cover + 6.2 * framenorm(r)).clamp(-1, 1)
                    seed = (seed + 1) / 2
                    seed = seed.reshape(1, seed.shape[0] * seed.shape[1] * seed.shape[2], seed.shape[3], seed.shape[4])

                return seed


        x_reshape = inputs.reshape(1, inputs.shape[0] * inputs.shape[1], inputs.shape[2], inputs.shape[3])
        x_init = random_init(x_reshape, self.net, True)

        x_adv, query, l2, linf = get_x_hat_arbitary(args, x_reshape, self.net,
                                                    (args.type == 'removal'),  # True for removal
                                                    init_x=x_init, dim_num=args.dim_num)

        l2 = np.array(l2)
        linf = np.array(linf)
        l2[0] = torch.norm(x_init-x_reshape, p=2).item()
        linf[0] = torch.norm(x_init - x_reshape, p=float('inf')).item()
        return x_adv, query, l2, linf