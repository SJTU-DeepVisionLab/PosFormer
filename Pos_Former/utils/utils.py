from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from Pos_Former.datamodule import vocab
from einops import rearrange
from torch import LongTensor
from torchmetrics import Metric


class Hypothesis:
    seq: List[int]
    score: float

    def __init__(
        self,
        seq_tensor: LongTensor,
        score: float,
        direction: str,
    ) -> None:
        assert direction in {"l2r", "r2l"}
        raw_seq = seq_tensor.tolist()

        if direction == "r2l":
            result = raw_seq[::-1]
        else:
            result = raw_seq

        self.seq = result
        self.score = score

    def __len__(self):
        if len(self.seq) != 0:
            return len(self.seq)
        else:
            return 1

    def __str__(self):
        return f"seq: {self.seq}, score: {self.score}"


class ExpRateRecorder(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total_line", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rec", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, indices_hat: List[List[int]], indices: List[List[int]]):
        for pred, truth in zip(indices_hat, indices):
            pred = vocab.indices2label(pred)
            truth = vocab.indices2label(truth)

            is_same = pred == truth

            if is_same:
                self.rec += 1

            self.total_line += 1

    def compute(self) -> float:
        exp_rate = self.rec / self.total_line
        return exp_rate


def ce_loss(
    output_hat: torch.Tensor,
    output: torch.Tensor,
    ignore_idx: int = vocab.PAD_IDX,
    reduction: str = "mean",
) -> torch.Tensor:
    """comput cross-entropy loss

    Args:
        output_hat (torch.Tensor): [batch, len, e]
        output (torch.Tensor): [batch, len]
        ignore_idx (int):

    Returns:
        torch.Tensor: loss value
    """
    flat_hat = rearrange(output_hat, "b l e -> (b l) e")
    flat = rearrange(output, "b l -> (b l)")
    loss = F.cross_entropy(flat_hat, flat, ignore_index=ignore_idx, reduction=reduction)
    return loss 

# def ce_loss_muti(
#     output_hat1: torch.Tensor,
#     output1: torch.Tensor,
#     ignore_idx: int = 0,
#     reduction: str = "mean",
# ) -> torch.Tensor:
#     """comput cross-entropy loss

#     Args:
#         output_hat (torch.Tensor): [batch, len, e]
#         output (torch.Tensor): [batch, len]
#         ignore_idx (int):

#     Returns:
#         torch.Tensor: loss value
#     """
#     h_size=output_hat1.size(0)
#     for i in range(h_size):
#         flat_hat = rearrange(output_hat1[i], "b l e -> (b l) e")
#         flat = rearrange(output1[:,:,i], "b l -> (b l)")
#         loss_i= F.cross_entropy(flat_hat, flat, ignore_index=ignore_idx, reduction=reduction).unsqueeze(0)
#         if i==0:
#             loss_tensor=loss_i
#         else:
#             loss_tensor=torch.cat((loss_tensor,loss_i),dim=0)
#     # loss = torch.mean(loss_tensor)
#     return loss_tensor 
def ce_loss_all(
    output_hat: torch.Tensor,
    output: torch.Tensor,
    output_hat_layer: torch.Tensor,
    output_layer: torch.Tensor,
    output_hat_pos: torch.Tensor,
    output_pos: torch.Tensor,
    ignore_idx: int = vocab.PAD_IDX,
    reduction: str = "mean",
) -> Tuple[torch.Tensor,torch.Tensor]:
    """comput cross-entropy loss

    Args:
        output_hat (torch.Tensor): [batch, len, e]
        output (torch.Tensor): [batch, len]
        ignore_idx (int):

    Returns:
        torch.Tensor: loss value
    """
    flat_hat = rearrange(output_hat, "b l e -> (b l) e")
    flat = rearrange(output, "b l -> (b l)")
    loss = F.cross_entropy(flat_hat, flat, ignore_index=ignore_idx, reduction=reduction)
    flag = flat != ignore_idx
    
    flat_hat_layer = rearrange(output_hat_layer, "b l e -> (b l) e")
    flat_layer = rearrange(output_layer, "b l -> (b l)")
    loss_layer = F.cross_entropy(flat_hat_layer, flat_layer, reduction='none')
    loss_layer = loss_layer[flag].mean()

    flat_hat_pos = rearrange(output_hat_pos, "b l e -> (b l) e")
    flat_pos = rearrange(output_pos, "b l -> (b l)")
    loss_pos = F.cross_entropy(flat_hat_pos, flat_pos, reduction='none')
    loss_pos = loss_pos[flag].mean()
    # n = flat_layer[flag].size(0)
    # i = (flat_layer[flag] == 0).sum()  # Count the number of zero elements in each sample
    #
    # weight = torch.where(flat_layer[flag] == 0, n / (10 * n - 9 * i), 10 * n / (10 * n - 9 * i))
    # weighted_loss_layer = loss_layer[flag] * weight  # Apply weighting to loss_layer
    # loss_layer = weighted_loss_layer.mean()
    # # loss_layer=loss_layer[flag].mean()
    #
    # flat_hat_pos = rearrange(output_hat_pos, "b l e -> (b l) e")
    # flat_pos = rearrange(output_pos, "b l -> (b l)")
    # loss_pos = F.cross_entropy(flat_hat_pos, flat_pos, reduction='none')
    # loss_pos=loss_pos[flag]* weight
    # loss_pos=loss_pos.mean()
    return loss , loss_layer , loss_pos
# def ce_loss_all(
#     output_hat: torch.Tensor,
#     output: torch.Tensor,
#     output_hat_pos: torch.Tensor,
#     output_pos: torch.Tensor,
#     ignore_idx: int = vocab.PAD_IDX,
#     reduction: str = "mean",
# ) -> Tuple[torch.Tensor,torch.Tensor]:
#     """comput cross-entropy loss

#     Args:
#         output_hat (torch.Tensor): [batch, len, e]
#         output (torch.Tensor): [batch, len]
#         ignore_idx (int):

#     Returns:
#         torch.Tensor: loss value
#     """
#     flat_hat = rearrange(output_hat, "b l e -> (b l) e")
#     flat = rearrange(output, "b l -> (b l)")
#     loss = F.cross_entropy(flat_hat, flat, ignore_index=ignore_idx, reduction=reduction)

#     flat_hat_pos = rearrange(output_hat_pos[:,:-1,:], "b l e -> (b l) e")
#     flat_pos = rearrange(output_pos[:, 1:], "b l -> (b l)")
#     loss_pos = F.cross_entropy(flat_hat_pos, flat_pos, reduction='mean', ignore_index=ignore_idx)
#     return loss, loss_pos

def to_tgt_output(
    tokens: Union[List[List[int]], List[LongTensor]],
    direction: str,
    device: torch.device,
    pad_to_len: Optional[int] = None,
) -> Tuple[LongTensor, LongTensor]:
    """Generate tgt and out for indices

    Parameters
    ----------
    tokens : Union[List[List[int]], List[LongTensor]]
        indices: [b, l]
    direction : str
        one of "l2f" and "r2l"
    device : torch.device

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        tgt, out: [b, l], [b, l]
    """
    assert direction in {"l2r", "r2l"}

    if isinstance(tokens[0], list):
        tokens = [torch.tensor(t, dtype=torch.long) for t in tokens]

    if direction == "l2r":
        tokens = tokens
        start_w = vocab.SOS_IDX
        stop_w = vocab.EOS_IDX
    else:
        tokens = [torch.flip(t, dims=[0]) for t in tokens]
        start_w = vocab.EOS_IDX
        stop_w = vocab.SOS_IDX

    batch_size = len(tokens)      #原始长度
    lens = [len(t) for t in tokens]

    length = max(lens) + 1               #pad补全（默认不补全）
    if pad_to_len is not None:
        length = max(length, pad_to_len)

    tgt = torch.full(
        (batch_size, length),
        fill_value=vocab.PAD_IDX,             #最大维度 全部打0
        dtype=torch.long,
        device=device,
    )
    out = torch.full(
        (batch_size, length),
        fill_value=vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )

    for i, token in enumerate(tokens):
        tgt[i, 0] = start_w
        tgt[i, 1 : (1 + lens[i])] = token

        out[i, : lens[i]] = token
        out[i, lens[i]] = stop_w

    return tgt, out


def to_bi_tgt_out(
    tokens: List[List[int]], device: torch.device
) -> Tuple[LongTensor, LongTensor]:
    """Generate bidirection tgt and out

    Parameters
    ----------
    tokens : List[List[int]]
        indices: [b, l]
    device : torch.device

    Returns
    -------
    Tuple[LongTensor, LongTensor]
        tgt, out: [2b, l], [2b, l]
    """
    l2r_tgt, l2r_out = to_tgt_output(tokens, "l2r", device)
    r2l_tgt, r2l_out = to_tgt_output(tokens, "r2l", device)

    tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
    out = torch.cat((l2r_out, r2l_out), dim=0)

    return tgt, out
