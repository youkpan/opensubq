"""分布式通信工具：Ulysses 序列并行 + P2P KV 拉取。"""

from __future__ import annotations

import torch
import torch.distributed as dist


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def all_gather_block_summary(
    block_summary: torch.Tensor,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """All-Gather Block Summary 到所有 Rank。

    Block Summary 数据量极小 (1M/128=8000 blocks × 64 dim ≈ 2MB)，
    All-Gather 开销可忽略。

    Args:
        block_summary: [batch, heads, local_num_blocks, route_dim]
    Returns:
        [batch, heads, global_num_blocks, route_dim]
    """
    if not dist.is_initialized():
        return block_summary

    ws = dist.get_world_size(group)
    if ws == 1:
        return block_summary

    gathered = [torch.zeros_like(block_summary) for _ in range(ws)]
    dist.all_gather(gathered, block_summary, group=group)
    return torch.cat(gathered, dim=2)


def async_p2p_kv_fetch(
    target_block_ids: torch.Tensor,
    local_block_offset: int,
    local_num_blocks: int,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    group: dist.ProcessGroup | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """异步跨卡拉取远程 Block 的 KV Cache。

    Args:
        target_block_ids: 需要的 Block IDs (全局编号)
        local_block_offset: 本 Rank 的起始 Block 偏移
        local_num_blocks: 本 Rank 的 Block 数量
        key_cache: [batch, heads, local_num_blocks * block_size, head_dim]
        value_cache: 同 key_cache

    Returns:
        fetched_k, fetched_v: 拼接后的 KV (本地+远程)
    """
    if not dist.is_initialized():
        return key_cache, value_cache

    rank = dist.get_rank(group)
    ws = dist.get_world_size(group)

    # 识别哪些 Block 在本地，哪些需要远程拉取
    local_min = local_block_offset
    local_max = local_block_offset + local_num_blocks

    # 构造每 Rank 需要的 Block ID 列表
    # (简化实现: 用 blocking 通信; 生产环境用 async)
    all_requests = [torch.empty(0, dtype=torch.long, device=key_cache.device) for _ in range(ws)]
    for bid in target_block_ids.cpu().tolist():
        owner_rank = bid // local_num_blocks
        owner_rank = min(owner_rank, ws - 1)
        all_requests[owner_rank] = torch.cat([all_requests[owner_rank], torch.tensor([bid])])

    # 发送请求
    remote_k, remote_v = [], []
    for r in range(ws):
        if r == rank:
            continue
        req_count = torch.tensor([len(all_requests[r])], device=key_cache.device)
        dist.send(req_count, dst=r, group=group)

        if len(all_requests[r]) > 0:
            local_bids = all_requests[r] - local_block_offset
            local_bids = local_bids.clamp(0, local_num_blocks - 1)
            bs = key_cache.shape[2] // local_num_blocks  # block_size
            k_send = key_cache[:, :, local_bids * bs : (local_bids + 1) * bs]
            v_send = value_cache[:, :, local_bids * bs : (local_bids + 1) * bs]
            dist.send(k_send, dst=r, group=group)
            dist.send(v_send, dst=r, group=group)

    # 接收
    for r in range(ws):
        if r == rank:
            continue
        cnt = torch.tensor([0], device=key_cache.device)
        dist.recv(cnt, src=r, group=group)
        if cnt.item() > 0:
            bs = key_cache.shape[2] // local_num_blocks
            rk = torch.zeros(1, key_cache.shape[1], cnt.item() * bs, key_cache.shape[3], device=key_cache.device)
            rv = torch.zeros_like(rk)
            dist.recv(rk, src=r, group=group)
            dist.recv(rv, src=r, group=group)
            remote_k.append(rk)
            remote_v.append(rv)

    if remote_k:
        return torch.cat([key_cache] + remote_k, dim=2), torch.cat([value_cache] + remote_v, dim=2)
    return key_cache, value_cache
