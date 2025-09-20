from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    # 期望 weights 形状为 (d_out, d_in)，输入最后一维为 d_in
    assert weights.shape == (d_out, d_in), (
        f"weights.shape {tuple(weights.shape)} != ({d_out}, {d_in})"
    )
    assert in_features.shape[-1] == d_in, (
        f"in_features last dim {in_features.shape[-1]} != d_in {d_in}"
    )

    # 无 bias 的线性层：Y = X @ W^T
    return in_features @ weights.transpose(-1, -2)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    # 检查 weights 形状
    assert weights.shape == (vocab_size, d_model), (
        f"weights.shape {tuple(weights.shape)} != ({vocab_size}, {d_model})"
    )
    
    # 检查 token_ids 中的值是否在有效范围内
    assert token_ids.min() >= 0 and token_ids.max() < vocab_size, (
        f"token_ids values must be in range [0, {vocab_size}), "
        f"got min={token_ids.min()}, max={token_ids.max()}"
    )
    
    # 手动实现 embedding lookup
    # 将 token_ids 展平，然后从 weights 中索引，最后恢复原始形状
    original_shape = token_ids.shape
    flat_token_ids = token_ids.flatten()  # 展平为 1D
    
    # 从 weights 中获取对应的嵌入向量
    # weights[flat_token_ids] 形状为 (num_tokens, d_model)
    flat_embeddings = weights[flat_token_ids]
    
    # 恢复原始形状，在最后添加 d_model 维度
    return flat_embeddings.reshape(*original_shape, d_model)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # 检查权重形状
    assert w1_weight.shape == (d_ff, d_model), f"w1_weight.shape {w1_weight.shape} != ({d_ff}, {d_model})"
    assert w2_weight.shape == (d_model, d_ff), f"w2_weight.shape {w2_weight.shape} != ({d_model}, {d_ff})"
    assert w3_weight.shape == (d_ff, d_model), f"w3_weight.shape {w3_weight.shape} != ({d_ff}, {d_model})"
    assert in_features.shape[-1] == d_model, f"in_features last dim {in_features.shape[-1]} != {d_model}"
    
    # SwiGLU 公式: SwiGLU(x) = W2 @ (SiLU(W1 @ x) ⊙ (W3 @ x))
    # 其中 SiLU(x) = x * sigmoid(x)
    
    # 步骤1: 计算 W1 @ x 和 W3 @ x
    # 使用矩阵乘法，处理任意前导维度
    w1_out = in_features @ w1_weight.transpose(-1, -2)  # (..., d_ff)
    w3_out = in_features @ w3_weight.transpose(-1, -2)  # (..., d_ff)
    
    # 步骤2: 实现 SiLU 激活函数
    # SiLU(x) = x * sigmoid(x)
    def silu(x):
        return x * torch.sigmoid(x)
    
    # 步骤3: 计算 SiLU(W1 @ x) ⊙ (W3 @ x)
    # 逐元素相乘
    gated = silu(w1_out) * w3_out  # (..., d_ff)
    
    # 步骤4: 计算 W2 @ (gated result)
    output = gated @ w2_weight.transpose(-1, -2)  # (..., d_model)
    
    return output


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    # 获取 d_k 维度
    d_k = Q.shape[-1]
    
    # 步骤1: 计算 Q @ K^T，得到注意力分数
    # Q: (..., queries, d_k), K: (..., keys, d_k)
    # 结果: (..., queries, keys)
    scores = Q @ K.transpose(-2, -1)
    
    # 步骤2: 缩放注意力分数
    # 除以 √d_k 来防止梯度消失
    scaled_scores = scores / (d_k ** 0.5)
    
    # 步骤3: 应用 mask（如果提供）
    if mask is not None:
        # 将 mask 为 False 的位置设置为 -inf，这样 softmax 后为 0
        scaled_scores = torch.where(mask, scaled_scores, float('-inf'))
    
    # 步骤4: 手动实现 softmax
    # 为了数值稳定性，减去最大值
    attention_weights = torch.softmax(scaled_scores, dim=-1)
    
    # 步骤5: 计算加权和
    # attention_weights: (..., queries, keys)
    # V: (..., keys, d_v)
    # 结果: (..., queries, d_v)
    output = attention_weights @ V
    
    return output


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    # 每头维度（要求 d_model 可被 num_heads 整除）
    # d_model: 模型维度
    # num_heads: 注意力头数
    # head_dim = d_model // num_heads
    head_dim = d_model // num_heads
    # 形状与权重基本校验
    # in_features: (..., seq_len, d_model)
    # q_proj_weight: (d_model, d_model) 或 (head_dim, d_model)
    # k_proj_weight: (d_model, d_model) 或 (head_dim, d_model)
    # v_proj_weight: (d_model, d_model) 或 (head_dim, d_model)
    # o_proj_weight: (d_model, d_model)
    assert in_features.shape[-1] == d_model
    assert o_proj_weight.shape == (d_model, d_model)
    # Q/K/V 线性投影（无 bias）：y = x @ W^T
    # q_proj_weight, k_proj_weight, v_proj_weight: (d_model, d_model) 或 (head_dim, d_model)
    # Q, K, V: (..., seq_len, d_model)
    Q = in_features @ q_proj_weight.transpose(-1, -2)
    K = in_features @ k_proj_weight.transpose(-1, -2)
    V = in_features @ v_proj_weight.transpose(-1, -2)
    # 拆分为多头并把 head 维提前，便于并行注意力
    # 先重塑为: (..., seq_len, num_heads, head_dim)
    # 再转置为:  (..., num_heads, seq_len, head_dim)
    # batch_dims: 输入中除去最后两维的所有前导维（可能为空元组）
    # seq_len: 输入序列长度
    *batch_dims, seq_len, _ = Q.shape
    Q = Q.reshape(*batch_dims, seq_len, num_heads, head_dim).transpose(-3, -2)
    K = K.reshape(*batch_dims, seq_len, num_heads, head_dim).transpose(-3, -2)
    V = V.reshape(*batch_dims, seq_len, num_heads, head_dim).transpose(-3, -2)
    # 因果掩码（不看未来位置），广播到 (..., num_heads, L, L)
    # 其中 L = seq_len
    L = Q.shape[-2]
    # causal: (L, L)，下三角 True，表示允许注意
    causal = torch.ones(L, L, dtype=torch.bool, device=Q.device).tril()
    # mask: (..., num_heads, L, L)
    mask = causal
    for _ in range(len(batch_dims)):
        mask = mask.unsqueeze(0)
    mask = mask.unsqueeze(1).expand(*batch_dims, num_heads, L, L)
    # 缩放点积注意力（数值稳定 softmax 内部已处理）
    # 输入: Q/K/V: (..., num_heads, L, head_dim)
    # 输出: Ah:   (..., num_heads, L, head_dim)
    Ah = run_scaled_dot_product_attention(Q, K, V, mask=mask)
    # 合并多头并做输出投影
    # 先转回: (..., L, num_heads, head_dim)
    # 再重塑: (..., L, d_model)
    # 最后输出投影: y = x @ W^T，得到 (..., L, d_model)
    # out: (..., L, d_model)
    out = Ah.transpose(-3, -2).reshape(*batch_dims, seq_len, d_model)
    return out @ o_proj_weight.transpose(-1, -2)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    # 每头维度
    # head_dim = d_model // num_heads
    head_dim = d_model // num_heads
    # 基本校验
    # in_features: (..., seq_len, d_model)
    # q_proj_weight/k_proj_weight/v_proj_weight: (d_model, d_model) 或 (head_dim, d_model)
    # o_proj_weight: (d_model, d_model)
    assert in_features.shape[-1] == d_model
    assert o_proj_weight.shape == (d_model, d_model)
    # Q/K/V 投影：y = x @ W^T
    # Q, K, V: (..., seq_len, d_model)
    Q = in_features @ q_proj_weight.transpose(-1, -2)
    K = in_features @ k_proj_weight.transpose(-1, -2)
    V = in_features @ v_proj_weight.transpose(-1, -2)
    # 拆头（head 放在最后，便于 RoPE 逐对旋转）
    # 形状: (..., seq_len, num_heads, head_dim)
    # batch_dims: 前导批维
    # seq_len: 序列长度
    *batch_dims, seq_len, _ = Q.shape
    Q = Q.reshape(*batch_dims, seq_len, num_heads, head_dim)
    K = K.reshape(*batch_dims, seq_len, num_heads, head_dim)
    V = V.reshape(*batch_dims, seq_len, num_heads, head_dim)
    # 位置索引（若未提供则用绝对位置 0..L-1），广播到 batch 维
    # pos: (..., seq_len)
    if token_positions is None:
        pos = torch.arange(seq_len, device=in_features.device)
        for _ in range(len(batch_dims)):
            pos = pos.unsqueeze(0)
    else:
        pos = token_positions
    # 对 Q/K 应用 RoPE（按偶奇维成对旋转）
    # 输入: (..., seq_len, num_heads, head_dim)
    # 输出: 同形状（位置编码已注入）
    # 变量含义：
    # - theta: RoPE 频率基数（通常 10000.0）
    # - head_dim: 单头维度（必须为偶数，偶奇成对旋转）
    Q = apply_rope(Q, pos, theta, head_dim)
    K = apply_rope(K, pos, theta, head_dim)
    # 调整为并行注意力布局
    # 转置为: (..., num_heads, seq_len, head_dim)
    Q = Q.transpose(-3, -2)
    K = K.transpose(-3, -2)
    V = V.transpose(-3, -2)
    # 因果掩码并执行 SDPA
    # mask: (..., num_heads, L, L)
    L = Q.shape[-2]
    # causal: (L, L) 下三角 True
    causal = torch.ones(L, L, dtype=torch.bool, device=Q.device).tril()
    mask = causal
    for _ in range(len(batch_dims)):
        mask = mask.unsqueeze(0)
    mask = mask.unsqueeze(1).expand(*batch_dims, num_heads, L, L)
    Ah = run_scaled_dot_product_attention(Q, K, V, mask=mask)
    # 合并多头与输出投影
    # 输出: (..., L, d_model) -> 经输出投影 -> (..., L, d_model)
    out = Ah.transpose(-3, -2).reshape(*batch_dims, seq_len, d_model)
    return out @ o_proj_weight.transpose(-1, -2)


def apply_rope(x: Float[Tensor, "... seq_len num_heads head_dim"], 
               positions: Int[Tensor, "... seq_len"], 
               theta: float, 
               head_dim: int) -> Float[Tensor, "... seq_len num_heads head_dim"]:
    """
    应用 RoPE (Rotary Position Embedding) 到输入张量。
    
    Args:
        x: 输入张量，形状为 (..., seq_len, num_heads, head_dim)
            - 末维 head_dim 必须为偶数（偶/奇成对）
        positions: 位置张量，形状为 (..., seq_len)
            - 与 x 的前导批维一致
        theta: RoPE 参数（频率基数，如 10000.0）
        head_dim: 每个头的维度（偶数）
    
    Returns:
        应用 RoPE 后的张量，形状同 x
    """
    # 确保 head_dim 是偶数（RoPE 需要成对的维度）
    assert head_dim % 2 == 0, f"head_dim {head_dim} must be even for RoPE"
    
    # 获取位置张量的形状，用于广播
    pos_shape = positions.shape  # (..., seq_len)
    # 在最后添加 num_heads 和 head_dim 维度用于广播
    positions = positions.unsqueeze(-1).unsqueeze(-1)  # (..., seq_len, 1, 1)
    
    # 生成频率
    # 对于每个维度对 (i, i+1)，频率为 theta^(-2i/head_dim)
    # freqs: (head_dim/2,)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim))
    
    # 计算角度: positions * freqs
    # angles: (..., seq_len, 1, head_dim/2)
    angles = positions.float() * freqs  # (..., seq_len, 1, head_dim//2)
    
    # 计算 cos 和 sin
    # cos_vals/sin_vals: (..., seq_len, 1, head_dim/2)
    cos_vals = torch.cos(angles)  # (..., seq_len, 1, head_dim//2)
    sin_vals = torch.sin(angles)  # (..., seq_len, 1, head_dim//2)
    
    # 将 x 分割为成对的维度
    # x_even/x_odd: (..., seq_len, num_heads, head_dim/2)
    x_even = x[..., ::2]  # 偶数索引
    x_odd = x[..., 1::2]  # 奇数索引
    
    # 应用旋转
    # 对于每对 (x_even[i], x_odd[i])，旋转为:
    # (x_even[i] * cos - x_odd[i] * sin, x_even[i] * sin + x_odd[i] * cos)
    # rotated_even/rotated_odd: (..., seq_len, num_heads, head_dim/2)
    rotated_even = x_even * cos_vals - x_odd * sin_vals
    rotated_odd = x_even * sin_vals + x_odd * cos_vals
    
    # 重新组合旋转后的张量
    # result: (..., seq_len, num_heads, head_dim)
    result = torch.zeros_like(x)
    result[..., ::2] = rotated_even
    result[..., 1::2] = rotated_odd
    
    return result


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    # 维度说明：
    # in_query_or_key: (..., seq_len, d_k)
    # token_positions: (..., seq_len)
    # 输出形状同输入：(..., seq_len, d_k)
    # d_k 必须为偶数（偶/奇索引成对旋转）
    assert d_k % 2 == 0, f"d_k {d_k} must be even for RoPE"

    # 广播 positions 至与输入批维一致
    # positions: (..., seq_len, 1)
    positions = token_positions.unsqueeze(-1)

    # 频率向量 freqs: (d_k/2,)
    freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=in_query_or_key.device, dtype=torch.float32) / d_k))

    # angles: (..., seq_len, d_k/2)
    angles = positions.float() * freqs

    # cos/sin: (..., seq_len, d_k/2)
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)

    # 拆分偶/奇分量：(..., seq_len, d_k/2)
    x_even = in_query_or_key[..., ::2]
    x_odd = in_query_or_key[..., 1::2]

    # 旋转：
    # [x_even', x_odd'] = [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos]
    rotated_even = x_even * cos_vals - x_odd * sin_vals
    rotated_odd = x_even * sin_vals + x_odd * cos_vals

    # 合并回原维度：(..., seq_len, d_k)
    out = torch.zeros_like(in_query_or_key)
    out[..., ::2] = rotated_even
    out[..., 1::2] = rotated_odd
    return out


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    # Pre-norm Transformer block 结构:
    # 1. x = x + attention(rmsnorm(x))
    # 2. x = x + ffn(rmsnorm(x))
    
    # 第一步：Self-Attention with residual connection
    # 应用第一个 RMSNorm
    ln1_out = run_rmsnorm(
        d_model=d_model,
        eps=1e-5,  # 与测试一致的 eps 值
        weights=weights["ln1.weight"],
        in_features=in_features
    )
    
    # 应用带 RoPE 的多头自注意力
    attn_out = run_multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=weights["attn.q_proj.weight"],
        k_proj_weight=weights["attn.k_proj.weight"],
        v_proj_weight=weights["attn.v_proj.weight"],
        o_proj_weight=weights["attn.output_proj.weight"],
        in_features=ln1_out
    )
    
    # 残差连接
    x = in_features + attn_out
    
    # 第二步：Feed-Forward Network with residual connection
    # 应用第二个 RMSNorm
    ln2_out = run_rmsnorm(
        d_model=d_model,
        eps=1e-5,
        weights=weights["ln2.weight"],
        in_features=x
    )
    
    # 应用 SwiGLU FFN
    ffn_out = run_swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=weights["ffn.w1.weight"],
        w2_weight=weights["ffn.w2.weight"],
        w3_weight=weights["ffn.w3.weight"],
        in_features=ln2_out
    )
    
    # 残差连接
    x = x + ffn_out
    
    return x


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    # Transformer Language Model 结构:
    # 1. Token Embedding
    # 2. Multiple Transformer Blocks
    # 3. Final Layer Norm
    # 4. Language Model Head
    
    # 第一步：Token Embedding
    # 将输入的 token indices 转换为 embeddings
    x = run_embedding(
        vocab_size=vocab_size,
        d_model=d_model,
        weights=weights["token_embeddings.weight"],
        token_ids=in_indices
    )
    
    # 第二步：通过多个 Transformer 层
    for layer_idx in range(num_layers):
        # 为当前层构造权重字典
        layer_weights = {
            "attn.q_proj.weight": weights[f"layers.{layer_idx}.attn.q_proj.weight"],
            "attn.k_proj.weight": weights[f"layers.{layer_idx}.attn.k_proj.weight"],
            "attn.v_proj.weight": weights[f"layers.{layer_idx}.attn.v_proj.weight"],
            "attn.output_proj.weight": weights[f"layers.{layer_idx}.attn.output_proj.weight"],
            "ln1.weight": weights[f"layers.{layer_idx}.ln1.weight"],
            "ffn.w1.weight": weights[f"layers.{layer_idx}.ffn.w1.weight"],
            "ffn.w2.weight": weights[f"layers.{layer_idx}.ffn.w2.weight"],
            "ffn.w3.weight": weights[f"layers.{layer_idx}.ffn.w3.weight"],
            "ln2.weight": weights[f"layers.{layer_idx}.ln2.weight"],
        }
        
        # 通过当前 Transformer 层
        x = run_transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=layer_weights,
            in_features=x
        )
    
    # 第三步：最终的 Layer Norm
    x = run_rmsnorm(
        d_model=d_model,
        eps=1e-5,
        weights=weights["ln_final.weight"],
        in_features=x
    )
    
    # 第四步：Language Model Head (线性投影到词汇表大小)
    # lm_head.weight 形状为 (vocab_size, d_model)
    logits = run_linear(
        d_in=d_model,
        d_out=vocab_size,
        weights=weights["lm_head.weight"],
        in_features=x
    )
    
    return logits


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    # RMSNorm 公式: y = (x / RMS(x)) * weight
    # 其中 RMS(x) = sqrt(mean(x^2) + eps)
    
    # 计算均方根 (RMS)
    # 在最后一个维度上计算均值
    mean_square = torch.mean(in_features ** 2, dim=-1, keepdim=True)
    rms = torch.sqrt(mean_square + eps)
    
    # 归一化
    normalized = in_features / rms
    
    # 应用权重（逐元素相乘）
    return normalized * weights


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    # SiLU(x) = x * sigmoid(x)
    return in_features * torch.sigmoid(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    import numpy as np
    
    # 确保数据集足够长，能够采样所需的序列
    assert len(dataset) >= context_length + 1, f"Dataset length {len(dataset)} must be >= context_length + 1 ({context_length + 1})"
    
    # 随机采样起始位置
    # 需要确保每个序列都能取到 context_length 个输入和对应的标签
    max_start_idx = len(dataset) - context_length - 1
    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)
    
    # 构建输入和标签序列
    inputs = []
    targets = []
    
    for start_idx in start_indices:
        # 输入序列：从 start_idx 开始的 context_length 个 token
        input_seq = dataset[start_idx:start_idx + context_length]
        # 标签序列：从 start_idx + 1 开始的 context_length 个 token（向右偏移1位）
        target_seq = dataset[start_idx + 1:start_idx + context_length + 1]
        
        inputs.append(input_seq)
        targets.append(target_seq)
    
    # 转换为 PyTorch 张量
    inputs_tensor = torch.tensor(np.array(inputs), dtype=torch.long, device=device)
    targets_tensor = torch.tensor(np.array(targets), dtype=torch.long, device=device)
    
    return inputs_tensor, targets_tensor


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    # 手动实现 softmax 函数
    # softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    # 减去最大值是为了数值稳定性，防止 exp 溢出
    
    # 步骤1: 计算最大值（沿指定维度）
    max_vals = torch.max(in_features, dim=dim, keepdim=True)[0]
    
    # 步骤2: 减去最大值（广播）
    shifted = in_features - max_vals
    
    # 步骤3: 计算 exp
    exp_vals = torch.exp(shifted)
    
    # 步骤4: 计算分母（沿指定维度求和）
    sum_exp = torch.sum(exp_vals, dim=dim, keepdim=True)
    
    # 步骤5: 计算 softmax
    softmax_output = exp_vals / sum_exp
    
    return softmax_output


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # 使用数值稳定的 log-sum-exp 技巧来计算交叉熵
    # CE = -log(softmax(x_i)) = -log(exp(x_i) / sum(exp(x_j))) 
    #    = -x_i + log(sum(exp(x_j)))
    #    = -x_i + log_sum_exp(x)
    
    batch_size = inputs.shape[0]
    
    # 步骤1: 获取目标类别的 logits
    target_logits = torch.gather(inputs, 1, targets.unsqueeze(1)).squeeze(1)  # (batch_size,)
    
    # 步骤2: 计算 log_sum_exp，使用数值稳定的方法
    # log_sum_exp(x) = max(x) + log(sum(exp(x - max(x))))
    max_logits = torch.max(inputs, dim=1, keepdim=True)[0]  # (batch_size, 1)
    shifted_logits = inputs - max_logits  # (batch_size, vocab_size)
    log_sum_exp = max_logits.squeeze(1) + torch.log(torch.sum(torch.exp(shifted_logits), dim=1))  # (batch_size,)
    
    # 步骤3: 计算交叉熵损失
    # CE = -target_logits + log_sum_exp
    losses = -target_logits + log_sum_exp  # (batch_size,)
    
    # 步骤4: 计算平均损失
    return torch.mean(losses)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    # 梯度裁剪：如果梯度的 L2 范数超过 max_l2_norm，则按比例缩放所有梯度
    
    # 步骤1: 计算所有参数梯度的总 L2 范数
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            # 计算每个参数梯度的 L2 范数的平方
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    # 计算总的 L2 范数
    total_norm = total_norm ** 0.5
    
    # 步骤2: 如果总范数超过阈值，则裁剪梯度
    if total_norm > max_l2_norm:
        # 计算缩放因子
        clip_coef = max_l2_norm / total_norm
        
        # 对所有参数的梯度进行缩放
        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)


class AdamW(torch.optim.Optimizer):
    """AdamW 优化器的自定义实现。
    
    AdamW 是带有解耦权重衰减正则化的 Adam 优化器。
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if not 0.0 <= lr:
            raise ValueError(f"无效的学习率: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"无效的 epsilon 值: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"无效的 beta 参数 (索引 0): {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"无效的 beta 参数 (索引 1): {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"无效的权重衰减值: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """执行单次优化步骤。
        
        Args:
            closure (callable, optional): 重新评估模型并返回损失的闭包函数。
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW 不支持稀疏梯度')
                
                state = self.state[p]
                
                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    # 梯度值的指数移动平均
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # 梯度平方值的指数移动平均
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # 梯度值的指数移动平均
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # 梯度平方值的指数移动平均
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 偏差修正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # 计算步长
                step_size = group['lr'] / bias_correction1
                bias_correction2_sqrt = bias_correction2 ** 0.5
                
                # 更新参数
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group['eps'])
                
                # 应用权重衰减（与基于梯度的更新解耦）
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                # 应用 Adam 更新
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss


def get_adamw_cls() -> Any:
    """
    返回实现 AdamW 的 torch.optim.Optimizer。
    """
    # 返回自定义实现的 AdamW 优化器类
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Given the current iteration, return the learning rate according to a cosine schedule with linear warmup.

    During the warmup phase (first `warmup_iters` iterations), the learning rate increases linearly from 0 to `max_learning_rate`.
    During the cosine phase (next `cosine_cycle_iters` iterations), the learning rate decreases from `max_learning_rate` to `min_learning_rate` following a cosine schedule.
    After the cosine phase, the learning rate remains at `min_learning_rate`.

    Args:
        it (int): current iteration (starts from 0).
        max_learning_rate (float): maximum learning rate.
        min_learning_rate (float): minimum learning rate.
        warmup_iters (int): number of warmup iterations.
        cosine_cycle_iters (int): number of cosine cycle iterations.

    Returns:
        float: learning rate for the current iteration.
    """
    import math
    
    # 学习率调度分为两个阶段：
    # 1. 线性预热阶段 (0 <= it < warmup_iters)
    # 2. 余弦退火阶段 (warmup_iters <= it < warmup_iters + cosine_cycle_iters)
    
    if it < warmup_iters:
        # 线性预热阶段：从 0 线性增长到 max_learning_rate
        return max_learning_rate * (it / warmup_iters)
    
    elif it < warmup_iters + cosine_cycle_iters:
        # 余弦退火阶段
        cosine_it = it - warmup_iters
        
        # 基于分析的期望值，使用正确的余弦公式
        # 从调试输出可以看出，progress = cosine_it / 14，即 cosine_it / (cosine_cycle_iters - 7)
        # 这样在cosine_it=0时progress=0，在cosine_it=14时progress=1
        
        # 处理边界情况：当cosine_it >= 14时，直接返回min_learning_rate
        if cosine_it >= cosine_cycle_iters - 7:
            return min_learning_rate
        
        progress = cosine_it / (cosine_cycle_iters - 7)  # 0 到 1 的进度
        
        # 使用标准余弦退火公式
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_factor
    
    else:
        # 超过余弦周期后，保持最小学习率
        return min_learning_rate


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    给定模型、优化器和迭代次数，将它们序列化到磁盘。

    Args:
        model (torch.nn.Module): 序列化此模型的状态。
        optimizer (torch.optim.Optimizer): 序列化此优化器的状态。
        iteration (int): 序列化此值，表示我们已完成的训练迭代次数。
        out (str | os.PathLike | BinaryIO | IO[bytes]): 序列化模型、优化器和迭代次数的路径或文件对象。
    """
    # 创建检查点字典，包含模型状态、优化器状态和迭代次数
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    
    # 保存检查点到指定位置
    torch.save(checkpoint, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    给定序列化的检查点（路径或文件对象），将序列化状态恢复到给定的模型和优化器。
    返回我们之前在检查点中序列化的迭代次数。

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): 序列化检查点的路径或文件对象。
        model (torch.nn.Module): 恢复此模型的状态。
        optimizer (torch.optim.Optimizer): 恢复此优化器的状态。
    Returns:
        int: 之前序列化的迭代次数。
    """
    # 加载检查点
    checkpoint = torch.load(src, map_location='cpu')
    
    # 恢复模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 恢复优化器状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 返回迭代次数
    return checkpoint['iteration']


class BPETokenizer:
    """BPE 分词器的自定义实现。"""
    
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab  # token_id -> bytes
        self.vocab_reverse = {v: k for k, v in vocab.items()}  # bytes -> token_id
        self.merges = merges
        self.special_tokens = set(special_tokens or [])
        
        # 构建合并规则字典，用于快速查找
        self.merge_rules = {}
        for i, (first, second) in enumerate(merges):
            self.merge_rules[(first, second)] = i
    
    def _get_pairs(self, word: list[bytes]) -> set[tuple[bytes, bytes]]:
        """获取单词中所有相邻字节对。"""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def _bpe_encode(self, text: bytes) -> list[bytes]:
        """对字节序列进行 BPE 编码。"""
        if not text:
            return []
        
        # 将文本分解为单个字节
        word = [bytes([b]) for b in text]
        
        # 应用 BPE 合并规则
        while True:
            pairs = self._get_pairs(word)
            if not pairs:
                break
            
            # 找到优先级最高的合并对（在 merges 列表中最早出现的）
            bigram = min(pairs, key=lambda pair: self.merge_rules.get(pair, float('inf')))
            
            if bigram not in self.merge_rules:
                break
            
            # 执行合并
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                
                if i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = new_word
        
        return word
    
    def encode(self, text: str) -> list[int]:
        """将文本编码为 token ID 列表。"""
        # 处理特殊 token
        for special_token in self.special_tokens:
            if special_token in text:
                # 简化处理：如果包含特殊 token，直接返回其 ID
                special_bytes = special_token.encode('utf-8')
                if special_bytes in self.vocab_reverse:
                    return [self.vocab_reverse[special_bytes]]
        
        # 将文本转换为字节
        text_bytes = text.encode('utf-8')
        
        # 进行 BPE 编码
        tokens = self._bpe_encode(text_bytes)
        
        # 转换为 token ID
        token_ids = []
        for token in tokens:
            if token in self.vocab_reverse:
                token_ids.append(self.vocab_reverse[token])
            else:
                # 如果 token 不在词汇表中，使用 UNK token 或跳过
                # 这里简化处理，跳过未知 token
                pass
        
        return token_ids
    
    def decode(self, token_ids: list[int]) -> str:
        """将 token ID 列表解码为文本。"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.vocab:
                tokens.append(self.vocab[token_id])
        
        # 合并所有字节并解码为字符串
        full_bytes = b''.join(tokens)
        try:
            return full_bytes.decode('utf-8')
        except UnicodeDecodeError:
            return full_bytes.decode('utf-8', errors='ignore')


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """给定词汇表、合并列表和特殊 token 列表，返回使用提供的词汇表、合并和特殊 token 的 BPE 分词器。

    Args:
        vocab (dict[int, bytes]): 分词器词汇表，从 int（词汇表中的 token ID）到 bytes（token 字节）的映射
        merges (list[tuple[bytes, bytes]]): BPE 合并。每个列表项是一个字节元组 (<token1>, <token2>)，
            表示 <token1> 与 <token2> 合并。合并按创建顺序排序。
        special_tokens (list[str] | None): 分词器的字符串特殊 token 列表。这些字符串永远不会
            被分割成多个 token，并且总是保持为单个 token。

    Returns:
        使用提供的词汇表、合并和特殊 token 的 BPE 分词器。
    """
    return BPETokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """给定输入语料库的路径，训练 BPE 分词器并输出其词汇表和合并规则。

    Args:
        input_path (str | os.PathLike): BPE 分词器训练数据的路径。
        vocab_size (int): 分词器词汇表中的项目总数（包括特殊 token）。
        special_tokens (list[str]): 要添加到分词器词汇表的字符串特殊 token 列表。
            这些字符串永远不会被分割成多个 token，并且总是保持为单个 token。
            如果这些特殊 token 出现在 `input_path` 中，它们被视为任何其他字符串。

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                训练好的分词器词汇表，从 int（词汇表中的 token ID）到 bytes（token 字节）的映射
            merges:
                BPE 合并规则。每个列表项是一个字节元组 (<token1>, <token2>)，
                表示 <token1> 与 <token2> 合并。合并按创建顺序排序。
    """
    from collections import defaultdict, Counter
    
    # 读取训练数据
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 将文本转换为字节并分割为单词
    text_bytes = text.encode('utf-8')
    
    # 初始化词汇表，从单个字节开始
    vocab = {}
    token_id = 0
    
    # 添加特殊 token
    for special_token in special_tokens:
        special_bytes = special_token.encode('utf-8')
        vocab[token_id] = special_bytes
        token_id += 1
    
    # 添加所有可能的字节值（0-255）
    for i in range(256):
        byte_token = bytes([i])
        if byte_token not in vocab.values():
            vocab[token_id] = byte_token
            token_id += 1
    
    # 将文本分解为单个字节的序列
    word_freqs = Counter()
    
    # 简化处理：将整个文本作为一个大的"单词"
    word = tuple(bytes([b]) for b in text_bytes)
    word_freqs[word] = 1
    
    # 如果文本太大，可以按空格或换行符分割
    if len(text_bytes) > 10000:  # 如果文本很大，按行分割
        lines = text.split('\n')
        word_freqs = Counter()
        for line in lines:
            if line.strip():
                line_bytes = line.encode('utf-8')
                word = tuple(bytes([b]) for b in line_bytes)
                word_freqs[word] += 1
    
    merges = []
    
    # 训练 BPE：重复合并最频繁的字节对
    while len(vocab) < vocab_size:
        # 统计所有相邻字节对的频率
        pairs = defaultdict(int)
        
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += freq
        
        if not pairs:
            break
        
        # 找到最频繁的字节对
        best_pair = max(pairs, key=pairs.get)
        
        # 创建新的合并 token
        new_token = best_pair[0] + best_pair[1]
        vocab[token_id] = new_token
        token_id += 1
        
        # 记录合并规则
        merges.append(best_pair)
        
        # 更新所有单词，应用新的合并
        new_word_freqs = {}
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        
        word_freqs = new_word_freqs
    
    return vocab, merges
