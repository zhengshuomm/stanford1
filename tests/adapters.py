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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    raise NotImplementedError


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    raise NotImplementedError
