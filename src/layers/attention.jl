"""
    MultiHeadSelfAttention(in_planes::Int, number_heads::Int; qkv_bias::Bool=false,
        attention_dropout_rate::T=0.0f0, projection_dropout_rate::T=0.0f0)

Multi-head self-attention layer

## Arguments

  - `planes`: number of input channels
  - `nheads`: number of heads
  - `qkv_bias`: whether to use bias in the layer to get the query, key and value
  - `attn_dropout_prob`: dropout probability after the self-attention layer
  - `proj_dropout_prob`: dropout probability after the projection layer
"""
function MultiHeadSelfAttention(in_planes::Int, number_heads::Int; qkv_bias::Bool=false,
        attention_dropout_rate::T=0.0f0, projection_dropout_rate::T=0.0f0) where {T}
    @argcheck in_planes % number_heads == 0

    qkv_layer = Lux.Dense(in_planes, in_planes * 3; use_bias=qkv_bias)
    attention_dropout = Lux.Dropout(attention_dropout_rate)
    projection = Lux.Chain(
        Lux.Dense(in_planes => in_planes), Lux.Dropout(projection_dropout_rate))

    return Lux.@compact(; number_heads, qkv_layer, attention_dropout,
        projection, dispatch=:MultiHeadSelfAttention) do x::AbstractArray{<:Real, 3}
        qkv = qkv_layer(x)
        q, k, v = _fast_chunk(qkv, Val(3), Val(4))
        y, _ = NNlib.dot_product_attention(
            q, k, v; fdrop=attention_dropout, nhead=number_heads)
        @return projection(y)
    end
end
