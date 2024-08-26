"""
    MultiHeadSelfAttention(in_planes::Int, number_heads::Int; use_qkv_bias::Bool=false,
        attention_dropout_rate::T=0.0f0, projection_dropout_rate::T=0.0f0)

Multi-head self-attention layer

## Arguments

  - `planes`: number of input channels
  - `nheads`: number of heads
  - `use_qkv_bias`: whether to use bias in the layer to get the query, key and value
  - `attn_dropout_prob`: dropout probability after the self-attention layer
  - `proj_dropout_prob`: dropout probability after the projection layer
"""
@concrete struct MultiHeadSelfAttention <:
                 AbstractExplicitContainerLayer{(:qkv_layer, :dropout, :projection)}
    qkv_layer
    dropout
    projection
    nheads::Int
end

function MultiHeadSelfAttention(in_planes::Int, number_heads::Int; use_qkv_bias::Bool=false,
        attention_dropout_rate::T=0.0f0, projection_dropout_rate::T=0.0f0) where {T}
    @argcheck in_planes % number_heads == 0
    return MultiHeadSelfAttention(
        Lux.Dense(in_planes, in_planes * 3; use_bias=use_qkv_bias),
        Lux.Dropout(attention_dropout_rate),
        Lux.Chain(Lux.Dense(in_planes => in_planes), Lux.Dropout(projection_dropout_rate)),
        number_heads
    )
end

function (mhsa::MultiHeadSelfAttention)(x::AbstractArray{T, 3}, ps, st) where {T}
    qkv, st_qkv = mhsa.qkv_layer(x, ps.qkv_layer, st.qkv_layer)
    q, k, v = fast_chunk(qkv, Val(3), Val(1))

    attn_dropout = StatefulLuxLayer{true}(mhsa.dropout, ps.dropout, st.dropout)
    y, _ = NNlib.dot_product_attention(q, k, v; fdrop=attn_dropout, mhsa.nheads)

    z, st_proj = mhsa.projection(y, ps.projection, st.projection)

    return z, (; qkv_layer=st_qkv, dropout=attn_dropout.st, projection=st_proj)
end
