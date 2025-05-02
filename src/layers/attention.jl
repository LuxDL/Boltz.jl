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

!!! danger "Dreprecated"

    Use `MultiHeadAttention` from `Lux` instead.
"""
@concrete struct MultiHeadSelfAttention <: AbstractLuxWrapperLayer{:model}
    model
end

function MultiHeadSelfAttention(
    in_planes::Int,
    number_heads::Int;
    use_qkv_bias::Bool=false,
    attention_dropout_rate::T=0.0f0,
    projection_dropout_rate::T=0.0f0,
) where {T}
    Base.depwarn(
        "MultiHeadSelfAttention is deprecated. Use MultiHeadAttention from Lux instead.",
        :MultiHeadSelfAttention,
    )
    return MultiHeadSelfAttention(
        Lux.Chain(
            Lux.MultiHeadAttention(
                in_planes; nheads=number_heads, dense_kwargs=(; use_bias=use_qkv_bias),
                attention_dropout_rate=attention_dropout_rate,
            ),
            Lux.WrappedFunction(first),
            Lux.Dropout(projection_dropout_rate),
        ),
    )
end
