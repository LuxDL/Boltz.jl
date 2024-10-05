"""
    VisionTransformerEncoder(in_planes, depth, number_heads; mlp_ratio = 4.0f0,
        dropout = 0.0f0)

Transformer as used in the base ViT architecture [dosovitskiy2020image](@citep).

## Arguments

  - `in_planes`: number of input channels
  - `depth`: number of attention blocks
  - `number_heads`: number of attention heads

## Keyword Arguments

  - `mlp_ratio`: ratio of MLP layers to the number of input channels
  - `dropout_rate`: dropout rate
"""
@concrete struct VisionTransformerEncoder <: AbstractLuxWrapperLayer{:chain}
    chain <: Lux.Chain
end

function VisionTransformerEncoder(
        in_planes, depth, number_heads; mlp_ratio=4.0f0, dropout_rate=0.0f0)
    hidden_planes = floor(Int, mlp_ratio * in_planes)
    layers = [Lux.Chain(
                  Lux.SkipConnection(
                      Lux.Chain(Lux.LayerNorm((in_planes, 1); dims=1, affine=true),
                          MultiHeadSelfAttention(
                              in_planes, number_heads; attention_dropout_rate=dropout_rate,
                              projection_dropout_rate=dropout_rate)),
                      +),
                  Lux.SkipConnection(
                      Lux.Chain(Lux.LayerNorm((in_planes, 1); dims=1, affine=true),
                          Lux.Chain(Lux.Dense(in_planes => hidden_planes, NNlib.gelu),
                              Lux.Dropout(dropout_rate),
                              Lux.Dense(hidden_planes => in_planes),
                              Lux.Dropout(dropout_rate))),
                      +)) for _ in 1:depth]
    return VisionTransformerEncoder(Lux.Chain(layers...))
end
