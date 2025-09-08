function FLAREBlock(;
    channel_dim::Int,
    use_rms_norm::Bool=false,
    num_layers_mlp::Int=3,
    mlp_hidden_dim::Int=128,
    kwargs...,
)
    ln1 = use_rms_norm ? Lux.RMSNorm(channel_dim) : Lux.LayerNorm((channel_dim, 1))
    ln2 = use_rms_norm ? Lux.RMSNorm(channel_dim) : Lux.LayerNorm((channel_dim, 1))
    attn = Layers.FLARE(; channel_dim, kwargs...)
    mlp = Layers.MLP(
        channel_dim, [[mlp_hidden_dim for _ in 1:(num_layers_mlp - 1)]..., channel_dim]
    )
    return Lux.Chain(
        Lux.SkipConnection(Lux.Chain(ln1, attn), +),
        Lux.SkipConnection(Lux.Chain(ln2, mlp), +),
    )
end

@concrete struct FLARE <: AbstractLuxWrapperLayer{:layer}
    layer
end

function FLARE(;
    in_dim::Int=-1,
    preprocess::Union{Nothing,AbstractLuxLayer}=nothing,
    channel_dim::Int=64,
    num_blocks::Int=8,
    mlp_ratio::Number=1.0,
    kv_proj_ratio::Number=1.0,
    in_out_proj_ratio::Number=1.0,
    kwargs...,
)
    mlp_hidden_dim = ceil(Int, channel_dim * mlp_ratio)
    kv_proj_hidden_dim = ceil(Int, channel_dim * kv_proj_ratio)
    in_out_proj_hidden_dim = ceil(Int, channel_dim * in_out_proj_ratio)

    if preprocess === nothing
        @argcheck in_dim > 0 "`in_dim` must be provided if preprocess is not nothing"
        preprocess = Layers.MLP(
            in_dim,
            [[in_out_proj_hidden_dim for _ in 1:(num_blocks - 1)]..., channel_dim],
            NNlib.swish;
            residual_connection=true,
        )
    else
        @argcheck in_dim == -1 "`in_dim` must not be provided if preprocess is not nothing"
    end

    main_block = Lux.Chain(
        [
            FLAREBlock(;
                channel_dim,
                kwargs...,
                mlp_hidden_dim=mlp_hidden_dim,
                kv_proj_hidden_dim=kv_proj_hidden_dim,
            ) for _ in 1:num_blocks
        ]...,
    )

    # Principled Initialization of the Model
    preprocess = fmap(preprocess) do m
        m isa Lux.Dense || return m
        @set! m.init_bias = zeros32
        @set! m.init_weight = truncated_normal(; std=0.02f0)
        return m
    end
    main_block = fmap(main_block) do m
        m isa Lux.Dense || return m
        @set! m.init_bias = zeros32
        @set! m.init_weight = truncated_normal(; std=0.02f0)
        return m
    end

    return FLARE(Lux.Chain(; preprocess, main_block))
end
