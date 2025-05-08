@concrete struct TransolverBlock <: AbstractLuxWrapperLayer{:model}
    model
end

function TransolverBlock(;
    nheads::Int,
    hidden_dim::Int,
    out_dim::Int,
    dropout_rate::AbstractFloat=0.0f0,
    activation=NNlib.gelu,
    mlp_ratio::Int=4,
    last_layer::Bool=false,
    slice_num::Int=32,
    geometry_kind::Symbol=:unstructured,
    shapelist=nothing,
)
    @assert geometry_kind === :unstructured "Transolver currently only supports \
                                             unstructured mesh"

    return TransolverBlock(
        Lux.Chain(
            Lux.SkipConnection(
                Lux.Chain(
                    Lux.LayerNorm(hidden_dim; dims=nothing),
                    PhysicsSelfAttentionIrregularMesh(
                        hidden_dim;
                        nheads,
                        dim_head=hidden_dim ÷ nheads,
                        dropout_rate,
                        slice_num,
                        shapelist,
                    ),
                ),
                +,
            ),
            Lux.SkipConnection(
                Lux.Chain(
                    Lux.LayerNorm(hidden_dim; dims=nothing),
                    Lux.Dense(hidden_dim => hidden_dim * mlp_ratio, activation),
                    Lux.Dense(hidden_dim * mlp_ratio => hidden_dim),
                ),
                +,
            ),
            if last_layer
                Lux.Chain(
                    Lux.LayerNorm(hidden_dim; dims=nothing),
                    Lux.Dense(hidden_dim => out_dim),
                )
            else
                Lux.NoOpLayer()
            end,
        ),
    )
end

"""
    Transolver(;
        out_dim::Int,
        nheads::Int=8,
        func_dim::Union{Int,Nothing}=nothing,
        spatial_dim::Union{Int,Nothing}=nothing,
        preprocess::Union{Nothing,AbstractLuxLayer}=nothing,
        geometry_kind::Symbol=:unstructured,
        temporal_input::Bool=false,
        hidden_dim::Int=128,
        activation=NNlib.gelu,
        num_layers::Int=4,
        nheads::Int,
        hidden_dim::Int,
        dropout_rate::AbstractFloat=0.0f0,
        activation=NNlib.gelu,
        mlp_ratio::Int=4,
        last_layer::Bool=false,
        slice_num::Int=32,
        shapelist=nothing,
    )

Implements the Transolver model from [wu2024transolver](@citep). Currently we only support
unstructured meshes without temporal input. Internally uses
[`PhysicsSelfAttentionIrregularMesh`](@ref).
"""
@concrete struct Transolver <: AbstractLuxContainerLayer{(:preprocess, :main_block)}
    preprocess
    main_block
    hidden_dim::Int
end

function LuxCore.initialparameters(rng::AbstractRNG, m::Transolver)
    return (;
        preprocess=LuxCore.initialparameters(rng, m.preprocess),
        main_block=LuxCore.initialparameters(rng, m.main_block),
        placeholder=randn32(rng, m.hidden_dim) ./ m.hidden_dim,
    )
end

function Transolver(;
    geometry_kind::Symbol=:unstructured,
    func_dim::Union{Int,Nothing}=nothing,
    spatial_dim::Union{Int,Nothing}=nothing,
    preprocess::Union{Nothing,AbstractLuxLayer}=nothing,
    temporal_input::Bool=false,
    hidden_dim::Int=128,
    activation=NNlib.gelu,
    num_layers::Int=4,
    kwargs...,
)
    @assert geometry_kind === :unstructured "Transolver currently only supports \
                                             unstructured mesh."
    @assert !temporal_input "Transolver currently does not support time input."

    if preprocess === nothing
        @assert func_dim !== nothing "func_dim must be provided if preprocess is not \
                                      nothing"
        @assert spatial_dim !== nothing "spatial_dim must be provided if preprocess is not \
                                         nothing"
        preprocess = Lux.Chain(
            Lux.Parallel(vcat, Lux.NoOpLayer(), Lux.NoOpLayer()),
            Lux.Dense(func_dim + spatial_dim => hidden_dim * 2, activation),
            Lux.Dense(hidden_dim * 2 => hidden_dim, activation),
        )
    end

    main_block = Lux.Chain(
        [
            TransolverBlock(;
                hidden_dim,
                activation,
                last_layer=(i == num_layers),
                geometry_kind,
                kwargs...,
            ) for i in 1:num_layers
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

    return Transolver(preprocess, main_block, hidden_dim)
end

function (m::Transolver)(input, ps, st)
    fx, st_preprocess = m.preprocess(input, ps.preprocess, st.preprocess)
    fx = fx .+ ps.placeholder
    out, st_main_block = m.main_block(fx, ps.main_block, st.main_block)
    return out, (; preprocess=st_preprocess, main_block=st_main_block)
end
