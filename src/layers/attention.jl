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
    attention_dropout_rate=0.0f0,
    projection_dropout_rate=0.0f0,
)
    Base.depwarn(
        "MultiHeadSelfAttention is deprecated. Use MultiHeadAttention from Lux instead.",
        :MultiHeadSelfAttention,
    )
    return MultiHeadSelfAttention(
        Lux.Chain(
            Lux.MultiHeadAttention(
                in_planes;
                nheads=number_heads,
                dense_kwargs=(; use_bias=use_qkv_bias),
                attention_dropout_probability=attention_dropout_rate,
            ),
            Lux.WrappedFunction(first),
            Lux.Dropout(projection_dropout_rate),
        ),
    )
end

"""
    PhysicsSelfAttentionIrregularMesh(
        dim; nheads=8, dim_head=64, dropout=0.0f0, slice_num=64
    )

Physics self-attention layer used in neural PDE solvers. See [luo2025transolver++](@citep)  and [wu2024transolver](@citep) for more details.
"""
@concrete struct PhysicsSelfAttentionIrregularMesh <: AbstractLuxContainerLayer{(
    :input_project_x,
    :input_project_fx,
    :input_project_slice,
    :q_layer,
    :k_layer,
    :v_layer,
    :out_layer,
    :dropout_layer,
)}
    input_project_x
    input_project_fx
    input_project_slice
    q_layer
    k_layer
    v_layer
    out_layer
    dropout_layer
    nheads::Int
    dim_head::Int
    slice_num::Int
end

function PhysicsSelfAttentionIrregularMesh(
    dim; nheads=8, dim_head=64, dropout_rate=0.0f0, slice_num=64, shapelist=nothing
)
    @assert shapelist === nothing "shapelist cannot be specified for \
                                   PhysicsSelfAttentionIrregularMesh"

    inner_dim = nheads * dim_head
    return PhysicsSelfAttentionIrregularMesh(
        Lux.Dense(dim => inner_dim),
        Lux.Dense(dim => inner_dim),
        Lux.Chain(
            Lux.Dense(dim_head => slice_num; init_weight=orthogonal),
            Lux.Scale(
                (1, 1, nheads);
                init_weight=(args...) -> ones32(args...) .* 2,
                use_bias=false,
            ),
        ),
        Lux.Dense(dim_head => dim_head; use_bias=false),
        Lux.Dense(dim_head => dim_head; use_bias=false),
        Lux.Dense(dim_head => dim_head; use_bias=false),
        Lux.Chain(Lux.Dense(inner_dim => dim), Lux.Dropout(dropout_rate)),
        Lux.Dropout(dropout_rate),
        nheads,
        dim_head,
        slice_num,
    )
end

function (model::PhysicsSelfAttentionIrregularMesh)(x::AbstractArray{T,3}, ps, st) where {T}
    # x : (dim, num_points, batch_size)
    _, N, B = size(x)

    # (1) Slice
    fx_mid, st_in_project_fx = model.input_project_fx(
        x, ps.input_project_fx, st.input_project_fx
    )
    fx_mid = permutedims(reshape(fx_mid, model.dim_head, model.nheads, N, B), (1, 3, 2, 4)) # C N H B

    x_mid, st_in_project_x = model.input_project_x(
        x, ps.input_project_x, st.input_project_x
    )
    x_mid = permutedims(reshape(x_mid, model.dim_head, model.nheads, N, B), (1, 3, 2, 4)) # C N H B

    slice_weights, st_in_project_slice = model.input_project_slice(
        x_mid, ps.input_project_slice, st.input_project_slice
    ) # G N H B
    slice_weights = NNlib.softmax(slice_weights; dims=1) # G N H B
    slice_norm = reshape(sum(slice_weights; dims=2), 1, :, model.nheads, B) # G 1 H B
    slice_token = NNlib.batched_mul(fx_mid, permutedims(slice_weights, (2, 1, 3, 4))) # C N H B

    slice_token = slice_token ./ (slice_norm .+ T(1e-5)) # C G H B

    # (2) Attention among slice tokens
    q_slice_token, st_q = model.q_layer(slice_token, ps.q_layer, st.q_layer) # C G H B
    k_slice_token, st_k = model.k_layer(slice_token, ps.k_layer, st.k_layer) # C G H B
    v_slice_token, st_v = model.v_layer(slice_token, ps.v_layer, st.v_layer) # C G H B
    dots =
        NNlib.batched_mul(q_slice_token, permutedims(k_slice_token, (2, 1, 3, 4))) ./
        T(sqrt(model.dim_head))
    attn = NNlib.softmax(dots; dims=1)
    attn, st_dropout = model.dropout_layer(attn, ps.dropout_layer, st.dropout_layer)
    out_slice_token = NNlib.batched_mul(attn, v_slice_token) # C G H B

    # (3) Deslice
    out_x = NNlib.batched_mul(out_slice_token, slice_weights) # C N H B
    out_x = permutedims(out_x, (1, 3, 2, 4)) # C H N B
    out_x = reshape(out_x, model.nheads * model.dim_head, N, B) # (C H) N B
    res, st_out = model.out_layer(out_x, ps.out_layer, st.out_layer)

    return (
        res,
        (;
            input_project_fx=st_in_project_fx,
            input_project_x=st_in_project_x,
            input_project_slice=st_in_project_slice,
            q_layer=st_q,
            k_layer=st_k,
            v_layer=st_v,
            out_layer=st_out,
            dropout_layer=st_dropout,
        ),
    )
end
