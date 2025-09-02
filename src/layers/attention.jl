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
    :input_project_x, :input_project_fx, :input_project_slice, :mha_layer, :out_layer
)}
    input_project_x
    input_project_fx
    input_project_slice
    out_layer
    mha_layer
    nheads::Int
    dim_head::Int
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
                (1, nheads, 1);
                init_weight=(args...) -> ones32(args...) .* 2,
                use_bias=false,
            ),
        ),
        Lux.Chain(Lux.Dense(inner_dim => dim), Lux.Dropout(dropout_rate)),
        Lux.MultiHeadAttention(
            dim_head;
            nheads=1,
            dense_kwargs=(; use_bias=false),
            attention_dropout_probability=dropout_rate,
        ),
        nheads,
        dim_head,
    )
end

function (model::PhysicsSelfAttentionIrregularMesh)(x::AbstractArray{T,3}, ps, st) where {T}
    # x : (dim, num_points, batch_size)
    _, N, B = size(x)

    # (1) Slice
    fx_mid, st_in_project_fx = model.input_project_fx(
        x, ps.input_project_fx, st.input_project_fx
    )
    fx_mid = reshape(fx_mid, model.dim_head, model.nheads, N, B) # C H N B

    x_mid, st_in_project_x = model.input_project_x(
        x, ps.input_project_x, st.input_project_x
    )
    x_mid = reshape(x_mid, model.dim_head, model.nheads, N, B) # C H N B

    slice_weights, st_in_project_slice = model.input_project_slice(
        x_mid,
        ps.input_project_slice,
        st.input_project_slice,
    ) # G H N B
    slice_weights = NNlib.softmax(slice_weights; dims=1) # G H N B
    slice_norm = reshape(sum(slice_weights; dims=3), 1, :, model.nheads, B) # 1 G H B
    slice_token = batched_matmul(
        fx_mid,
        slice_weights;
        lhs_contracting_dim=3,
        rhs_contracting_dim=3,
        lhs_batching_dims=(2, 4),
        rhs_batching_dims=(2, 4),
    ) # C G H B

    slice_token = slice_token ./ (slice_norm .+ T(1e-5)) # C G H B

    # (2) Attention among slice tokens
    (out_slice_token, _), st_attn = model.mha_layer(
        reshape(slice_token, model.dim_head, :, model.nheads * B),
        ps.mha_layer,
        st.mha_layer,
    )
    out_slice_token = reshape(out_slice_token, model.dim_head, :, model.nheads, B) # C G H B

    # (3) Deslice
    out_x = batched_matmul(
        out_slice_token,
        slice_weights;
        lhs_contracting_dim=2,
        rhs_contracting_dim=1,
        lhs_batching_dims=(3, 4),
        rhs_batching_dims=(2, 4),
    ) # C N H B
    out_x = permutedims(out_x, (1, 3, 2, 4)) # C H N B
    out_x = reshape(out_x, model.nheads * model.dim_head, N, B) # (C H) N B
    res, st_out = model.out_layer(out_x, ps.out_layer, st.out_layer)

    return (
        res,
        (;
            input_project_x=st_in_project_x,
            input_project_fx=st_in_project_fx,
            input_project_slice=st_in_project_slice,
            mha_layer=st_attn,
            out_layer=st_out,
        ),
    )
end
