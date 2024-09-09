@doc doc"""
    TensorProductLayer(basis_fns, out_dim::Int; init_weight = randn32)

Constructs the Tensor Product Layer, which takes as input an array of n tensor product
basis, $[B_1, B_2, \dots, B_n]$ a data point x, computes

$$z_i = W_{i, :} \odot [B_1(x_1) \otimes B_2(x_2) \otimes \dots \otimes B_n(x_n)]$$

where $W$ is the layer's weight, and returns $[z_1, \dots, z_{out}]$.

## Arguments

  - `basis_fns`: Array of TensorProductBasis $[B_1(n_1), \dots, B_k(n_k)]$, where $k$
    corresponds to the dimension of the input.
  - `out_dim`: Dimension of the output.

## Keyword Arguments

  - `init_weight`: Initializer for the weight matrix. Defaults to `randn32`.

!!! warning "Limited Backend Support"

    Support for backends apart from CPU and CUDA is limited and slow due to limited
    support for `kron` in the backend.
"""
@concrete struct TensorProductLayer <: AbstractLuxContainerLayer{(:dense,)}
    basis_fns
    dense
    out_dim::Int
end

function TensorProductLayer(basis_fns, out_dim::Int; init_weight::F=randn32) where {F}
    dense = Lux.Dense(
        prod(Base.Fix2(getproperty, :n), basis_fns) => out_dim; use_bias=false, init_weight)
    return TensorProductLayer(Tuple(basis_fns), dense, out_dim)
end

function (tp::TensorProductLayer)(x::AbstractVector, ps, st)
    y, stₙ = tp(reshape(x, :, 1), ps, st)
    return vec(y), stₙ
end

function (tp::TensorProductLayer)(x::AbstractArray{T, N}, ps, st) where {T, N}
    x′ = LuxOps.eachslice(x, Val(N - 1))                           # [I1, I2, ..., B] × T
    @argcheck length(x′) == length(tp.basis_fns)

    y = mapfoldl(safe_kron, zip(tp.basis_fns, x′)) do (fn, xᵢ)
        eachcol(reshape(fn(xᵢ), :, prod(size(xᵢ))))
    end                                            # [[D₁, ..., Dₙ] × (I1, I2, ..., B)]

    z, stₙ = tp.dense(mapreduce_stack(y), ps, st)
    return reshape(z, size(x)[1:(end - 2)]..., tp.out_dim, size(x)[end]), stₙ
end
