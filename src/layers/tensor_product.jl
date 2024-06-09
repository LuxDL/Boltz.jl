@doc doc"""
    TensorProductLayer(model, out_dim::Int; init_weight = randn32)

Constructs the Tensor Product Layer, which takes as input an array of n tensor product
basis, $[B_1, B_2, \dots, B_n]$ a data point x, computes

$$z_i = W_{i, :} \odot [B_1(x_1) \otimes B_2(x_2) \otimes \dots \otimes B_n(x_n)]$$

where $W$ is the layer's weight, and returns $[z_1, \dots, z_{out}]$.

## Arguments

  - `basis_fns`: Array of TensorProductBasis $[B_1(n_1), \dots, B_k(n_k)]$, where $k$
    corresponds to the dimension of the input.
  - `out_dim`: Dimension of the output.
  - `init_weight`: Initializer for the weight matrix. Defaults to `randn32`.

!!! warning

    This layer currently only works on CPU and CUDA devices.
"""
function TensorProductLayer(basis_fns, out_dim::Int; init_weight::F=randn32) where {F}
    dense = Lux.Dense(
        prod(Base.Fix2(getproperty, :n), basis_fns) => out_dim; use_bias=false, init_weight)
    return Lux.@compact(; basis_fns=Tuple(basis_fns), dense,
        out_dim, dispatch=:TensorProductLayer) do x::AbstractArray #  I1 x I2 x ... x T  x B
        dev = get_device(x)
        @argcheck dev isa LuxCPUDevice || dev isa LuxCUDADevice # kron is not widely supported

        x_ = Lux._eachslice(x, Val(ndims(x) - 1))                  # [I1 x I2 x ... x B] x T
        @argcheck length(x_) == length(basis_fns)

        y = mapfoldl(_kron, zip(basis_fns, x_)) do (fn, xᵢ)
            eachcol(reshape(fn(xᵢ), :, prod(size(xᵢ))))
        end                                        # [[D₁ x ... x Dₙ] x (I1 x I2 x ... x B)]

        @return reshape(dense(_stack(y)), size(x)[1:(end - 2)]..., out_dim, size(x)[end])
    end
end

# CUDA `kron` exists only for `CuMatrix` so we define `kron` directly by converting to
# a matrix
@inline _kron(a, b) = map(__kron, a, b)
@inline function __kron(a::AbstractVector, b::AbstractVector)
    return vec(kron(reshape(a, :, 1), reshape(b, 1, :)))
end
