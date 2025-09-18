module Utils

using ForwardDiff: ForwardDiff
using Statistics: mean

using MLDataDevices:
    MLDataDevices,
    AbstractDevice,
    get_device_type,
    get_device,
    CPUDevice,
    CUDADevice,
    ReactantDevice

is_extension_loaded(::Val) = false

"""
    flatten_spatial(x::AbstractArray{T, 4})

Flattens the first 2 dimensions of `x`, and permutes the remaining dimensions to (2, 1, 3).
"""
function flatten_spatial(x::AbstractArray{T,4}) where {T}
    # TODO: Should we do lazy permutedims for non-GPU arrays?
    return permutedims(reshape(x, (:, size(x, 3), size(x, 4))), (2, 1, 3))
end

"""
    second_dim_mean(x)

Computes the mean of `x` along dimension `2`.
"""
second_dim_mean(x) = dropdims(mean(x; dims=2); dims=2)

"""
    should_type_assert(x)

In certain cases, to ensure type-stability we want to add type-asserts. But this won't work
for exotic types like `ForwardDiff.Dual`. We use this function to check if we should add a
type-assert for `x`.
"""
should_type_assert(x::AbstractArray{T}) where {T} = isbitstype(T) && parent(x) === x
should_type_assert(::AbstractArray{<:ForwardDiff.Dual}) = false
should_type_assert(::ForwardDiff.Dual) = false
should_type_assert(x) = true

unsqueeze1(x::AbstractArray) = reshape(x, 1, size(x)...)
unsqueezeN(x::AbstractArray) = reshape(x, size(x)..., 1)

catN(x::AbstractArray{T,N}, y::AbstractArray{T,N}) where {T,N} = cat(x, y; dims=Val(N))

mapreduce_stack(xs) = mapreduce(unsqueezeN, catN, xs)

unwrap_val(x) = x
unwrap_val(::Val{T}) where {T} = T

function safe_warning(msg::AbstractString)
    @warn msg maxlog = 1
    return nothing
end

safe_kron(a, b) = safe_kron(get_device_type((a, b)), a, b)

safe_kron(T::Type{<:AbstractDevice}, a, b) = safe_kron_internal.(Ref(T), a, b)

function safe_kron(::Type{ReactantDevice}, a::AbstractVector, b::AbstractVector)
    res = [safe_kron_internal(ReactantDevice, first(a), first(b))]
    for (aᵢ, bᵢ) in zip(a[2:end], b[2:end])
        push!(res, safe_kron_internal(ReactantDevice, aᵢ, bᵢ))
    end
    return res
end

function safe_kron_internal(
    ::Type{<:Union{CPUDevice,ReactantDevice,CUDADevice}},
    a::AbstractVector,
    b::AbstractVector,
)
    return kron(a, b)
end
function safe_kron_internal(::Type{D}, a::AbstractVector, b::AbstractVector) where {D}
    safe_warning("`kron` is not supported on $(D). Falling back to `kron` on CPU.")
    a_cpu = CPUDevice()(a)
    b_cpu = CPUDevice()(b)
    return get_device((a, b))(safe_kron_internal(CPUDevice, a_cpu, b_cpu))
end

struct DataTransferBarrier{V}
    val::V
end

MLDataDevices.isleaf(::DataTransferBarrier) = true

end
