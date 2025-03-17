module PytorchLoadUtils

function rebuild_conv_filter!(dst::AbstractArray{<:Any,4}, src::AbstractArray{<:Any,4})
    shape = size(dst)
    filter_x, filter_y = shape[1:2] .+ 1
    @inbounds for (i, j, k, m) in Iterators.product([1:s for s in shape]...)
        dst[filter_x - i, filter_y - j, k, m] = src[m, k, j, i]
    end
end

end
