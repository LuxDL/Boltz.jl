module BoltzDataInterpolationsExt

using Boltz: Boltz, Layers, Utils
using DataInterpolations: AbstractInterpolation

for train_grid in (true, false)
    grid_expr = train_grid ? :(grid = ps.grid) : :(grid = st.grid)
    @eval function (spl::Layers.SplineLayer{$(train_grid), Basis})(
            t::AbstractVector, ps, st) where {Basis <: AbstractInterpolation}
        $(grid_expr)
        interp = __construct_basis(Basis, ps.saved_points, grid; extrapolate=true)
        sol = interp.(t)
        spl.in_dims == () && return sol, st
        return Utils.mapreduce_stack(sol), st
    end
end

@inline function __construct_basis(
        ::Type{Basis}, saved_points::AbstractVector, grid; extrapolate=false) where {Basis}
    return Basis(saved_points, grid; extrapolate)
end

@inline function __construct_basis(::Type{Basis}, saved_points::AbstractArray{T, N},
        grid; extrapolate=false) where {Basis, T, N}
    return __construct_basis(
        # Unfortunately DataInterpolations.jl is not very robust to different array types
        # so we have to make a copy
        Basis, [copy(selectdim(saved_points, N, i)) for i in 1:size(saved_points, N)],
        grid; extrapolate)
end

end
