module BoltzDataInterpolationsExt

using DataInterpolations: AbstractInterpolation

using Boltz: Boltz, Layers, Utils

for train_grid in (true, false), tType in (AbstractVector, Number)

    grid_expr = train_grid ? :(grid = ps.grid) : :(grid = st.grid)
    sol_expr = tType === Number ? :(sol = interp(t)) : :(sol = interp.(t))
    @eval function (spl::Layers.SplineLayer{$(train_grid), Basis})(
            t::$(tType), ps, st) where {Basis <: AbstractInterpolation}
        $(grid_expr)
        interp = construct_basis(Basis, ps.saved_points, grid; extrapolate=true)
        $(sol_expr)
        spl.in_dims == () && return sol, st
        return Utils.mapreduce_stack(sol), st
    end
end

function construct_basis(
        ::Type{Basis}, saved_points::AbstractVector, grid; extrapolate=false) where {Basis}
    return Basis(saved_points, grid; extrapolate)
end

function construct_basis(::Type{Basis}, saved_points::AbstractArray{T, N},
        grid; extrapolate=false) where {Basis, T, N}
    return construct_basis(
        # Unfortunately DataInterpolations.jl is not very robust to different array types
        # so we have to make a copy
        Basis, [copy(selectdim(saved_points, N, i)) for i in 1:size(saved_points, N)],
        grid; extrapolate)
end

end
