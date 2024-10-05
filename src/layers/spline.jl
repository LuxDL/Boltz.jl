"""
    SplineLayer(in_dims, grid_min, grid_max, grid_step, basis::Type{Basis};
        train_grid::Union{Val, Bool}=Val(false), init_saved_points=nothing)

Constructs a spline layer with the given basis function.

## Arguments

  - `in_dims`: input dimensions of the layer. This must be a tuple of integers, to construct
    a flat vector of saved_points pass in `()`.

  - `grid_min`: minimum value of the grid.
  - `grid_max`: maximum value of the grid.
  - `grid_step`: step size of the grid.
  - `basis`: basis function to use for the interpolation. Currently only the basis functions
    from DataInterpolations.jl are supported:

     1. `ConstantInterpolation`
     2. `LinearInterpolation`
     3. `QuadraticInterpolation`
     4. `QuadraticSpline`
     5. `CubicSpline`

## Keyword Arguments

  - `train_grid`: whether to train the grid or not.
  - `init_saved_points`: values of the function at multiples of the time step. Initialized
    by default to a random vector sampled from the unit normal. Alternatively, can take a
    function with the signature
    `init_saved_points(rng, in_dims, grid_min, grid_max, grid_step)`.

!!! warning

    Currently this layer is limited since it relies on DataInterpolations.jl which doesn't
    work with GPU arrays. This will be fixed in the future by extending support to different
    basis functions.
"""
@concrete struct SplineLayer{TG, B, T} <: AbstractLuxLayer
    grid_min::T
    grid_max::T
    grid_step::T
    basis
    in_dims
    init_saved_points
end

function SplineLayer(in_dims::Dims, grid_min, grid_max, grid_step, basis::Type{Basis};
        train_grid::Union{Val, Bool}=Val(false), init_saved_points=nothing) where {Basis}
    return SplineLayer{unwrap_val(train_grid), Basis}(
        grid_min, grid_max, grid_step, basis, in_dims, init_saved_points)
end

function LuxCore.initialparameters(
        rng::AbstractRNG, layer::SplineLayer{TG, B, T}) where {TG, B, T}
    if layer.init_saved_points === nothing
        saved_points = rand(rng, T, layer.in_dims...,
            length((layer.grid_min):(layer.grid_step):(layer.grid_max)))
    else
        saved_points = layer.init_saved_points(
            rng, in_dims, layer.grid_min, layer.grid_max, layer.grid_step)
    end
    TG || return (; saved_points)
    return (;
        saved_points, grid=collect((layer.grid_min):(layer.grid_step):(layer.grid_max)))
end

function LuxCore.initialstates(::AbstractRNG, layer::SplineLayer{false})
    return (; grid=collect((layer.grid_min):(layer.grid_step):(layer.grid_max)),)
end
