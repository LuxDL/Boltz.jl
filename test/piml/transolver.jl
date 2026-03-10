using Reactant, Boltz, Lux, StableRNGs, Test, LuxTestUtils

include("../testutils.jl")

rdev = reactant_device(; force=true)

@testset "Base Model: $(use_rms_norm)" for use_rms_norm in (true, false)
    model = PIML.Transolver(;
        func_dim=6, spatial_dim=3, nheads=8, num_layers=1, out_dim=2, use_rms_norm
    )
    ps, st = Lux.setup(StableRNG(0), model)
    x = randn(Float32, 3, 32, 4)
    fx = randn(Float32, 6, 32, 4)

    y, _ = model((x, fx), ps, st)
    @test size(y) == (2, 32, 4)

    ps_ra, st_ra = rdev((ps, st))
    x_ra, fx_ra = rdev((x, fx))

    @test @jit(model((x_ra, fx_ra), ps_ra, st_ra))[1] ≈ model((x, fx), ps, st)[1] atol =
        1e-3 rtol = 1e-3
end

@testset "Conditioned Model: $(use_rms_norm)" for use_rms_norm in (true, false)
    hidden_dim = 128
    activation = gelu
    preprocess = Parallel(
        .+,
        Dense(3 => hidden_dim, activation),
        Chain(
            Dense(6 => hidden_dim, activation),
            WrappedFunction(x -> reshape(x, hidden_dim, 1, size(x, ndims(x)))),
        ),
    )

    model = PIML.Transolver(; nheads=8, num_layers=1, out_dim=2, preprocess, use_rms_norm)
    ps, st = Lux.setup(StableRNG(0), model)

    x = randn(Float32, 3, 32, 4)
    fx = randn(Float32, 6, 4) # Global information about the mesh

    y, _ = model((x, fx), ps, st)
    @test size(y) == (2, 32, 4)

    ps_ra, st_ra = rdev((ps, st))
    x_ra, fx_ra = rdev((x, fx))

    @test @jit(model((x_ra, fx_ra), ps_ra, st_ra))[1] ≈ model((x, fx), ps, st)[1] atol =
        1e-3 rtol = 1e-3
end
