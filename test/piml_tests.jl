@testitem "Transolver" setup = [SharedTestSetup] tags = [:piml] begin
    for (mode, aType, dev) in MODES
        @testset "Base Model" begin
            model = PIML.Transolver(;
                func_dim=6, spatial_dim=3, nheads=8, num_layers=1, out_dim=2
            )
            ps, st = dev(Lux.setup(Random.default_rng(), model))
            x = aType(randn(Float32, 3, 32, 4))
            fx = aType(randn(Float32, 6, 32, 4))

            y, _ = model((x, fx), ps, st)
            @test size(y) == (2, 32, 4)

            if test_reactant(mode)
                set_reactant_backend!(mode)

                rdev = reactant_device(; force=true)

                ps_ra, st_ra = rdev(cpu_device()((ps, st)))
                x_ra, fx_ra = rdev(cpu_device()((x, fx)))

                Reactant.with_config(;
                    dot_general_precision=PrecisionConfig.HIGH,
                    convolution_precision=PrecisionConfig.HIGH,
                ) do
                    @test @jit(model((x_ra, fx_ra), ps_ra, st_ra))[1] ≈
                        model((x, fx), ps, st)[1] atol = 1e-3 rtol = 1e-3
                end
            end
        end

        @testset "Conditioned Model" begin
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

            model = PIML.Transolver(; nheads=8, num_layers=1, out_dim=2, preprocess)
            ps, st = dev(Lux.setup(Random.default_rng(), model))

            x = aType(randn(Float32, 3, 32, 4))
            fx = aType(randn(Float32, 6, 4)) # Global information about the mesh

            y, _ = model((x, fx), ps, st)
            @test size(y) == (2, 32, 4)

            if test_reactant(mode)
                set_reactant_backend!(mode)

                rdev = reactant_device(; force=true)

                ps_ra, st_ra = rdev(cpu_device()((ps, st)))
                x_ra, fx_ra = rdev(cpu_device()((x, fx)))

                Reactant.with_config(;
                    dot_general_precision=PrecisionConfig.HIGH,
                    convolution_precision=PrecisionConfig.HIGH,
                ) do
                    @test @jit(model((x_ra, fx_ra), ps_ra, st_ra))[1] ≈
                        model((x, fx), ps, st)[1] atol = 1e-3 rtol = 1e-3
                end
            end
        end
    end
end
