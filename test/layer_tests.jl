# Only tests that are not run via `vision` or other higher-level test suites are
# included in this snippet.
@testitem "MLP" setup=[SharedTestSetup] tags=[:layers] begin
    @testset "$(mode)" for (mode, aType, dev, ongpu) in MODES
        @testset "$(act)" for act in (tanh, NNlib.gelu)
            @testset "nType" for nType in (BatchNorm, GroupNorm, nothing)
                norm = if nType === nothing
                    nType
                elseif nType === BatchNorm
                    (i, ch, act; kwargs...) -> BatchNorm(ch, act; kwargs...)
                elseif nType === GroupNorm
                    (i, ch, act; kwargs...) -> GroupNorm(ch, 2, act; kwargs...)
                end

                model = Layers.MLP(2, (4, 4, 2), act; norm_layer=norm)
                ps, st = Lux.setup(StableRNG(0), model) |> dev

                x = randn(Float32, 2, 2) |> aType

                @jet model(x, ps, st)

                __f = (x, ps) -> sum(abs2, first(model(x, ps, st)))
                test_gradients(__f, x, ps; atol=1e-3, rtol=1e-3)
            end
        end
    end
end

@testitem "Hamiltonian Neural Network" setup=[SharedTestSetup] tags=[:layers] begin
    using ComponentArrays, ForwardDiff, Zygote

    _remove_nothing(xs) = map(x -> x === nothing ? 0 : x, xs)

    @testset "$(mode): $(autodiff)" for (mode, aType, dev, ongpu) in MODES,
        autodiff in (nothing, AutoZygote(), AutoForwardDiff())

        ongpu && autodiff === AutoForwardDiff() && continue

        hnn = Layers.HamiltonianNN{true}(Layers.MLP(2, (4, 4, 2), NNlib.gelu); autodiff)
        ps, st = Lux.setup(StableRNG(0), hnn) |> dev

        x = randn(Float32, 2, 4) |> aType

        @test_throws ArgumentError hnn(x, ps, st)

        hnn = Layers.HamiltonianNN{true}(Layers.MLP(2, (4, 4, 1), NNlib.gelu); autodiff)
        ps, st = Lux.setup(StableRNG(0), hnn) |> dev
        ps_ca = ComponentArray(ps |> cpu_device()) |> dev

        @test st.first_call
        y, st = hnn(x, ps, st)
        @test !st.first_call

        ∂x_zyg, ∂ps_zyg = Zygote.gradient(
            (x, ps) -> sum(abs2, first(hnn(x, ps, st))), x, ps)
        @test ∂x_zyg !== nothing
        @test ∂ps_zyg !== nothing
        if !ongpu
            ∂ps_zyg = _remove_nothing(getdata(ComponentArray(∂ps_zyg |> cpu_device()) |>
                                              dev))
            ∂x_fd = ForwardDiff.gradient(x -> sum(abs2, first(hnn(x, ps, st))), x)
            ∂ps_fd = getdata(ForwardDiff.gradient(
                ps -> sum(abs2, first(hnn(x, ps, st))), ps_ca))

            @test ∂x_zyg≈∂x_fd atol=1e-3 rtol=1e-3
            @test ∂ps_zyg≈∂ps_fd atol=1e-3 rtol=1e-3
        end

        st = Lux.initialstates(StableRNG(0), hnn) |> dev

        @test st.first_call
        y, st = hnn(x, ps_ca, st)
        @test !st.first_call

        ∂x_zyg, ∂ps_zyg = Zygote.gradient(
            (x, ps) -> sum(abs2, first(hnn(x, ps, st))), x, ps_ca)
        @test ∂x_zyg !== nothing
        @test ∂ps_zyg !== nothing
        if !ongpu
            ∂ps_zyg = _remove_nothing(getdata(ComponentArray(∂ps_zyg |> cpu_device()) |>
                                              dev))
            ∂x_fd = ForwardDiff.gradient(x -> sum(abs2, first(hnn(x, ps_ca, st))), x)
            ∂ps_fd = getdata(ForwardDiff.gradient(
                ps -> sum(abs2, first(hnn(x, ps, st))), ps_ca))

            @test ∂x_zyg≈∂x_fd atol=1e-3 rtol=1e-3
            @test ∂ps_zyg≈∂ps_fd atol=1e-3 rtol=1e-3
        end
    end
end

@testitem "Tensor Product Layer" setup=[SharedTestSetup] tags=[:layers] begin
    @testset "$(mode)" for (mode, aType, dev, ongpu) in MODES
        @testset "$(basis)" for basis in (Basis.Chebyshev, Basis.Sin, Basis.Cos,
            Basis.Fourier, Basis.Legendre, Basis.Polynomial)
            tensor_project = Layers.TensorProductLayer([basis(n + 2) for n in 1:3], 4)
            ps, st = Lux.setup(StableRNG(0), tensor_project) |> dev

            x = tanh.(randn(Float32, 2, 4, 5)) |> aType

            @test_throws ArgumentError tensor_project(x, ps, st)

            x = tanh.(randn(Float32, 2, 3, 5)) |> aType

            y, st = tensor_project(x, ps, st)
            @test size(y) == (2, 4, 5)

            @jet tensor_project(x, ps, st)

            __f = (x, ps) -> sum(abs2, first(tensor_project(x, ps, st)))
            test_gradients(__f, x, ps; atol=1e-3, rtol=1e-3,
                skip_backends=[AutoTracker(), AutoEnzyme()])
        end
    end
end

@testitem "Basis Functions" setup=[SharedTestSetup] tags=[:layers] begin
    @testset "$(mode)" for (mode, aType, dev, ongpu) in MODES
        @testset "$(basis)" for basis in (Basis.Chebyshev, Basis.Sin, Basis.Cos,
            Basis.Fourier, Basis.Legendre, Basis.Polynomial)
            x = tanh.(randn(Float32, 2, 4)) |> aType
            grid = collect(1:3) |> aType

            fn = basis(3)
            @test size(fn(x)) == (3, 2, 4)
            @jet fn(x)
            @test size(fn(x, grid)) == (3, 2, 4)
            @jet fn(x, grid)

            fn = basis(3; dim=2)
            @test size(fn(x)) == (2, 3, 4)
            @jet fn(x)
            @test size(fn(x, grid)) == (2, 3, 4)
            @jet fn(x, grid)

            fn = basis(3; dim=3)
            @test size(fn(x)) == (2, 4, 3)
            @jet fn(x)
            @test size(fn(x, grid)) == (2, 4, 3)
            @jet fn(x, grid)

            fn = basis(3; dim=4)
            @test_throws ArgumentError fn(x)

            grid = 1:5 |> aType
            @test_throws ArgumentError fn(x, grid)
        end
    end
end

@testitem "Spline Layer" setup=[SharedTestSetup] tags=[:layers] begin
    using ComponentArrays, DataInterpolations, ForwardDiff, Zygote

    @testset "$(mode)" for (mode, aType, dev, ongpu) in MODES
        ongpu && continue

        @testset "$(spl): train_grid $(train_grid), dims $(dims)" for spl in (
                ConstantInterpolation, LinearInterpolation,
                QuadraticInterpolation, QuadraticSpline, CubicSpline),
            train_grid in (true, false),
            dims in ((), (8,))

            spline = Layers.SplineLayer(dims, 0.0f0, 1.0f0, 0.1f0, spl; train_grid)
            ps, st = Lux.setup(StableRNG(0), spline) |> dev
            ps_ca = ComponentArray(ps |> cpu_device()) |> dev

            x = tanh.(randn(Float32, 4)) |> aType

            y, st = spline(x, ps, st)
            @test size(y) == (dims..., 4)

            opt_broken = !ongpu && dims != () && spl !== ConstantInterpolation

            @jet spline(x, ps, st)

            y, st = spline(x, ps_ca, st)
            @test size(y) == (dims..., 4)

            @jet spline(x, ps_ca, st)

            ∂x, ∂ps = Zygote.gradient((x, ps) -> sum(abs2, first(spline(x, ps, st))), x, ps)
            spl !== ConstantInterpolation && @test ∂x !== nothing
            @test ∂ps !== nothing

            ∂x_fd = ForwardDiff.gradient(x -> sum(abs2, first(spline(x, ps, st))), x)
            ∂ps_fd = ForwardDiff.gradient(ps -> sum(abs2, first(spline(x, ps, st))), ps_ca)

            spl !== ConstantInterpolation && @test ∂x≈∂x_fd atol=1e-3 rtol=1e-3

            @test ∂ps.saved_points≈∂ps_fd.saved_points atol=1e-3 rtol=1e-3
            if train_grid
                if ∂ps.grid === nothing
                    @test all(Base.Fix1(isapprox, 0), ∂ps_fd.grid)
                else
                    @test ∂ps.grid≈∂ps_fd.grid atol=1e-3 rtol=1e-3
                end
            end
        end
    end
end
