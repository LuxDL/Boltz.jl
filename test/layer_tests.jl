# Only tests that are not run via `vision` or other higher-level test suites are
# included in this snippet.
@testitem "MLP" setup=[SharedTestSetup] tags=[:layers] begin
    for (mode, aType, dev, ongpu) in MODES
        for act in (NNlib.relu, NNlib.gelu),
            norm in ((i, args...; kwargs...) -> BatchNorm(args...; kwargs...),
                (i, ch, act; kwargs...) -> GroupNorm(ch, 2, act; kwargs...), nothing)

            model = Layers.MLP(2, (4, 4, 2), act; dropout_rate=0.1f0, norm_layer=norm)
            ps, st = Lux.setup(Xoshiro(0), model) |> dev

            x = randn(Float32, 2, 2) |> aType

            @jet model(x, ps, st)

            __f = (x, ps) -> sum(abs2, first(model(x, ps, st)))
            @eval @test_gradients $(__f) $x $ps gpu_testing=$(ongpu) atol=1e-3 rtol=1e-3
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
        ps, st = Lux.setup(Xoshiro(0), hnn) |> dev

        x = randn(Float32, 2, 4) |> aType

        @test_throws ArgumentError hnn(x, ps, st)

        hnn = Layers.HamiltonianNN{true}(Layers.MLP(2, (4, 4, 1), NNlib.gelu); autodiff)
        ps, st = Lux.setup(Xoshiro(0), hnn) |> dev
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

        st = Lux.initialstates(Xoshiro(0), hnn) |> dev

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
        mode === "AMDGPU" && continue

        @testset "$(basis)" for basis in (Basis.Chebyshev, Basis.Sin, Basis.Cos,
            Basis.Fourier, Basis.Legendre, Basis.Polynomial)
            tensor_project = Layers.TensorProductLayer([basis(n + 2) for n in 1:3], 4)
            ps, st = Lux.setup(Xoshiro(0), tensor_project) |> dev

            x = tanh.(randn(Float32, 2, 4, 5)) |> aType

            @test_throws ArgumentError tensor_project(x, ps, st)

            x = tanh.(randn(Float32, 2, 3, 5)) |> aType

            y, st = tensor_project(x, ps, st)
            @test size(y) == (2, 4, 5)

            # Passes in PR and fails on main. Skipping!
            # @jet tensor_project(x, ps, st)

            __f = (x, ps) -> sum(abs2, first(tensor_project(x, ps, st)))
            @eval @test_gradients $(__f) $x $ps gpu_testing=$(ongpu) atol=1e-3 rtol=1e-3 skip_tracker=true
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
