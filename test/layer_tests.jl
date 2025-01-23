# Only tests that are not run via `vision` or other higher-level test suites are
# included in this snippet.
@testitem "MLP" setup=[SharedTestSetup] tags=[:layers] begin
    using NNlib

    @testset "$(mode)" for (mode, aType, dev, ongpu) in MODES
        @testset "$(act)" for act in (tanh, NNlib.gelu)
            @testset "$(nType)" for nType in (BatchNorm, GroupNorm, nothing)
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
                @test_gradients(__f, x, ps; atol=1e-3, rtol=1e-3,
                    soft_fail=[AutoFiniteDiff()], enzyme_set_runtime_activity=true)
            end
        end
    end
end

@testitem "Hamiltonian Neural Network" setup=[SharedTestSetup] tags=[:layers] begin
    using ComponentArrays, ForwardDiff, Zygote, MLDataDevices, NNlib

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
            @test_gradients(__f, x, ps; atol=1e-3, rtol=1e-3,
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

@testitem "Spline Layer" setup=[SharedTestSetup] tags=[:integration] begin
    using ComponentArrays, DataInterpolations, ForwardDiff, Zygote, MLDataDevices

    @testset "$(mode)" for (mode, aType, dev, ongpu) in MODES
        ongpu && continue

        @testset "$(spl): train_grid $(train_grid), dims $(dims)" for spl in (
                ConstantInterpolation, LinearInterpolation,
                QuadraticInterpolation,
                # QuadraticSpline, # XXX: DataInterpolations.jl broke it again!!!
                CubicSpline),
            train_grid in (true, false),
            dims in ((), (8,))

            spline = Layers.SplineLayer(dims, 0.0f0, 1.0f0, 0.1f0, spl; train_grid)
            ps, st = Lux.setup(StableRNG(0), spline) |> dev
            ps_ca = ComponentArray(ps |> cpu_device()) |> dev

            x = tanh.(randn(Float32, 4)) |> aType

            y, st = spline(x, ps, st)
            @test size(y) == (dims..., 4)

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
                    @test_softfail all(Base.Fix1(isapprox, 0), ∂ps_fd.grid)
                else
                    @test ∂ps.grid≈∂ps_fd.grid atol=1e-3 rtol=1e-3
                end
            end
        end
    end
end

@testitem "Periodic Embedding" setup=[SharedTestSetup] tags=[:layers] begin
    @testset "$(mode)" for (mode, aType, dev, ongpu) in MODES
        layer = Layers.PeriodicEmbedding([2, 3], [4.0, π / 5])
        ps, st = Lux.setup(StableRNG(0), layer) |> dev
        x = randn(StableRNG(0), 6, 4, 3, 2) |> aType
        Δx = [0.0, 12.0, -2π / 5, 0.0, 0.0, 0.0] |> aType

        val = layer(x, ps, st)[1] |> Array
        shifted_val = layer(x .+ Δx, ps, st)[1] |> Array

        @test all(val[1:4, :, :, :] .== shifted_val[1:4, :, :, :]) && all(isapprox.(
            val[5:8, :, :, :], shifted_val[5:8, :, :, :]; atol=5 * eps(Float32)))

        @jet layer(x, ps, st)

        __f = x -> sum(first(layer(x, ps, st)))
        @test_gradients(__f, x; atol=1.0f-3, rtol=1.0f-3, enzyme_set_runtime_activity=true)
    end
end

@testitem "Dynamic Expressions Layer" setup=[SharedTestSetup] tags=[:integration] begin
    using DynamicExpressions, ForwardDiff, ComponentArrays, Bumper, LoopVectorization

    operators = OperatorEnum(; binary_operators=[+, -, *], unary_operators=[cos])

    x1 = Node(; feature=1)
    x2 = Node(; feature=2)

    expr_1 = x1 * cos(x2 - 3.2)
    expr_2 = x2 - x1 * x2 + 2.5 - 1.0 * x1

    for exprs in ((expr_1,), (expr_1, expr_2), ([expr_1, expr_2],)),
        turbo in (Val(false), Val(true)),
        bumper in (Val(false), Val(true))

        layer = Layers.DynamicExpressionsLayer(operators, exprs...; turbo, bumper)
        ps, st = Lux.setup(StableRNG(0), layer)

        x = [1.0f0 2.0f0 3.0f0
             4.0f0 5.0f0 6.0f0]

        y, st_ = layer(x, ps, st)
        @test eltype(y) == Float32
        __f = (x, p) -> sum(abs2, first(layer(x, p, st)))
        @test_gradients(__f, x, ps; atol=1.0f-3, rtol=1.0f-3, skip_backends=[AutoEnzyme()])

        # Particular ForwardDiff dispatches
        ps_ca = ComponentArray(ps)
        dps_ca = ForwardDiff.gradient(ps_ca) do ps_
            sum(abs2, first(layer(x, ps_, st)))
        end
        dx = ForwardDiff.gradient(x) do x_
            sum(abs2, first(layer(x_, ps, st)))
        end
        dxps = ForwardDiff.gradient(ComponentArray(; x, ps)) do ca
            sum(abs2, first(layer(ca.x, ca.ps, st)))
        end

        @test dx≈dxps.x atol=1.0f-3 rtol=1.0f-3
        @test dps_ca≈dxps.ps atol=1.0f-3 rtol=1.0f-3

        x = Float64.(x)
        y, st_ = layer(x, ps, st)
        @test eltype(y) == Float64
        __f = (x, p) -> sum(abs2, first(layer(x, p, st)))
        @test_gradients(__f, x, ps; atol=1.0e-3, rtol=1.0e-3, skip_backends=[AutoEnzyme()])
    end

    @testset "$(mode)" for (mode, aType, dev, ongpu) in MODES
        layer = Layers.DynamicExpressionsLayer(operators, expr_1)
        ps, st = Lux.setup(StableRNG(0), layer) |> dev

        x = [1.0f0 2.0f0 3.0f0
             4.0f0 5.0f0 6.0f0] |> aType

        if ongpu
            @test_throws ArgumentError layer(x, ps, st)
        end
    end
end

@testitem "Positive Definite Container" setup=[SharedTestSetup] tags=[:layers] begin
    using NNlib

    @testset "$(mode)" for (mode, aType, dev, ongpu) in MODES
        model = Layers.MLP(2, (4, 4, 2), NNlib.gelu)
        pd = Layers.PositiveDefinite(model; in_dims=2)
        ps, st = Lux.setup(StableRNG(0), pd) |> dev

        x = randn(StableRNG(0), Float32, 2, 2) |> aType
        x0 = zeros(Float32, 2) |> aType

        y, _ = pd(x, ps, st)
        z, _ = model(x, ps, st.model)
        z0, _ = model(x0, ps, st.model)
        y_by_hand = sum(abs2, z .- z0; dims = 1) .+ sum(abs2, x .- x0; dims = 1)

        @test maximum(abs, y - y_by_hand) < 1.0f-8

        @jet pd(x, ps, st)

        __f = (x, ps) -> sum(first(pd(x, ps, st)))
        @test_gradients(__f, x, ps; atol=1.0f-3, rtol=1.0f-3)
    end
end

@testitem "ShiftTo Container" setup=[SharedTestSetup] tags=[:layers] begin
    using NNlib

    @testset "$(mode)" for (mode, aType, dev, ongpu) in MODES
        model = Layers.MLP(2, (4, 4, 2), NNlib.gelu)
        s = Layers.ShiftTo(model, ones(2), zeros(2))
        ps, st = Lux.setup(StableRNG(0), s) |> dev

        y0, _ = s(st.in_val, ps, st)
        @test maximum(abs, y0) < 1.0f-8

        x = randn(StableRNG(0), Float32, 2, 2) |> aType
        @jet s(x, ps, st)

        __f = (x, ps) -> sum(first(s(x, ps, st)))
        @test_gradients(__f, x, ps; atol=1.0f-3, rtol=1.0f-3)
    end
end
