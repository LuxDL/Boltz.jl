using Boltz, Lux, StableRNGs, Test, LuxTestUtils
using DynamicExpressions, ForwardDiff, ComponentArrays

include("../testutils.jl")

operators = OperatorEnum(; binary_operators=[+, -, *], unary_operators=[cos])

x1 = Node(; feature=1)
x2 = Node(; feature=2)

expr_1 = x1 * cos(x2 - 3.2)
expr_2 = x2 - x1 * x2 + 2.5 - 1.0 * x1

@testset "expressions: $(exprs)" for exprs in
                                     ((expr_1,), (expr_1, expr_2), ([expr_1, expr_2],))
    layer = Layers.DynamicExpressionsLayer(operators, exprs...)
    ps, st = Lux.setup(StableRNG(0), layer)

    x = Float32[
        1.0 2.0 3.0
        4.0 5.0 6.0
    ]

    y, st_ = layer(x, ps, st)
    @test eltype(y) == Float32

    @test_gradients(
        TestUtils.sumabs2first,
        Constant(layer),
        x,
        ps,
        Constant(st);
        atol=1.0f-2,
        rtol=1.0f-2,
        skip_backends=[AutoEnzyme()]
    )

    # Test ForwardDiff dispatches
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

    @test dx ≈ dxps.x atol = 1.0f-3 rtol = 1.0f-3
    @test dps_ca ≈ dxps.ps atol = 1.0f-3 rtol = 1.0f-3

    # Test with Float64
    x64 = Float64.(x)
    y64, st_ = layer(x64, ps, st)
    @test eltype(y64) == Float64

    @test_gradients(
        TestUtils.sumabs2first,
        Constant(layer),
        x64,
        ps,
        Constant(st);
        atol=1.0e-2,
        rtol=1.0e-2,
        skip_backends=[AutoEnzyme()]
    )
end
