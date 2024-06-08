# Only tests that are not run via `vision` or other higher-level test suites are
# included in this snippet.
@testitem "MLP" setup=[SharedTestSetup] tags=[:layers] begin
    for (mode, aType, dev, ongpu) in MODES
        for act in (NNlib.relu, NNlib.gelu),
            norm in ((i, args...; kwargs...) -> BatchNorm(args...; kwargs...),
                (i, ch, act; kwargs...) -> GroupNorm(ch, 2, act; kwargs...), nothing)

            model = Layers.MLP(2, (4, 4, 2), act; dropout_rate=0.1f0, norm_layer=norm)
            ps, st = Lux.setup(Random.default_rng(), model) |> dev

            x = randn(Float32, 2, 2) |> aType

            @jet model(x, ps, st)

            __f = (x, ps) -> sum(abs2, first(model(x, ps, st)))
            @eval @test_gradients $(__f) $x $ps gpu_testing=$(ongpu) atol=1e-3 rtol=1e-3
        end
    end
end
