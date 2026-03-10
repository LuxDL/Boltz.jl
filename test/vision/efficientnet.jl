using Boltz, Test

include("testutils.jl")

variants = if parse(Bool, get(ENV, "CI", "false"))
    [:b0, :b1]
else
    [:b0, :b1, :b2, :b3, :b4, :b5, :b6, :b7]
end

@testset "EfficientNet: $(variant): pretrained: $(pretrained)" for variant in variants,
    pretrained in [false, true]

    model = Vision.EfficientNet(variant; pretrained)
    VisionTestUtils.test_model(model; pretrained)
end
