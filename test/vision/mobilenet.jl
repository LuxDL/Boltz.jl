using Boltz, Test

include("testutils.jl")

@testset "MobileNet: $(name): pretrained: $(pretrained)" for name in [
        :v1, :v2, :v3_small, :v3_large
    ],
    pretrained in [false]

    model = Vision.MobileNet(name; pretrained)
    VisionTestUtils.test_model(model; pretrained)
end
