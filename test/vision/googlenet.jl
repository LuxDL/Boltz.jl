using Boltz, Test

include("testutils.jl")

@testset "GoogleNet: pretrained: $(pretrained)" for pretrained in [false]
    model = Vision.GoogleNet(; pretrained)
    VisionTestUtils.test_model(model; pretrained)
end
