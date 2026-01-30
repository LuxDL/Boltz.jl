using Boltz, Test, Metalhead

include("testutils.jl")

@testset "GoogleNet: pretrained: $(pretrained)" for pretrained in [false]
    model = Vision.GoogLeNet(; pretrained)
    VisionTestUtils.test_model(model; pretrained)
end
