using Boltz, Test, Metalhead

include("testutils.jl")

@testset "SqueezeNet: pretrained: $(pretrained)" for pretrained in [false, true]
    model = Vision.SqueezeNet(; pretrained)
    VisionTestUtils.test_model(model; pretrained)
end
