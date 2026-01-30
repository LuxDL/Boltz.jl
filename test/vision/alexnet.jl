using Boltz, Test

include("testutils.jl")

@testset "AlexNet: pretrained: $(pretrained)" for pretrained in [true, false]
    model = Vision.AlexNet(; pretrained)
    VisionTestUtils.test_model(model; pretrained)
end
