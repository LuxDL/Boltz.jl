using Boltz, Test, Metalhead

include("testutils.jl")

@testset "ResNet: $(depth): pretrained: $(pretrained)" for depth in [18, 34, 50, 101, 152],
    pretrained in [false, true]

    pretrained && pkgversion(Metalhead) > v"0.9.4" && continue

    model = Vision.ResNet(depth; pretrained)
    VisionTestUtils.test_model(model; pretrained)
end
