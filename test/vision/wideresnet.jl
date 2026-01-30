using Boltz, Test, Metalhead

include("testutils.jl")

@testset "WideResNet: $(depth) : pretrained: $(pretrained)" for depth in [50, 101, 152],
    pretrained in [false, true]

    depth == 152 && pretrained && continue
    pretrained && pkgversion(Metalhead) > v"0.9.4" && continue

    model = Vision.WideResNet(depth; pretrained)
    VisionTestUtils.test_model(model; pretrained)
end
