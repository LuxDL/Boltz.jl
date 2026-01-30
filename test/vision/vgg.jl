using Boltz, Test

include("testutils.jl")

@testset "VGG: $(depth): pretrained: $(pretrained) batchnorm: $(batchnorm)" for depth in [
        11, 13, 16, 19
    ],
    pretrained in [false, true],
    batchnorm in [false, true]

    model = Vision.VGG(depth; pretrained, batchnorm)
    VisionTestUtils.test_model(model; pretrained)
end
