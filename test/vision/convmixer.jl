using Boltz, Test

include("testutils.jl")

@testset "ConvMixer: $(name): pretrained: $(pretrained)" for name in
                                                             [:small, :base, :large],
    pretrained in [false]

    model = Vision.ConvMixer(name; pretrained)
    VisionTestUtils.test_model(model; pretrained)
end
