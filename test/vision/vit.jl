using Boltz, Test

variants = if parse(Bool, get(ENV, "CI", "false"))
    [:tiny, :small, :base]
else
    [:tiny, :small, :base, :large, :huge, :giant, :gigantic]
end

@testset "VisionTransformer: $(variant): pretrained: $(pretrained)" for variant in variants,
    pretrained in [false]

    model = Vision.VisionTransformer(variant; pretrained)
    VisionTestUtils.test_model(model; pretrained, size=256)
end
