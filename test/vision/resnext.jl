using Boltz, Test, Metalhead

include("testutils.jl")

@testset "ResNeXt: $(depth) : $(cardinality) : $(width) : pretrained: $(pretrained)" for (
        depth, cardinality, width
    ) in [
        (50, 32, 4), (101, 32, 4), (152, 32, 4)
    ],
    pretrained in [false, true]

    depth == 152 && pretrained && continue
    pretrained && pkgversion(Metalhead) > v"0.9.4" && continue

    model = Vision.ResNeXt(depth; pretrained, cardinality, base_width=width)
    VisionTestUtils.test_model(model; pretrained)
end
