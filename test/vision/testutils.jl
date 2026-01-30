module VisionTestUtils

using Lux, Downloads, JLD2, Pickle, Reactant, StableRNGs, Reactant, Test

function normalize_imagenet(data)
    cmean = reshape(Float32[0.485, 0.456, 0.406], (1, 1, 3, 1))
    cstd = reshape(Float32[0.229, 0.224, 0.225], (1, 1, 3, 1))
    return (data .- cmean) ./ cstd
end

# The images are normalized and saved
@load joinpath(@__DIR__, "../", "testimages", "monarch_color.jld2") monarch_color_224 monarch_color_256
const MONARCH_224 = monarch_color_224
const MONARCH_256 = monarch_color_256

const TEST_LBLS = readlines(
    Downloads.download(
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    ),
)

function get_test_image(size)
    size == 224 && return MONARCH_224
    size == 256 && return MONARCH_256
    return error("size must be 224 or 256")
end

function imagenet_acctest(model, ps, st, dev; size=224)
    ps = dev(ps)
    st = dev(Lux.testmode(st))
    TEST_X = get_test_image(size)
    x = dev(TEST_X)

    if dev isa ReactantDevice
        res = @jit model(x, ps, st)
    else
        res = model(x, ps, st)
    end

    ypred = vec(collect(cpu_device()(first(res))))
    top5 = TEST_LBLS[partialsortperm(ypred, 1:5; rev=true)]
    return "monarch" in top5
end

function test_model(model; size=224, pretrained::Bool=false, nclasses::Int=1000)
    ps, st = Lux.setup(StableRNGs.StableRNG(1234), model)
    st = Lux.testmode(st)
    img = get_test_image(size)

    res = first(model(img, ps, st))
    @test size(res) == (nclasses, 1)

    if pretrained
        @test imagenet_acctest(model, ps, st, CPUDevice(); size)
    end

    GC.gc(true)

    rdev = reactant_device()
    ps_ra, st_ra, img_ra = rdev((ps, st, img))
    res_ra = first(@jit model(img_ra, ps_ra, st_ra))
    @test res_ra ≈ res atol = 1e-3 rtol = 1e-3

    if pretrained
        @test imagenet_acctest(model, ps, st, rdev; size)
    end
end

end
