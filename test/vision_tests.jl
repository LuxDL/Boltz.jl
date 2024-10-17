@testsetup module PretrainedWeightsTestSetup

using Lux, Downloads, JLD2

function normalize_imagenet(data)
    cmean = reshape(Float32[0.485, 0.456, 0.406], (1, 1, 3, 1))
    cstd = reshape(Float32[0.229, 0.224, 0.225], (1, 1, 3, 1))
    return (data .- cmean) ./ cstd
end

# The images are normalized and saved
@load joinpath(@__DIR__, "testimages", "monarch_color.jld2") monarch_color_224 monarch_color_256
const MONARCH_224 = monarch_color_224
const MONARCH_256 = monarch_color_256

const TEST_LBLS = readlines(Downloads.download(
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
))

function imagenet_acctest(model, ps, st, dev; size=224)
    ps = ps |> dev
    st = Lux.testmode(st) |> dev
    TEST_X = size == 224 ? MONARCH_224 :
             (size == 256 ? MONARCH_256 : error("size must be 224 or 256"))
    x = TEST_X |> dev
    ypred = first(model(x, ps, st)) |> collect |> vec
    top5 = TEST_LBLS[sortperm(ypred; rev=true)]
    return "monarch" in top5
end

export imagenet_acctest

end

@testitem "AlexNet" setup=[SharedTestSetup, PretrainedWeightsTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES
        @testset "pretrained: $(pretrained)" for pretrained in [true, false]
            model = Vision.AlexNet(; pretrained)
            ps, st = Lux.setup(Random.default_rng(), model) |> dev
            st = Lux.testmode(st)
            img = randn(Float32, 224, 224, 3, 2) |> aType

            @jet model(img, ps, st)
            @test size(first(model(img, ps, st))) == (1000, 2)

            if pretrained
                @test imagenet_acctest(model, ps, st, dev)
            end

            GC.gc(true)
        end
    end
end

@testitem "ConvMixer" setup=[SharedTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES, name in [:small, :base, :large]

        model=Vision.ConvMixer(name; pretrained=false)
        ps, st=Lux.setup(Random.default_rng(), model)|>dev
        st=Lux.testmode(st)
        img=randn(Float32, 256, 256, 3, 2)|>aType

        @jet model(img, ps, st)
        @test size(first(model(img, ps, st))) == (1000, 2)

        GC.gc(true)
    end
end

@testitem "GoogLeNet" setup=[SharedTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES
        model=Vision.GoogLeNet(; pretrained=false)
        ps, st=Lux.setup(Random.default_rng(), model)|>dev
        st=Lux.testmode(st)
        img=randn(Float32, 224, 224, 3, 2)|>aType

        @jet model(img, ps, st)
        @test size(first(model(img, ps, st))) == (1000, 2)

        GC.gc(true)
    end
end

@testitem "MobileNet" setup=[SharedTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES, name in [:v1, :v2, :v3_small, :v3_large]

        model=Vision.MobileNet(name; pretrained=false)
        ps, st=Lux.setup(Random.default_rng(), model)|>dev
        st=Lux.testmode(st)
        img=randn(Float32, 224, 224, 3, 2)|>aType

        @jet model(img, ps, st)
        @test size(first(model(img, ps, st))) == (1000, 2)

        GC.gc(true)
    end
end

@testitem "ResNet" setup=[SharedTestSetup, PretrainedWeightsTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES, depth in [18, 34, 50, 101, 152]

        @testset for pretrained in [false, true]
            model = Vision.ResNet(depth; pretrained)
            ps, st = Lux.setup(Random.default_rng(), model) |> dev
            st = Lux.testmode(st)
            img = randn(Float32, 224, 224, 3, 2) |> aType

            @jet model(img, ps, st)
            @test size(first(model(img, ps, st))) == (1000, 2)

            if pretrained
                @test imagenet_acctest(model, ps, st, dev)
            end

            GC.gc(true)
        end
    end
end

@testitem "ResNeXt" setup=[SharedTestSetup, PretrainedWeightsTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES
        @testset for (depth, cardinality, base_width) in [
            (50, 32, 4), (101, 32, 8), (101, 64, 4), (152, 64, 4)]
            @testset for pretrained in [false, true]
                depth == 152 && pretrained && continue

                model = Vision.ResNeXt(depth; pretrained, cardinality, base_width)
                ps, st = Lux.setup(Random.default_rng(), model) |> dev
                st = Lux.testmode(st)
                img = randn(Float32, 224, 224, 3, 2) |> aType

                @jet model(img, ps, st)
                @test size(first(model(img, ps, st))) == (1000, 2)

                if pretrained
                    @test imagenet_acctest(model, ps, st, dev)
                end

                GC.gc(true)
            end
        end
    end
end

@testitem "WideResNet" setup=[SharedTestSetup, PretrainedWeightsTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES, depth in [50, 101, 152]

        @testset for pretrained in [false, true]
            depth == 152 && pretrained && continue

            model = Vision.WideResNet(depth; pretrained)
            ps, st = Lux.setup(Random.default_rng(), model) |> dev
            st = Lux.testmode(st)
            img = randn(Float32, 224, 224, 3, 2) |> aType

            @jet model(img, ps, st)
            @test size(first(model(img, ps, st))) == (1000, 2)

            if pretrained
                @test imagenet_acctest(model, ps, st, dev)
            end

            GC.gc(true)
        end
    end
end

@testitem "SqueezeNet" setup=[SharedTestSetup, PretrainedWeightsTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES
        @testset for pretrained in [false, true]
            model = Vision.SqueezeNet(; pretrained)
            ps, st = Lux.setup(Random.default_rng(), model) |> dev
            st = Lux.testmode(st)
            img = randn(Float32, 224, 224, 3, 2) |> aType

            @jet model(img, ps, st)
            @test size(first(model(img, ps, st))) == (1000, 2)

            if pretrained
                @test imagenet_acctest(model, ps, st, dev)
            end

            GC.gc(true)
        end
    end
end

@testitem "VGG" setup=[SharedTestSetup, PretrainedWeightsTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES, depth in [11, 13, 16, 19]

        @testset for pretrained in [false, true], batchnorm in [false, true]

            model = Vision.VGG(depth; batchnorm, pretrained)
            ps, st = Lux.setup(Random.default_rng(), model) |> dev
            st = Lux.testmode(st)
            img = randn(Float32, 224, 224, 3, 2) |> aType

            @jet model(img, ps, st)
            @test size(first(model(img, ps, st))) == (1000, 2)

            if pretrained
                @test imagenet_acctest(model, ps, st, dev)
            end

            GC.gc(true)
        end
    end
end

@testitem "VisionTransformer" setup=[SharedTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES, name in [:tiny, :small, :base]
        # :large, :huge, :giant, :gigantic --> too large for CI
        model=Vision.VisionTransformer(name; pretrained=false)
        ps, st=Lux.setup(Random.default_rng(), model)|>dev
        st=Lux.testmode(st)
        img=randn(Float32, 256, 256, 3, 2)|>aType

        @jet model(img, ps, st)
        @test size(first(model(img, ps, st))) == (1000, 2)

        model=Vision.VisionTransformer(name; pretrained=false)
        ps, st=Lux.setup(Random.default_rng(), model)|>dev
        st=Lux.testmode(st)
        img=randn(Float32, 256, 256, 3, 2)|>aType

        @jet model(img, ps, st)
        @test size(first(model(img, ps, st))) == (1000, 2)

        GC.gc(true)
    end
end
