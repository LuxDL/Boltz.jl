@testitem "AlexNet" setup=[SharedTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES
        @testset "pretrained: $(pretrained)" for pretrained in [true, false]
            model, ps, st = Vision.AlexNet(; pretrained)
            ps = ps |> dev
            st = Lux.testmode(st) |> dev
            img = randn(Float32, 224, 224, 3, 2) |> aType

            @jet model(img, ps, st)
            @test size(first(model(img, ps, st))) == (1000, 2)

            @test_deprecated alexnet(:alexnet)

            GC.gc(true)
        end
    end
end

@testitem "ConvMixer" setup=[SharedTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES, name in [:small, :base, :large]
        model, ps, st = Vision.ConvMixer(name; pretrained=false)
        ps = ps |> dev
        st = Lux.testmode(st) |> dev
        img = randn(Float32, 256, 256, 3, 2) |> aType

        @jet model(img, ps, st)
        @test size(first(model(img, ps, st))) == (1000, 2)

        @test_deprecated convmixer(name)

        GC.gc(true)
    end
end

@testitem "GoogLeNet" setup=[SharedTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES
        model, ps, st = Vision.GoogLeNet(; pretrained=false)
        ps = ps |> dev
        st = Lux.testmode(st) |> dev
        img = randn(Float32, 224, 224, 3, 2) |> aType

        @jet model(img, ps, st)
        @test size(first(model(img, ps, st))) == (1000, 2)

        @test_deprecated googlenet(:googlenet)

        GC.gc(true)
    end
end

@testitem "MobileNet" setup=[SharedTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES, name in [:v1, :v2, :v3_small, :v3_large]
        model, ps, st = Vision.MobileNet(name; pretrained=false)
        ps = ps |> dev
        st = Lux.testmode(st) |> dev
        img = randn(Float32, 224, 224, 3, 2) |> aType

        @jet model(img, ps, st)
        @test size(first(model(img, ps, st))) == (1000, 2)

        @test_deprecated mobilenet(Symbol("mobilenet_", name))

        GC.gc(true)
    end
end

@testitem "ResNet" setup=[SharedTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES, depth in [18, 34, 50, 101, 152]
        model, ps, st = Vision.ResNet(depth; pretrained=false)
        ps = ps |> dev
        st = Lux.testmode(st) |> dev
        img = randn(Float32, 224, 224, 3, 2) |> aType

        @jet model(img, ps, st)
        @test size(first(model(img, ps, st))) == (1000, 2)

        @test_deprecated resnet(Symbol("resnet", depth))

        GC.gc(true)
    end
end

@testitem "ResNeXt" setup=[SharedTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES, depth in [50, 101, 152]
        model, ps, st = Vision.ResNeXt(depth; pretrained=false)
        ps = ps |> dev
        st = Lux.testmode(st) |> dev
        img = randn(Float32, 224, 224, 3, 2) |> aType

        @jet model(img, ps, st)
        @test size(first(model(img, ps, st))) == (1000, 2)

        @test_deprecated resnext(Symbol("resnext", depth))

        GC.gc(true)
    end
end

@testitem "VGG" setup=[SharedTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES, depth in [11, 13, 16, 19]
        @testset "pretrained: $(pretrained), batchnorm: $(batchnorm)" for pretrained in [
                false, true],
            batchnorm in [false, true]

            model, ps, st = Vision.VGG(depth; batchnorm, pretrained)
            ps = ps |> dev
            st = Lux.testmode(st) |> dev
            img = randn(Float32, 224, 224, 3, 2) |> aType

            @jet model(img, ps, st)
            @test size(first(model(img, ps, st))) == (1000, 2)

            name = Symbol("vgg", depth, batchnorm ? "_bn" : "")
            @test_deprecated vgg(name; pretrained)

            GC.gc(true)
        end
    end
end

@testitem "VisionTransformer" setup=[SharedTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES, name in [:tiny, :small, :base]
        # :large, :huge, :giant, :gigantic --> too large for CI
        model, ps, st = Vision.VisionTransformer(name; pretrained=false)
        ps = ps |> dev
        st = Lux.testmode(st) |> dev
        img = randn(Float32, 256, 256, 3, 2) |> aType

        @jet model(img, ps, st)
        @test size(first(model(img, ps, st))) == (1000, 2)

        # @test_deprecated doesnt work since other @warn s are present
        model, ps, st = vision_transformer(name; pretrained=false)
        ps = ps |> dev
        st = Lux.testmode(st) |> dev
        img = randn(Float32, 256, 256, 3, 2) |> aType

        @jet model(img, ps, st)
        @test size(first(model(img, ps, st))) == (1000, 2)

        GC.gc(true)
    end
end
