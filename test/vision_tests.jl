@testitem "AlexNet" setup=[SharedTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES
        model, ps, st = Layers.AlexNet(; pretrained=false)
        ps = ps |> dev
        st = Lux.testmode(st) |> dev
        img = randn(Float32, 224, 224, 3, 2) |> aType

        @jet model(img, ps, st)
        @test size(first(model(img, ps, st))) == (1000, 2)

        GC.gc(true)
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

        GC.gc(true)
    end
end

@testitem "VGG" setup=[SharedTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES,
        name in [:vgg11, :vgg11_bn, :vgg13, :vgg13_bn, :vgg16, :vgg16_bn, :vgg19, :vgg19_bn],
        pretrained in [false, true]

        model, ps, st = Vision.VGG(name; pretrained)
        ps = ps |> dev
        st = Lux.testmode(st) |> dev
        img = randn(Float32, 224, 224, 3, 2) |> aType

        @jet model(img, ps, st)
        @test size(first(model(img, ps, st))) == (1000, 2)

        GC.gc(true)
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

        GC.gc(true)
    end
end
