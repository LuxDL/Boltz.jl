@testitem "AlexNet" setup=[SharedTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES
        @testset "pretrained: $(pretrained)" for pretrained in [true, false]
            model = Vision.AlexNet(; pretrained)
            ps, st = Lux.setup(Random.default_rng(), model) |> dev
            st = Lux.testmode(st)
            img = randn(Float32, 224, 224, 3, 2) |> aType

            @jet model(img, ps, st)
            @test size(first(model(img, ps, st))) == (1000, 2)

            GC.gc(true)
        end
    end
end

@testitem "ConvMixer" setup=[SharedTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES, name in [:small, :base, :large]
        model = Vision.ConvMixer(name; pretrained=false)
        ps, st = Lux.setup(Random.default_rng(), model) |> dev
        st = Lux.testmode(st)
        img = randn(Float32, 256, 256, 3, 2) |> aType

        @jet model(img, ps, st)
        @test size(first(model(img, ps, st))) == (1000, 2)

        GC.gc(true)
    end
end

@testitem "GoogLeNet" setup=[SharedTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES
        model = Vision.GoogLeNet(; pretrained=false)
        ps, st = Lux.setup(Random.default_rng(), model) |> dev
        st = Lux.testmode(st)
        img = randn(Float32, 224, 224, 3, 2) |> aType

        @jet model(img, ps, st)
        @test size(first(model(img, ps, st))) == (1000, 2)

        GC.gc(true)
    end
end

@testitem "MobileNet" setup=[SharedTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES, name in [:v1, :v2, :v3_small, :v3_large]
        model = Vision.MobileNet(name; pretrained=false)
        ps, st = Lux.setup(Random.default_rng(), model) |> dev
        st = Lux.testmode(st)
        img = randn(Float32, 224, 224, 3, 2) |> aType

        @jet model(img, ps, st)
        @test size(first(model(img, ps, st))) == (1000, 2)

        GC.gc(true)
    end
end

@testitem "ResNet" setup=[SharedTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES, depth in [18, 34, 50, 101, 152]
        @testset for pretrained in [false, true]
            model = Vision.ResNet(depth; pretrained)
            ps, st = Lux.setup(Random.default_rng(), model) |> dev
            st = Lux.testmode(st)
            img = randn(Float32, 224, 224, 3, 2) |> aType

            @jet model(img, ps, st)
            @test size(first(model(img, ps, st))) == (1000, 2)

            GC.gc(true)
        end
    end
end

@testitem "ResNeXt" setup=[SharedTestSetup] tags=[:vision] begin
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

                GC.gc(true)
            end
        end
    end
end

@testitem "WideResNet" setup=[SharedTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES, depth in [50, 101, 152]
        @testset for pretrained in [false, true]
            depth == 152 && pretrained && continue

            model = Vision.WideResNet(depth; pretrained)
            ps, st = Lux.setup(Random.default_rng(), model) |> dev
            st = Lux.testmode(st)
            img = randn(Float32, 224, 224, 3, 2) |> aType

            @jet model(img, ps, st)
            @test size(first(model(img, ps, st))) == (1000, 2)

            GC.gc(true)
        end
    end
end

@testitem "SqueezeNet" setup=[SharedTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES
        @testset for pretrained in [false, true]
            model = Vision.SqueezeNet(; pretrained)
            ps, st = Lux.setup(Random.default_rng(), model) |> dev
            st = Lux.testmode(st)
            img = randn(Float32, 224, 224, 3, 2) |> aType

            @jet model(img, ps, st)
            @test size(first(model(img, ps, st))) == (1000, 2)

            GC.gc(true)
        end
    end
end

@testitem "VGG" setup=[SharedTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES, depth in [11, 13, 16, 19]
        @testset for pretrained in [false, true], batchnorm in [false, true]
            model = Vision.VGG(depth; batchnorm, pretrained)
            ps, st = Lux.setup(Random.default_rng(), model) |> dev
            st = Lux.testmode(st)
            img = randn(Float32, 224, 224, 3, 2) |> aType

            @jet model(img, ps, st)
            @test size(first(model(img, ps, st))) == (1000, 2)

            GC.gc(true)
        end
    end
end

@testitem "VisionTransformer" setup=[SharedTestSetup] tags=[:vision] begin
    for (mode, aType, dev, ongpu) in MODES, name in [:tiny, :small, :base]
        # :large, :huge, :giant, :gigantic --> too large for CI
        model = Vision.VisionTransformer(name; pretrained=false)
        ps, st = Lux.setup(Random.default_rng(), model) |> dev
        st = Lux.testmode(st)
        img = randn(Float32, 256, 256, 3, 2) |> aType

        @jet model(img, ps, st)
        @test size(first(model(img, ps, st))) == (1000, 2)

        model = Vision.VisionTransformer(name; pretrained=false)
        ps, st = Lux.setup(Random.default_rng(), model) |> dev
        st = Lux.testmode(st)
        img = randn(Float32, 256, 256, 3, 2) |> aType

        @jet model(img, ps, st)
        @test size(first(model(img, ps, st))) == (1000, 2)

        GC.gc(true)
    end
end
