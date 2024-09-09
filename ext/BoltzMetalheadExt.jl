module BoltzMetalheadExt

using ArgCheck: @argcheck
using Metalhead: Metalhead

using Lux: Lux, FromFluxAdaptor

using Boltz: Boltz, Utils, Vision

Utils.is_extension_loaded(::Val{:Metalhead}) = true

function Vision.AlexNetMetalhead()
    model = FromFluxAdaptor()(Metalhead.AlexNet().layers)
    return :alexnet, model
end

function Vision.ResNetMetalhead(depth::Int)
    @argcheck depth in (18, 34, 50, 101, 152)
    model = FromFluxAdaptor()(Metalhead.ResNet(depth).layers)
    return Symbol(:resnet, depth), model
end

function Vision.ResNeXtMetalhead(depth::Int)
    @argcheck depth in (50, 101, 152)
    model = FromFluxAdaptor()(Metalhead.ResNeXt(depth).layers)
    return Symbol(:resnext, depth), model
end

function Vision.GoogLeNetMetalhead()
    model = FromFluxAdaptor()(Metalhead.GoogLeNet().layers)
    return :googlenet, model
end

function Vision.DenseNetMetalhead(depth::Int)
    @argcheck depth in (121, 161, 169, 201)
    model = FromFluxAdaptor()(Metalhead.DenseNet(depth).layers)
    return Symbol(:densenet, depth), model
end

function Vision.MobileNetMetalhead(name::Symbol)
    @argcheck name in (:v1, :v2, :v3_small, :v3_large)
    model = if name == :v1
        FromFluxAdaptor()(Metalhead.MobileNetv1().layers)
    elseif name == :v2
        FromFluxAdaptor()(Metalhead.MobileNetv2().layers)
    elseif name == :v3_small
        FromFluxAdaptor()(Metalhead.MobileNetv3(:small).layers)
    elseif name == :v3_large
        FromFluxAdaptor()(Metalhead.MobileNetv3(:large).layers)
    end
    return Symbol(:mobilenet_, name), model
end

function Vision.ConvMixerMetalhead(name::Symbol)
    @argcheck name in (:base, :large, :small)
    model = FromFluxAdaptor()(Metalhead.ConvMixer(name).layers)
    return Symbol(:convmixer_, name), model
end

end
