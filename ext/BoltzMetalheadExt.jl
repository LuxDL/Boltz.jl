module BoltzMetalheadExt

using ArgCheck: @argcheck
using Metalhead: Metalhead

using Lux: Lux, FromFluxAdaptor

using Boltz: Boltz, Utils, Vision
using Boltz.InitializeModels: maybe_initialize_model

Utils.is_extension_loaded(::Val{:Metalhead}) = true

function Vision.AlexNetMetalhead(; pretrained=false, kwargs...)
    model = FromFluxAdaptor()(Metalhead.AlexNet().layers)
    pretrained && (model = Lux.Chain(model[1], model[2])) # Compatibility with pretrained weights
    return maybe_initialize_model(:alexnet, model; pretrained, kwargs...)
end

function Vision.ResNetMetalhead(depth::Int; kwargs...)
    @argcheck depth in (18, 34, 50, 101, 152)
    model = FromFluxAdaptor()(Metalhead.ResNet(depth).layers)
    return maybe_initialize_model(Symbol(:resnet, depth), model; kwargs...)
end

function Vision.ResNeXtMetalhead(depth::Int; kwargs...)
    @argcheck depth in (50, 101, 152)
    model = FromFluxAdaptor()(Metalhead.ResNeXt(depth).layers)
    return maybe_initialize_model(Symbol(:resnext, depth), model; kwargs...)
end

function Vision.GoogLeNetMetalhead(; kwargs...)
    model = FromFluxAdaptor()(Metalhead.GoogLeNet().layers)
    return maybe_initialize_model(:googlenet, model; kwargs...)
end

function Vision.DenseNetMetalhead(depth::Int; kwargs...)
    @argcheck depth in (121, 161, 169, 201)
    model = FromFluxAdaptor()(Metalhead.DenseNet(depth).layers)
    return maybe_initialize_model(Symbol(:densenet, depth), model; kwargs...)
end

function Vision.MobileNetMetalhead(name::Symbol; kwargs...)
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
    return maybe_initialize_model(Symbol(:mobilenet, "_", name), model; kwargs...)
end

function Vision.ConvMixerMetalhead(name::Symbol; kwargs...)
    @argcheck name in (:base, :large, :small)
    model = FromFluxAdaptor()(Metalhead.ConvMixer(name).layers)
    return maybe_initialize_model(Symbol(:convmixer, "_", name), model; kwargs...)
end

end
