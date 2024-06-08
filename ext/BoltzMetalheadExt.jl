module BoltzMetalheadExt

using ArgCheck: @argcheck
using Boltz: Boltz, __maybe_initialize_model, Vision
using Lux: Lux, FromFluxAdaptor
using Metalhead: Metalhead

@inline Boltz._is_extension_loaded(::Val{:Metalhead}) = true

function Vision.__AlexNet(; pretrained=false, kwargs...)
    model = FromFluxAdaptor()(Metalhead.AlexNet().layers)
    pretrained && (model = Lux.Chain(model[1], model[2])) # Compatibility with pretrained weights
    return __maybe_initialize_model(:alexnet, model; pretrained, kwargs...)
end

function Vision.__ResNet(depth::Int; kwargs...)
    @argcheck depth in (18, 34, 50, 101, 152)
    model = FromFluxAdaptor()(Metalhead.ResNet(depth).layers)
    return __maybe_initialize_model(Symbol(:resnet, depth), model; kwargs...)
end

function Vision.__ResNeXt(depth::Int; kwargs...)
    @argcheck depth in (50, 101, 152)
    model = FromFluxAdaptor()(Metalhead.ResNeXt(depth).layers)
    return __maybe_initialize_model(Symbol(:resnext, depth), model; kwargs...)
end

function Vision.__GoogLeNet(; kwargs...)
    model = FromFluxAdaptor()(Metalhead.GoogLeNet().layers)
    return __maybe_initialize_model(:googlenet, model; kwargs...)
end

function Vision.__DenseNet(depth::Int; kwargs...)
    @argcheck depth in (121, 161, 169, 201)
    model = FromFluxAdaptor()(Metalhead.DenseNet(depth).layers)
    return __maybe_initialize_model(Symbol(:densenet, depth), model; kwargs...)
end

function Vision.__MobileNet(name::Symbol; kwargs...)
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
    return __maybe_initialize_model(Symbol(:mobilenet, "_", name), model; kwargs...)
end

function Vision.__ConvMixer(name::Symbol; kwargs...)
    @argcheck name in (:base, :large, :small)
    model = FromFluxAdaptor()(Metalhead.ConvMixer(name).layers)
    return __maybe_initialize_model(Symbol(:convmixer, "_", name), model; kwargs...)
end

end
