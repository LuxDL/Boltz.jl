module BoltzMetalheadExt

using ArgCheck: @argcheck
using Metalhead: Metalhead

using Lux: Lux, FromFluxAdaptor

using Boltz: Boltz, Utils, Vision

Utils.is_extension_loaded(::Val{:Metalhead}) = true

function Vision.ResNetMetalhead(depth::Int; pretrained::Bool=false)
    @argcheck depth in (18, 34, 50, 101, 152)
    model = FromFluxAdaptor(; preserve_ps_st=pretrained)(Metalhead.ResNet(
        depth; pretrain=pretrained).layers)
    return Symbol(:resnet, depth), model
end

function Vision.ResNeXtMetalhead(depth::Int; pretrained::Bool=false)
    @argcheck depth in (50, 101, 152)
    model = FromFluxAdaptor(; preserve_ps_st=pretrained)(Metalhead.ResNeXt(
        depth; pretrain=pretrained).layers)
    return Symbol(:resnext, depth), model
end

function Vision.GoogLeNetMetalhead(; pretrained::Bool=false)
    model = FromFluxAdaptor(; preserve_ps_st=pretrained)(Metalhead.GoogLeNet(
        ; pretrain=pretrained).layers)
    return :googlenet, model
end

function Vision.DenseNetMetalhead(depth::Int; pretrained::Bool=false)
    @argcheck depth in (121, 161, 169, 201)
    model = FromFluxAdaptor(; preserve_ps_st=pretrained)(Metalhead.DenseNet(
        depth; pretrain=pretrained).layers)
    return Symbol(:densenet, depth), model
end

function Vision.MobileNetMetalhead(name::Symbol; pretrained::Bool=false)
    @argcheck name in (:v1, :v2, :v3_small, :v3_large)
    adaptor = FromFluxAdaptor(; preserve_ps_st=pretrained)
    model = if name == :v1
        adaptor(Metalhead.MobileNetv1(; pretrain=pretrained).layers)
    elseif name == :v2
        adaptor(Metalhead.MobileNetv2(; pretrain=pretrained).layers)
    elseif name == :v3_small
        adaptor(Metalhead.MobileNetv3(:small; pretrain=pretrained).layers)
    elseif name == :v3_large
        adaptor(Metalhead.MobileNetv3(:large; pretrain=pretrained).layers)
    end
    return Symbol(:mobilenet_, name), model
end

function Vision.ConvMixerMetalhead(name::Symbol; pretrained::Bool=false)
    @argcheck name in (:base, :large, :small)
    model = FromFluxAdaptor(; preserve_ps_st=pretrained)(Metalhead.ConvMixer(
        name; pretrain=pretrained).layers)
    return Symbol(:convmixer_, name), model
end

end
