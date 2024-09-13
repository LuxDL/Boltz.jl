module BoltzMetalheadExt

using ArgCheck: @argcheck
using Metalhead: Metalhead

using Lux: Lux, FromFluxAdaptor

using Boltz: Boltz, Utils, Vision

Utils.is_extension_loaded(::Val{:Metalhead}) = true

function Vision.ResNetMetalhead(depth::Int; pretrained::Bool=false)
    @argcheck depth in (18, 34, 50, 101, 152)
    return FromFluxAdaptor(; preserve_ps_st=pretrained, force_preserve=true)(Metalhead.ResNet(
        depth; pretrain=pretrained).layers)
end

function Vision.ResNeXtMetalhead(depth::Int; pretrained::Bool=false)
    @argcheck depth in (50, 101, 152)
    return FromFluxAdaptor(; preserve_ps_st=pretrained, force_preserve=true)(Metalhead.ResNeXt(
        depth; pretrain=pretrained).layers)
end

function Vision.GoogLeNetMetalhead(; pretrained::Bool=false)
    return FromFluxAdaptor(; preserve_ps_st=pretrained, force_preserve=true)(Metalhead.GoogLeNet(;
        pretrain=pretrained).layers)
end

function Vision.DenseNetMetalhead(depth::Int; pretrained::Bool=false)
    @argcheck depth in (121, 161, 169, 201)
    return FromFluxAdaptor(; preserve_ps_st=pretrained, force_preserve=true)(Metalhead.DenseNet(
        depth; pretrain=pretrained).layers)
end

function Vision.MobileNetMetalhead(name::Symbol; pretrained::Bool=false)
    @argcheck name in (:v1, :v2, :v3_small, :v3_large)
    adaptor = FromFluxAdaptor(; preserve_ps_st=pretrained, force_preserve=true)
    model = if name == :v1
        adaptor(Metalhead.MobileNetv1(; pretrain=pretrained).layers)
    elseif name == :v2
        adaptor(Metalhead.MobileNetv2(; pretrain=pretrained).layers)
    elseif name == :v3_small
        adaptor(Metalhead.MobileNetv3(:small; pretrain=pretrained).layers)
    elseif name == :v3_large
        adaptor(Metalhead.MobileNetv3(:large; pretrain=pretrained).layers)
    end
    return model
end

function Vision.ConvMixerMetalhead(name::Symbol; pretrained::Bool=false)
    @argcheck name in (:base, :large, :small)
    return FromFluxAdaptor(; preserve_ps_st=pretrained, force_preserve=true)(Metalhead.ConvMixer(
        name; pretrain=pretrained).layers)
end

function Vision.SqueezeNetMetalhead(; pretrained::Bool=false)
    return FromFluxAdaptor(; preserve_ps_st=pretrained, force_preserve=true)(Metalhead.SqueezeNet(;
        pretrain=pretrained).layers)
end

function Vision.WideResNetMetalhead(depth::Int; pretrained::Bool=false)
    @argcheck depth in (18, 34, 50, 101, 152)
    return FromFluxAdaptor(; preserve_ps_st=pretrained, force_preserve=true)(Metalhead.WideResNet(
        depth; pretrain=pretrained).layers)
end

end
