module BoltzMetalheadExt

using ArgCheck: @argcheck
using Metalhead: Metalhead

using Lux: Lux, FromFluxAdaptor

using Boltz: Boltz, Utils, Vision

Utils.is_extension_loaded(::Val{:Metalhead}) = true

function Vision.ResNetMetalhead(depth::Int; pretrained::Bool=false)
    @argcheck depth in (18, 34, 50, 101, 152)
    return FromFluxAdaptor(; preserve_ps_st=pretrained)(Metalhead.ResNet(
        depth; pretrain=pretrained).layers)
end

function Vision.ResNeXtMetalhead(
        depth::Int; cardinality=32, base_width=nothing, pretrained::Bool=false)
    @argcheck depth in (50, 101, 152)
    base_width = base_width === nothing ? (depth == 101 ? 8 : 4) : base_width
    return FromFluxAdaptor(; preserve_ps_st=pretrained)(Metalhead.ResNeXt(
        depth; pretrain=pretrained, cardinality, base_width).layers)
end

function Vision.GoogLeNetMetalhead(; pretrained::Bool=false)
    return FromFluxAdaptor(; preserve_ps_st=pretrained)(Metalhead.GoogLeNet(;
        pretrain=pretrained).layers)
end

function Vision.DenseNetMetalhead(depth::Int; pretrained::Bool=false)
    @argcheck depth in (121, 161, 169, 201)
    return FromFluxAdaptor(; preserve_ps_st=pretrained)(Metalhead.DenseNet(
        depth; pretrain=pretrained).layers)
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
    return model
end

function Vision.ConvMixerMetalhead(name::Symbol; pretrained::Bool=false)
    @argcheck name in (:base, :large, :small)
    return FromFluxAdaptor(; preserve_ps_st=pretrained)(Metalhead.ConvMixer(
        name; pretrain=pretrained).layers)
end

function Vision.SqueezeNetMetalhead(; pretrained::Bool=false)
    return FromFluxAdaptor(; preserve_ps_st=pretrained)(Metalhead.SqueezeNet(;
        pretrain=pretrained).layers)
end

function Vision.WideResNetMetalhead(depth::Int; pretrained::Bool=false)
    @argcheck depth in (18, 34, 50, 101, 152)
    return FromFluxAdaptor(; preserve_ps_st=pretrained)(Metalhead.WideResNet(
        depth; pretrain=pretrained).layers)
end

end
