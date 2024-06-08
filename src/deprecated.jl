Base.@deprecate alexnet(args...; kwargs...) Vision.AlexNet(; kwargs...)
Base.@deprecate convmixer(args...; kwargs...) Vision.ConvMixer(args...; kwargs...)
Base.@deprecate googlenet(args...; kwargs...) Vision.GoogLeNet(; kwargs...)

Base.@deprecate_binding vision_transformer Vision.VisionTransformer true
Base.@deprecate_binding transformer_encoder Layers.VisionTransformerEncoder

function vgg(name::Symbol; kwargs...)
    Base.depwarn("`vgg(::Symbol)` is deprecated. See `Vision.VGG` for more details.", :vgg)
    @argcheck name in (
        :vgg11, :vgg11_bn, :vgg13, :vgg13_bn, :vgg16, :vgg16_bn, :vgg19, :vgg19_bn)
    name == :vgg11 && return Vision.VGG(11; kwargs...)
    name == :vgg11_bn && return Vision.VGG(11; batchnorm=true, kwargs...)
    name == :vgg13 && return Vision.VGG(13; kwargs...)
    name == :vgg13_bn && return Vision.VGG(13; batchnorm=true, kwargs...)
    name == :vgg16 && return Vision.VGG(16; kwargs...)
    name == :vgg16_bn && return Vision.VGG(16; batchnorm=true, kwargs...)
    name == :vgg19 && return Vision.VGG(19; kwargs...)
    name == :vgg19_bn && return Vision.VGG(19; batchnorm=true, kwargs...)
end

function resnet(name::Symbol; kwargs...)
    depth = parse(Int, String(name)[7:end])
    Base.depwarn(
        "`resnet(::Symbol)` is deprecated. See `Vision.ResNet` for more details.", :resnet)
    return Vision.ResNet(depth; kwargs...)
end

function densenet(name::Symbol; kwargs...)
    depth = parse(Int, String(name)[9:end])
    Base.depwarn(
        "`densenet(::Symbol)` is deprecated. See `Vision.DenseNet` for more details.",
        :densenet)
    return Vision.DenseNet(depth; kwargs...)
end

function resnext(name::Symbol; kwargs...)
    depth = parse(Int, String(name)[8:end])
    Base.depwarn(
        "`resnext(::Symbol)` is deprecated. See `Vision.ResNeXt` for more details.",
        :resnext)
    return Vision.ResNeXt(depth; kwargs...)
end

function mobilenet(name::Symbol; kwargs...)
    name = Symbol(String(name)[11:end])
    Base.depwarn(
        "`mobilenet(::Symbol)` is deprecated. See `Vision.MobileNet` for more details.",
        :mobilenet)
    return Vision.MobileNet(name; kwargs...)
end

export vgg, resnet, densenet, resnext, mobilenet
