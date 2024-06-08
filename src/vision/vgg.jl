function __vgg_convolutional_layers(config, batchnorm, inchannels)
    layers = Vector{Any}(undef, length(config) * 2)
    input_filters = inchannels
    for (i, (chs, depth)) in enumerate(config)
        layers[2i - 1] = Layers.ConvBatchNormActivation(
            (3, 3), input_filters => chs, depth, relu;
            conv_kwargs=(; pad=(1, 1)), use_norm=batchnorm)
        layers[2i] = Lux.MaxPool((2, 2))
        input_filters = chs
    end
    return Lux.Chain(layers...)
end

function __vgg_classifier_layers(imsize, nclasses, fcsize, dropout)
    return Lux.Chain(Lux.FlattenLayer(), Lux.Dense(Int(prod(imsize)) => fcsize, relu),
        Lux.Dropout(dropout), Lux.Dense(fcsize => fcsize, relu),
        Lux.Dropout(dropout), Lux.Dense(fcsize => nclasses))
end

"""
    VGG(imsize; config, inchannels, batchnorm = false, nclasses, fcsize, dropout)

Create a VGG model [1].

## Arguments

  - `imsize`: input image width and height as a tuple
  - `config`: the configuration for the convolution layers
  - `inchannels`: number of input channels
  - `batchnorm`: set to `true` to use batch normalization after each convolution
  - `nclasses`: number of output classes
  - `fcsize`: intermediate fully connected layer size
  - `dropout`: dropout level between fully connected layers

## References

[1] Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale
    image recognition." arXiv preprint arXiv:1409.1556 (2014).
"""
function VGG(imsize; config, inchannels, batchnorm=false, nclasses, fcsize, dropout)
    feature_extrator = __vgg_convolutional_layers(config, batchnorm, inchannels)

    img = ones(Float32, (imsize..., inchannels, 2))
    rng = Xoshiro(0)
    _ps, _st = LuxCore.setup(rng, feature_extrator)
    outsize = size(first(feature_extrator(img, _ps, _st)))

    classifier = __vgg_classifier_layers(outsize[1:((end - 1))], nclasses, fcsize, dropout)

    return Lux.Chain(feature_extrator, classifier)
end

#! format: off
const VGG_CONFIG = Dict(
    11 => [(64, 1), (128, 1), (256, 2), (512, 2), (512, 2)],
    13 => [(64, 2), (128, 2), (256, 2), (512, 2), (512, 2)],
    16 => [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)],
    19 => [(64, 2), (128, 2), (256, 4), (512, 4), (512, 4)]
)
#! format: on

"""
    VGG(depth::Int; batchnorm=false, kwargs...)

Create a VGG model [1] with ImageNet Configuration.

## Arguments

  * `depth::Int`: the depth of the VGG model. Choices: {`11`, `13`, `16`, `19`}.

## Keyword Arguments

  * `batchnorm = false`: set to `true` to use batch normalization after each convolution.
$(INITIALIZE_KWARGS)

## References

[1] Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale
    image recognition." arXiv preprint arXiv:1409.1556 (2014).
"""
function VGG(depth::Int; batchnorm::Bool=false, kwargs...)
    name = Symbol(:vgg, depth, ifelse(batchnorm, "_bn", ""))
    config, inchannels, nclasses, fcsize = VGG_CONFIG[depth], 3, 1000, 4096
    model = VGG((224, 224); config, inchannels, batchnorm, nclasses, fcsize, dropout=0.5f0)
    return __maybe_initialize_model(name, model; kwargs...)
end
