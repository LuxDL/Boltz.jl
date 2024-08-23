"""
    AlexNet(; kwargs...)

Create an AlexNet model [1]

## Keyword Arguments

$(INITIALIZE_KWARGS)

## References

[1] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with
deep convolutional neural networks." Advances in neural information processing systems 25
(2012): 1097-1105.
"""
function AlexNet end

"""
    ResNet(depth::Int; kwargs...)

Create a ResNet model [1].

## Arguments

  * `depth::Int`: The depth of the ResNet model. Must be one of 18, 34, 50, 101, or 152.

## Keyword Arguments

$(INITIALIZE_KWARGS)

## References

[1] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the
    IEEE conference on computer vision and pattern recognition. 2016.
"""
function ResNet end

"""
    ResNeXt(depth::Int; kwargs...)

Create a ResNeXt model [1].

## Arguments

  * `depth::Int`: The depth of the ResNeXt model. Must be one of 50, 101, or 152.

## Keyword Arguments

$(INITIALIZE_KWARGS)

## References

[1] Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He, Ross Gorshick, and
    Piotr Dollár. "Aggregated residual transformations for deep neural networks."
    Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
"""
function ResNeXt end

"""
    GoogLeNet(; kwargs...)

Create a GoogLeNet model [1].

## Keyword Arguments

$(INITIALIZE_KWARGS)

## References

[1] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov,
    Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. "Going deeper with
    convolutions." Proceedings of the IEEE conference on computer vision and pattern
    recognition. 2015.
"""
function GoogLeNet end

"""
    DenseNet(depth::Int; kwargs...)

Create a DenseNet model [1].

## Arguments

  * `depth::Int`: The depth of the DenseNet model. Must be one of 121, 161, 169, or 201.

## Keyword Arguments

$(INITIALIZE_KWARGS)

## References

[1] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger. "Densely connected
    convolutional networks." Proceedings of the IEEE conference on computer vision and
    pattern recognition. 2016.
"""
function DenseNet end

"""
    MobileNet(name::Symbol; kwargs...)

Create a MobileNet model [1, 2, 3].

## Arguments

  * `name::Symbol`: The name of the MobileNet model. Must be one of `:v1`, `:v2`,
    `:v3_small`, or `:v3_large`.

## Keyword Arguments

$(INITIALIZE_KWARGS)

## References

[1] Howard, Andrew G., et al. "Mobilenets: Efficient convolutional neural networks for
    mobile vision applications." arXiv preprint arXiv:1704.04861 (2017).
[2] Sandler, Mark, et al. "Mobilenetv2: Inverted residuals and linear bottlenecks."
    Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
[3] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias
    Weyand, Marco Andreetto, Hartwig Adam. "Searching for MobileNetV3." arXiv preprint
    arXiv:1905.02244. 2019.
"""
function MobileNet end

"""
    ConvMixer(name::Symbol; kwargs...)

Create a ConvMixer model [1].

## Arguments

  * `name::Symbol`: The name of the ConvMixer model. Must be one of `:base`, `:small`, or
    `:large`.

## Keyword Arguments

$(INITIALIZE_KWARGS)

## References

[1] Zhu, Zhuoyuan, et al. "ConvMixer: A Convolutional Neural Network with Faster
    Depth-wise Convolutions for Computer Vision." arXiv preprint arXiv:1911.11907 (2019).
"""
function ConvMixer end

for f in [:AlexNet, :ResNet, :ResNeXt, :GoogLeNet, :DenseNet, :MobileNet, :ConvMixer]
    f_inner = Symbol("__", f)
    @eval begin
        function $(f_inner) end
        function $f(args...; kwargs...)
            if !is_extension_loaded(Val(:Metalhead))
                error("Metalhead.jl is not loaded. Please load Metalhead.jl to use this \
                       function.")
            end
            $(f_inner)(args...; kwargs...)
        end
    end
end
