module BoltzMetalheadExt

using Boltz, Lux, Metalhead
import Boltz: alexnet, convmixer, densenet, googlenet, mobilenet, resnet, resnext
using Boltz: _initialize_model, assert_name_present_in

function alexnet(name::Symbol; pretrained=false, kwargs...)
    assert_name_present_in(name, (:alexnet,))
    model = FromFluxAdaptor()(AlexNet().layers)

    # Compatibility with pretrained weights
    pretrained && (model = Chain(model[1], model[2]))

    return _initialize_model(name, model; pretrained, kwargs...)
end

function resnet(name::Symbol; pretrained=false, kwargs...)
    assert_name_present_in(name, (:resnet18, :resnet34, :resnet50, :resnet101, :resnet152))
    model = if name == :resnet18
        FromFluxAdaptor()(ResNet(18).layers)
    elseif name == :resnet34
        FromFluxAdaptor()(ResNet(34).layers)
    elseif name == :resnet50
        FromFluxAdaptor()(ResNet(50).layers)
    elseif name == :resnet101
        FromFluxAdaptor()(ResNet(101).layers)
    elseif name == :resnet152
        FromFluxAdaptor()(ResNet(152).layers)
    end

    return _initialize_model(name, model; pretrained, kwargs...)
end

function resnext(name::Symbol; kwargs...)
    assert_name_present_in(name, (:resnext50, :resnext101, :resnext152))
    model = if name == :resnext50
        FromFluxAdaptor()(ResNeXt(50).layers)
    elseif name == :resnext101
        FromFluxAdaptor()(ResNeXt(101).layers)
    elseif name == :resnext152
        FromFluxAdaptor()(ResNeXt(152).layers)
    end
    return _initialize_model(name, model; kwargs...)
end

function googlenet(name::Symbol; kwargs...)
    assert_name_present_in(name, (:googlenet,))
    model = FromFluxAdaptor()(GoogLeNet().layers)
    return _initialize_model(name, model; kwargs...)
end

function densenet(name::Symbol; kwargs...)
    assert_name_present_in(name, (:densenet121, :densenet161, :densenet169, :densenet201))
    model = if name == :densenet121
        FromFluxAdaptor()(DenseNet(121).layers)
    elseif name == :densenet161
        FromFluxAdaptor()(DenseNet(161).layers)
    elseif name == :densenet169
        FromFluxAdaptor()(DenseNet(169).layers)
    elseif name == :densenet201
        FromFluxAdaptor()(DenseNet(201).layers)
    end
    return _initialize_model(name, model; kwargs...)
end

function mobilenet(name::Symbol; kwargs...)
    assert_name_present_in(name,
        (:mobilenet_v1, :mobilenet_v2, :mobilenet_v3_small, :mobilenet_v3_large))
    model = if name == :mobilenet_v1
        FromFluxAdaptor()(MobileNetv1().layers)
    elseif name == :mobilenet_v2
        FromFluxAdaptor()(MobileNetv2().layers)
    elseif name == :mobilenet_v3_small
        FromFluxAdaptor()(MobileNetv3(:small).layers)
    elseif name == :mobilenet_v3_large
        FromFluxAdaptor()(MobileNetv3(:large).layers)
    end
    return _initialize_model(name, model; kwargs...)
end

function convmixer(name::Symbol; kwargs...)
    assert_name_present_in(name, (:base, :large, :small))
    model = FromFluxAdaptor()(ConvMixer(name).layers)
    return _initialize_model(name, model; kwargs...)
end

end
