```@meta
CollapsedDocStrings = true
```

# Computer Vision Models (`Vision` API)

## Native Lux Models

```@docs
Vision.AlexNet
Vision.EfficientNet
Vision.VGG
Vision.VisionTransformer
```

## Imported from Metalhead.jl

!!! tip "Load Metalhead"

    You need to load `Metalhead` before using these models.

```@docs
Vision.ConvMixer
Vision.DenseNet
Vision.GoogLeNet
Vision.MobileNet
Vision.ResNet
Vision.ResNeXt
Vision.SqueezeNet
Vision.WideResNet
```

## Pretrained Models

!!! tip "Load Pretrained Weights"

    Pass `pretrained=true` to the model constructor to load the pretrained weights.

| MODEL                                        | Additional Packages | TOP 1 ACCURACY (%) | TOP 5 ACCURACY (%) |
| :------------------------------------------- | :-----------------: | :----------------: | :----------------: |
| `AlexNet()`                                  |       `JLD2`        |       54.48        |       77.72        |
| `VGG(11)`                                    |       `JLD2`        |       67.35        |       87.91        |
| `VGG(13)`                                    |       `JLD2`        |       68.40        |       88.48        |
| `VGG(16)`                                    |       `JLD2`        |       70.24        |       89.80        |
| `VGG(19)`                                    |       `JLD2`        |       71.09        |       90.27        |
| `VGG(11; batchnorm=true)`                    |       `JLD2`        |       69.09        |       88.94        |
| `VGG(13; batchnorm=true)`                    |       `JLD2`        |       69.66        |       89.49        |
| `VGG(16; batchnorm=true)`                    |       `JLD2`        |       72.11        |       91.02        |
| `VGG(19; batchnorm=true)`                    |       `JLD2`        |       72.95        |       91.32        |
| `EfficientNet(:b0)`                          |      `Pickle`       |         -          |         -          |
| `EfficientNet(:b1)`                          |      `Pickle`       |         -          |         -          |
| `EfficientNet(:b2)`                          |      `Pickle`       |         -          |         -          |
| `EfficientNet(:b3)`                          |      `Pickle`       |         -          |         -          |
| `EfficientNet(:b4)`                          |      `Pickle`       |         -          |         -          |
| `EfficientNet(:b5)`                          |      `Pickle`       |         -          |         -          |
| `EfficientNet(:b6)`                          |      `Pickle`       |         -          |         -          |
| `EfficientNet(:b7)`                          |      `Pickle`       |         -          |         -          |
| `ResNet(18)`                                 | `JLD2`, `Metalhead` |         -          |         -          |
| `ResNet(34)`                                 | `JLD2`, `Metalhead` |         -          |         -          |
| `ResNet(50)`                                 | `JLD2`, `Metalhead` |         -          |         -          |
| `ResNet(101)`                                | `JLD2`, `Metalhead` |         -          |         -          |
| `ResNet(152)`                                | `JLD2`, `Metalhead` |         -          |         -          |
| `ResNeXt(50; cardinality=32, base_width=4)`  | `JLD2`, `Metalhead` |         -          |         -          |
| `ResNeXt(101; cardinality=32, base_width=8)` | `JLD2`, `Metalhead` |         -          |         -          |
| `ResNeXt(101; cardinality=64, base_width=4)` | `JLD2`, `Metalhead` |         -          |         -          |
| `SqueezeNet()`                               | `JLD2`, `Metalhead` |         -          |         -          |
| `WideResNet(50)`                             | `JLD2`, `Metalhead` |         -          |         -          |
| `WideResNet(101)`                            | `JLD2`, `Metalhead` |         -          |         -          |

!!! note "Pretrained Models from Metalhead"

    For Models imported from Metalhead, the pretrained weights can be loaded if they are
    available in Metalhead. Refer to the [Metalhead.jl docs](https://fluxml.ai/Metalhead.jl/stable/#Image-Classification)
    for a list of available pretrained models.

### Preprocessing

All the pretrained models require that the images be normalized with the parameters
`mean = [0.485f0, 0.456f0, 0.406f0]` and `std = [0.229f0, 0.224f0, 0.225f0]`.
