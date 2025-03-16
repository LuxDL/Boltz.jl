# # Getting Started
#
# !!! tip "Prerequisites"
#
#     Here we assume that you are familiar with [`Lux.jl`](https://lux.csail.mit.edu/stable/).
#     If not please take a look at the
#     [Lux.jl tutoials](https://lux.csail.mit.edu/stable/tutorials/).
#
# `Boltz.jl` is just like `Lux.jl` but comes with more "batteries included". Let's start by
# defining an MLP model.

using Lux, Boltz, Random

# ## Multi-Layer Perceptron
#
# If we were to do this in `Lux.jl` we would write the following:

model = Chain(Dense(784, 256, relu), Dense(256, 10))

# But in `Boltz.jl` we can do this:

model = Layers.MLP(784, (256, 10), relu)

# The `MLP` function is just a convenience wrapper around `Lux.Chain` that constructs a
# multi-layer perceptron with the given number of layers and activation function.

# ## How about VGG?
#
# Let's take a look at the `Vision` module. We can construct a VGG model with the
# following code:

Vision.VGG(13)

# We can also load pretrained ImageNet weights using

# !!! note "Load JLD2"
#
#     You need to load `JLD2` before being able to load pretrained weights.

using JLD2

Vision.VGG(13; pretrained=true)

# ## Loading Models from Metalhead (Flux.jl)

# We can load models from Metalhead (Flux.jl), just remember to load `Metalhead` before.

using Metalhead

Vision.ResNet(18)
