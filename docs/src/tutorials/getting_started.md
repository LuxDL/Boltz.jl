# Getting Started

!!! tip "Prerequisites"

    Here we assume that you are familiar with [`Lux.jl`](https://lux.csail.mit.edu/stable/).
    If not please take a look at the
    [Lux.jl tutoials](https://lux.csail.mit.edu/stable/tutorials/).

`Boltz.jl` is just like `Lux.jl` but comes with more "batteries included". Let's start by
defining an MLP model.

```@example getting_started
using Lux, Boltz, Random
```

## Multi-Layer Perceptron

If we were to do this in `Lux.jl` we would write the following:

```@example getting_started
model = Chain(
    Dense(784, 256, relu),
    Dense(256, 10)
)
```

But in `Boltz.jl` we can do this:

```@example getting_started
model = Layers.MLP(784, (256, 10), relu)
```

The `MLP` function is just a convenience wrapper around `Lux.Chain` that constructs a
multi-layer perceptron with the given number of layers and activation function.

## How about VGG?

!!! warning "Returned Values"

    The returned value from `Vision` module functions are a 3 tuple of (model, ps, st).
    The `ps` and `st` are the parameters and states of the model respectively.

Let's take a look at the `Vision` module. We can construct a VGG model with the
following code:

```@example getting_started
model, _, _ = Vision.VGG(13)
model
```

We can also load pretrained ImageNet weights using

```@example getting_started
model, _, _ = Vision.VGG(13; pretrained=true)
model
```
