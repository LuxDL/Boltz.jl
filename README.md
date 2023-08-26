# Boltz ⚡

[![Join the chat at https://julialang.zulipchat.com #machine-learning](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/machine-learning)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://lux.csail.mit.edu/dev/api/)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://lux.csail.mit.edu/stable/api/)

[![CI](https://github.com/LuxDL/Boltz.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/LuxDL/Boltz.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/LuxDL/Boltz.jl/branch/main/graph/badge.svg?token=YBImUxz5qO)](https://codecov.io/gh/LuxDL/Boltz.jl)
[![Package Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/Boltz)](https://pkgs.genieframework.com?packages=Boltz)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

Accelerate ⚡ your ML research using pre-built Deep Learning Models with Lux

## Installation

```julia
using Pkg
Pkg.add("Boltz")
```

## Getting Started

```julia
using Boltz, Lux, Metalhead

model, ps, st = resnet(:alexnet; pretrained=true)
```

## Changelog


### Updating from v0.2 to v0.3

CUDA is not loaded by default. To use GPUs follow
[Lux.jl documentation](https://lux.csail.mit.edu/stable/manual/gpu_management/).

### Updating from v0.1 to v0.2

We have moved some dependencies into weak dependencies. This means that you will have to
manually load them for certain features to be available.

* To load Flux & Metalhead models, do `using Metalhead`.
