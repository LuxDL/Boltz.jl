# Boltz ⚡

[![GitHub Discussions](https://img.shields.io/github/discussions/LuxDL/Lux.jl?color=white&logo=github&label=Discussions)](https://github.com/LuxDL/Lux.jl/discussions)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://luxdl.github.io/Boltz.jl/dev/)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://luxdl.github.io/Boltz.jl/stable/)

[![CI](https://github.com/LuxDL/Boltz.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/LuxDL/Boltz.jl/actions/workflows/CI.yml)
[![Build status](https://badge.buildkite.com/33d66eb556ba88f60e733e97ff65c133fdb5f0ac683e823cfb.svg?branch=main)](https://buildkite.com/julialang/boltz-dot-jl)
[![codecov](https://codecov.io/gh/LuxDL/Boltz.jl/branch/main/graph/badge.svg?token=YBImUxz5qO)](https://codecov.io/gh/LuxDL/Boltz.jl)

[![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FBoltz&query=total_requests&suffix=%2Fmonth&label=Downloads)](https://juliapkgstats.com/pkg/Boltz)
[![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FBoltz&query=total_requests&&label=Total%20Downloads)](https://juliapkgstats.com/pkg/Boltz)

[![JET Testing](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)

Accelerate ⚡ your ML research using pre-built Deep Learning Models with Lux.

## Installation

```julia
using Pkg
Pkg.add("Boltz")
```

## Getting Started

```julia
using Boltz, Lux, Random
using JLD2 # Needed to load pretrained weights

model = Vision.AlexNet(; pretrained="ImageNet1K") # or "DEFAULT"
ps, st = Lux.setup(Random.default_rng(), model)

x = rand(Float32, 224, 224, 3, 1)
model(x, ps, Lux.testmode(st))
```
