```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: Boltz.jl ⚡ Docs
  text: Pre-built Deep Learning Models in Julia
  tagline: Accelerate ⚡ your ML research using pre-built Deep Learning Models with Lux
  actions:
    - theme: brand
      text: Lux.jl Docs
      link: https://lux.csail.mit.edu/
    - theme: alt
      text: Tutorials 📚
      link: /tutorials/getting_started
    - theme: alt
      text: Vision Models 👀
      link: /api/vision
    - theme: alt
      text: Layers API 🧩
      link: /api/layers
    - theme: alt
      text: View on GitHub
      link: https://github.com/LuxDL/Boltz.jl
  image:
    src: /lux-logo.svg
    alt: Lux.jl

features:
  - icon: 🔥
    title: Powered by Lux.jl
    details: Boltz.jl is built on top of Lux.jl, a pure Julia Deep Learning Framework designed for Scientific Machine Learning.
    link: https://lux.csail.mit.edu/

  - icon: 🧩
    title: Pre-built Models
    details: Boltz.jl provides pre-built models for common deep learning tasks, such as image classification.
    link: /api/vision

  - icon: 🧑‍🔬
    title: SciML Primitives
    details: Common deep learning primitives needed for scientific machine learning.
    link: https://sciml.ai/
---
```

## How to Install Boltz.jl?

Its easy to install Boltz.jl. Since Boltz.jl is registered in the Julia General registry,
you can simply run the following command in the Julia REPL:

```julia
julia> using Pkg
julia> Pkg.add("Boltz")
```

If you want to use the latest unreleased version of Boltz.jl, you can run the following
command: (in most cases the released version will be same as the version on github)

```julia
julia> using Pkg
julia> Pkg.add(url="https://github.com/LuxDL/Boltz.jl")
```

## Want GPU Support?

Install the following package(s):

:::code-group

```julia [NVIDIA GPUs]
using Pkg
Pkg.add("LuxCUDA")
# or
Pkg.add(["CUDA", "cuDNN"])
```

```julia [AMD ROCm GPUs]
using Pkg
Pkg.add("AMDGPU")
```

```julia [Metal M-Series GPUs]
using Pkg
Pkg.add("Metal")
```

```julia [Intel GPUs]
using Pkg
Pkg.add("oneAPI")
```

:::
