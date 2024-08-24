using Documenter, DocumenterVitepress, Boltz

#! format: off
pages = [
    "Boltz.jl" => "index.md",
    "Tutorials" => [
        "Getting Started" => "tutorials/getting_started.md",
    ],
    "API Reference" => [
        "Index" => "api/index.md",
        "Basis Functions" => "api/basis.md",
        "Layers" => "api/layers.md",
        "Vision Models" => "api/vision.md",
        "Private API" => "api/private.md",
    ]
]
#! format: on

deploy_config = Documenter.auto_detect_deploy_system()
deploy_decision = Documenter.deploy_folder(deploy_config; repo="github.com/LuxDL/Boltz.jl",
    devbranch="main", devurl="dev", push_preview=true)

makedocs(; sitename="Boltz.jl Docs",
    authors="Avik Pal et al.",
    clean=true,
    doctest=false,  # We test it in the CI, no need to run it here
    modules=[Boltz],
    linkcheck=true,
    repo="https://github.com/LuxDL/Boltz.jl/blob/{commit}{path}#{line}",
    format=DocumenterVitepress.MarkdownVitepress(;
        repo="github.com/LuxDL/Boltz.jl", devbranch="main", devurl="dev", deploy_decision),
    draft=false,
    pages)

deploydocs(; repo="github.com/LuxDL/Boltz.jl.git",
    push_preview=true, target="build", devbranch="main")
