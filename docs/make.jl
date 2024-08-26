using Documenter, DocumenterVitepress, DocumenterCitations, Boltz

#! format: off
pages = [
    "Boltz.jl" => "index.md",
    "Tutorials" => [
        "Getting Started" => "tutorials/getting_started.md",
    ],
    "API Reference" => [
        "Index" => "api/index.md",
        "Basis Functions" => "api/basis.md",
        "Layers API" => "api/layers.md",
        "Vision Models" => "api/vision.md",
        "Private API" => "api/private.md",
    ]
]
#! format: on

bib = CitationBibliography(
    joinpath(@__DIR__, "ref.bib");
    style=:authoryear
)

doctestexpr = quote
    using Random, Lux
end

DocMeta.setdocmeta!(Boltz, :DocTestSetup, doctestexpr; recursive=true)

deploy_config = Documenter.auto_detect_deploy_system()
deploy_decision = Documenter.deploy_folder(deploy_config; repo="github.com/LuxDL/Boltz.jl",
    devbranch="main", devurl="dev", push_preview=true)

makedocs(; sitename="Boltz.jl Docs",
    authors="Avik Pal et al.",
    clean=true,
    modules=[Boltz],
    linkcheck=true,
    repo="https://github.com/LuxDL/Boltz.jl/blob/{commit}{path}#{line}",
    format=DocumenterVitepress.MarkdownVitepress(;
        repo="github.com/LuxDL/Boltz.jl", devbranch="main", devurl="dev", deploy_decision),
    draft=false,
    plugins=[bib],
    pages)

deploydocs(; repo="github.com/LuxDL/Boltz.jl.git",
    push_preview=true, target="build", devbranch="main")
