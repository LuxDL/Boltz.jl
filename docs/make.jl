using Documenter, DocumenterVitepress, DocumenterCitations, Boltz

pages = [
    "Boltz.jl" => "index.md",
    "Tutorials" => [
        "Getting Started" => "tutorials/1_GettingStarted.md",
        "Symbolic Optimal Control" => "tutorials/2_SymbolicOptimalControl.md",
    ],
    "API Reference" => [
        "Basis Functions" => "api/basis.md",
        "Layers API" => "api/layers.md",
        "Vision Models" => "api/vision.md",
        "Physics-Informed Models" => "api/piml.md",
        "Private API" => "api/private.md",
    ],
]

doctestexpr = quote
    using Boltz, Random, Lux, Zygote
end

DocMeta.setdocmeta!(Boltz, :DocTestSetup, doctestexpr; recursive=true)

deploy_config = Documenter.auto_detect_deploy_system()
deploy_decision = Documenter.deploy_folder(
    deploy_config;
    repo="github.com/LuxDL/Boltz.jl",
    devbranch="main",
    devurl="dev",
    push_preview=true,
)

makedocs(;
    sitename="Boltz.jl Docs",
    authors="Avik Pal et al.",
    clean=true,
    modules=[Boltz],
    linkcheck=true,
    repo="https://github.com/LuxDL/Boltz.jl/blob/{commit}{path}#{line}",
    format=DocumenterVitepress.MarkdownVitepress(;
        repo="github.com/LuxDL/Boltz.jl", devbranch="main", devurl="dev", deploy_decision
    ),
    plugins=[CitationBibliography(joinpath(@__DIR__, "ref.bib"))],
    pages,
)

DocumenterVitepress.deploydocs(;
    repo="github.com/LuxDL/Boltz.jl.git",
    push_preview=true,
    target="build",
    devbranch="main",
)
