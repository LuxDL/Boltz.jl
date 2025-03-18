module PretrainedWeights

using Artifacts: Artifacts, @artifact_str
using Downloads: Downloads
using LazyArtifacts: LazyArtifacts
using Scratch: @get_scratch!

abstract type AbstractPretrainedWeights end

## Subtypes need to implement the following methods
load_with(::Val{:JLD2}, ::AbstractPretrainedWeights) = false
load_with(::Val{:Pickle}, ::AbstractPretrainedWeights) = false

## Optional methods
function checkpoint_extension(weight::AbstractPretrainedWeights)
    load_with(Val(:JLD2), weight) && return :jld2
    # For pickle there are quite a few possible extensions, so we simply error
    return error(
        "Unknown pretrained weights format. Implement `checkpoint_extension` for the \
         desired format."
    )
end

#  Use Artifacts.jl & LazyArtifacts.jl to download pretrained weights
#  This is the preferred method for downloading pretrained weights, but in some cases
#  we don't want to redistribute the weights with the package, so we use the other options
#  Required fields:
##   artifact_name::String (If not provided, must implement `get_artifact_name`)
#  Optional fields:
##   path::String - if not provided inferred from artifact_name and checkpoint_extension
abstract type ArtifactsPretrainedWeight <: AbstractPretrainedWeights end

get_artifact_name(weight::ArtifactsPretrainedWeight) = weight.artifact_name

function get_checkpoint_path(weight::ArtifactsPretrainedWeight)
    return "$(get_artifact_name(weight)).$(checkpoint_extension(weight))"
end

function download_and_get_checkpoint_path(weight::ArtifactsPretrainedWeight)
    name = get_artifact_name(weight)
    try
        return joinpath(@artifact_str(name), get_checkpoint_path(weight))
    catch err
        err isa ErrorException &&
            throw(ArgumentError("no pretrained weights available for `$(name)`"))
        rethrow(err)
    end
end

# Use Downloads.jl to download pretrained weights
#  Required fields:
##   url::String
#  Optional fields:
##   name::String (If not provided, inferred from url using `basename(url)`)
abstract type DownloadsPretrainedWeight <: AbstractPretrainedWeights end

function get_checkpoint_path(weight::DownloadsPretrainedWeight)
    name = hasfield(typeof(weight), :name) ? weight.name : basename(weight.url)
    ext = checkpoint_extension(weight)
    _, ext_from_name = splitext(name)
    @assert ext == ext_from_name "Extension mismatch for pretrained weights: \
                                  $(ext) ≠ $(ext_from_name)"
    return name
end

function download_and_get_checkpoint_path(weight::DownloadsPretrainedWeight)
    ckpt_name = get_checkpoint_path(weight)
    name = first(splitext(ckpt_name))
    ckpt_path = joinpath(@get_scratch!(name), ckpt_name)
    !isfile(ckpt_path) && Downloads.download(weight.url, ckpt_path)
    return ckpt_path
end

end
