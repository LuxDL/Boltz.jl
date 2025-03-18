const POINT2VEC_PRETRAINED_BASE_URL = "https://github.com/kabouzeid/point2vec/releases/download/paper/"

const POINT2VEC_PRETRAINED_MODELS = Dict(
    :embedding_shapenet => "pre_point2vec-epoch.799-step.64800.ckpt",
    :classification_modelnet40 => "fine_modelnet40-epoch.125-step.38682-val_acc.0.9465.ckpt",
    :classification_scanobject => "fine_modelnet10-epoch.125-step.38682-val_acc.0.9465.ckpt",
    :segmentation_shapenetpart => "fine_point2vec-epoch.125-step.38682-val_acc.0.9465.ckpt",
)
