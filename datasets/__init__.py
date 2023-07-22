# from .baseset import BaseSet
#from .transform_wrapper import TRANSFORMS
def build_dataset(image_set, args):

    from .wsi_feat_dataset import build as build_wsi_feat_dataset
    return build_wsi_feat_dataset(image_set, args)

