from srl.modules import timm
from srl.modules.decoders import build as build_decoder
from srl.modules.encoders import build as build_encoder
from srl.modules.groupers import build as build_grouper
from srl.modules.projector import build as build_projector
from srl.modules.initializers import build as build_initializer
from srl.modules.networks import build as build_network
from srl.modules.utils import Resizer, SoftToHardMask
from srl.modules.utils import build as build_utils
from srl.modules.utils import build_module, build_torch_function, build_torch_module
from srl.modules.video import LatentProcessor, MapOverTime, ScanOverTime
from srl.modules.video import build as build_video

__all__ = [
    "build_decoder",
    "build_encoder",
    "build_grouper",
    "build_projector",
    "build_initializer",
    "build_network",
    "build_utils",
    "build_module",
    "build_torch_module",
    "build_torch_function",
    "timm",
    "MapOverTime",
    "ScanOverTime",
    "LatentProcessor",
    "Resizer",
    "SoftToHardMask",
]


BUILD_FNS_BY_MODULE_GROUP = {
    "decoders": build_decoder,
    "encoders": build_encoder,
    "groupers": build_grouper,
    "projectors": build_projector,
    "initializers": build_initializer,
    "networks": build_network,
    "utils": build_utils,
    "video": build_video,
    "torch": build_torch_function,
    "torch.nn": build_torch_module,
    "nn": build_torch_module,
}
