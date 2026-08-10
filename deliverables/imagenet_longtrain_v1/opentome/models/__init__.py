"""CV model registrations included in the ImageNet delivery."""

from opentome.models.deit.deit import DeiTModel, deit_s, deit_s_extend
from opentome.models.mergenet.model import HybridToMeModel

__all__ = [
    'DeiTModel', 'deit_s', 'deit_s_extend',
    'HybridToMeModel',
]
