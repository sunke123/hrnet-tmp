from .resnet import ResNet
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .hrnet import HighResolutionNet
from .hrnet_sync import SyncHighResolutionNet
from .sync_resnet import SyncResNet
__all__ = ['ResNet', 'ResNeXt', 'SSDVGG', 'HighResolutionNet', 'SyncHighResolutionNet', 'SyncResNet']
