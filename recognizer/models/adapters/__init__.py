"""
模型适配器
为不同的AI服务提供商提供统一接口
"""

from .base_adapter import BaseModelAdapter
from .google_adapter import GoogleAdapter
from .replicate_adapter import ReplicateAdapter
from .azure_adapter import AzureAdapter

__all__ = [
    'BaseModelAdapter',
    'GoogleAdapter',
    'ReplicateAdapter',
    'AzureAdapter'
]

