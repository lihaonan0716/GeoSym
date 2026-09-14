"""
基础模型适配器
所有具体适配器的抽象基类
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseModelAdapter(ABC):
    """所有模型适配器的基类"""
    
    def __init__(self, config):
        """
        初始化适配器
        
        Args:
            config: 模型配置字典（从 YAML 中读取的 config 部分）
        """
        self.config = config
    
    @abstractmethod
    def process_image(self, image, prompt):
        """
        处理图像（子类必须实现）
        
        Args:
            image: OpenCV格式图像 (numpy.ndarray, BGR)
            prompt: 处理提示词
        
        Returns:
            dict: {
                'success': bool,
                'processed_image': numpy.ndarray (BGR) or None,
                'error': str or None,
                'response_text': str or None
            }
        """
        raise NotImplementedError("Subclasses must implement process_image")
    
    def _create_error_result(self, error_message, response_text=None):
        """创建错误结果"""
        return {
            'success': False,
            'processed_image': None,
            'error': error_message,
            'response_text': response_text
        }
    
    def _create_success_result(self, processed_image, response_text=None):
        """创建成功结果"""
        return {
            'success': True,
            'processed_image': processed_image,
            'error': None,
            'response_text': response_text
        }

