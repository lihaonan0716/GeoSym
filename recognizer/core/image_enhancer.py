"""
统一的图像增强处理器
支持多种AI模型（Gemini、Replicate、Azure等）

此文件为向后兼容保留，内部调用 ModelManager 实现
新功能和模型配置请直接使用 ModelManager
"""

import logging
import cv2
import numpy as np
from ..models import ModelManager

logger = logging.getLogger(__name__)


class ImageEnhancer:
    """
    统一的图像增强接口（向后兼容）
    
    实际功能由 ModelManager 实现
    所有模型配置在 models/config/models_config.yaml 中管理
    """
    
    def __init__(self):
        """初始化增强器"""
        # 创建模型管理器
        self.manager = ModelManager()
        
        # 为了向后兼容，保留这些属性
        self.supported_models = self.manager.get_available_models()
        self.default_prompts = {
            model_id: self.manager.get_default_prompt(model_id)
            for model_id in self.supported_models.keys()
        }
        
        logger.debug("ImageEnhancer 初始化完成，支持 %d 个模型", len(self.supported_models))
    
    def get_default_prompt(self, model_name):
        """
        获取模型的默认prompt
        
        Args:
            model_name: 模型名称/ID
        
        Returns:
            str: 默认prompt
        """
        return self.manager.get_default_prompt(model_name)
    
    def process_image(self, image, model_name, prompt):
        """
        处理图像（统一接口）
        
        Args:
            image: OpenCV格式图像 (BGR)
            model_name: 模型名称/ID
            prompt: 处理提示词
        
        Returns:
            dict: {
                'success': bool,
                'processed_image': OpenCV图像 (BGR) or None,
                'error': str or None,
                'response_text': str or None
            }
        """
        return self.manager.process_image(image, model_name, prompt)
    
    def reload_config(self):
        """重新加载配置（用于动态更新模型配置）"""
        self.manager.reload_config()
        self.supported_models = self.manager.get_available_models()
        self.default_prompts = {
            model_id: self.manager.get_default_prompt(model_id)
            for model_id in self.supported_models.keys()
        }
        print("[ImageEnhancer] 配置已重新加载")


# 测试代码
if __name__ == "__main__":
    enhancer = ImageEnhancer()
    
    print("\n支持的模型:")
    for key, name in enhancer.supported_models.items():
        print(f"  - {key}: {name}")
    
    print("\n默认Prompts:")
    for model, prompt in enhancer.default_prompts.items():
        print(f"\n{model}:")
        print(f"  {prompt[:100]}...")
