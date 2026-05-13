"""
模型管理器
负责加载配置、管理模型适配器、提供统一的图像处理接口
"""

import logging
import os
import yaml
import cv2
from .adapters import (
    GoogleAdapter,
    ReplicateAdapter,
    AzureAdapter
)

logger = logging.getLogger(__name__)


class ModelManager:
    """AI模型管理器"""
    
    def __init__(self, config_path=None):
        """
        初始化模型管理器
        
        Args:
            config_path: 配置文件路径，默认为 models/config/models_config.yaml
        """
        if config_path is None:
            # 默认配置文件路径（相对于当前文件）
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, 'config', 'models_config.yaml')
        
        self.config_path = config_path
        self.models = {}
        self.adapters = {}
        
        # 适配器映射
        self.adapter_classes = {
            'google': GoogleAdapter,
            'replicate': ReplicateAdapter,
            'azure': AzureAdapter
        }
        
        # 加载配置
        self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            models_config = config_data.get('models', [])
            
            # 解析每个模型配置
            for model_config in models_config:
                model_id = model_config.get('id')
                enabled = model_config.get('enabled', True)
                
                if not model_id:
                    logger.warning("发现无效的模型配置（缺少 id）")
                    continue
                
                if not enabled:
                    logger.debug("跳过已禁用的模型: %s", model_id)
                    continue
                
                # 存储模型配置
                self.models[model_id] = model_config
                
                # 创建适配器
                provider = model_config.get('provider')
                adapter_class = self.adapter_classes.get(provider)
                
                if adapter_class:
                    adapter_config = model_config.get('config', {})
                    self.adapters[model_id] = adapter_class(adapter_config)
                    logger.debug("加载模型: %s (%s)", model_id, model_config.get('name'))
                else:
                    logger.warning("不支持的提供商: %s (模型: %s)", provider, model_id)
            
            logger.debug("共加载 %d 个模型", len(self.models))
            
        except FileNotFoundError:
            logger.error("配置文件不存在: %s", self.config_path)
            raise
        except yaml.YAMLError as e:
            logger.error("配置文件解析失败: %s", e)
            raise
        except Exception as e:
            logger.error("加载配置失败: %s", e)
            raise
    
    def get_available_models(self):
        """
        获取所有可用的模型列表
        
        Returns:
            dict: {model_id: model_name}
        """
        return {
            model_id: model_config.get('name', model_id)
            for model_id, model_config in self.models.items()
        }
    
    def get_model_info(self, model_id):
        """
        获取模型详细信息
        
        Args:
            model_id: 模型ID
        
        Returns:
            dict: 模型配置，不存在返回 None
        """
        return self.models.get(model_id)
    
    def get_default_prompt(self, model_id):
        """
        获取模型的默认 prompt
        
        Args:
            model_id: 模型ID
        
        Returns:
            str: 默认 prompt，不存在返回空字符串
        """
        model_config = self.models.get(model_id)
        if model_config:
            return model_config.get('default_prompt', '').strip()
        return ''
    
    def process_image(self, image, model_id, prompt=None):
        """
        使用指定模型处理图像
        
        Args:
            image: OpenCV格式图像 (numpy.ndarray, BGR)
            model_id: 模型ID
            prompt: 处理提示词，如果为 None 则使用默认 prompt
        
        Returns:
            dict: {
                'success': bool,
                'processed_image': numpy.ndarray (BGR) or None,
                'error': str or None,
                'response_text': str or None
            }
        """
        # 检查模型是否存在
        if model_id not in self.models:
            available_models = ', '.join(self.models.keys())
            return {
                'success': False,
                'processed_image': None,
                'error': f'模型 {model_id} 不存在。可用模型: {available_models}',
                'response_text': None
            }
        
        # 获取适配器
        adapter = self.adapters.get(model_id)
        if not adapter:
            return {
                'success': False,
                'processed_image': None,
                'error': f'模型 {model_id} 的适配器未初始化',
                'response_text': None
            }
        
        # 如果没有提供 prompt，使用默认值
        if prompt is None or prompt.strip() == '':
            prompt = self.get_default_prompt(model_id)
        
        # 记录原始图片尺寸
        original_height, original_width = image.shape[:2]
        
        # 将图片拉伸成正方形（以最小边为基准）
        try:
            min_size = min(original_height, original_width)
            square_image = cv2.resize(image, (min_size, min_size), interpolation=cv2.INTER_LINEAR)
            logger.debug("图片已从 %dx%d 拉伸为 %dx%d", original_width, original_height, min_size, min_size)
        except Exception as e:
            logger.warning("图片拉伸失败，使用原图: %s", e)
            square_image = image
        
        logger.debug("使用模型 %s 处理图像...", model_id)
        result = adapter.process_image(square_image, prompt)
        
        # 如果处理成功且返回了图片，将图片拉伸回原始尺寸
        if result.get('success') and result.get('processed_image') is not None:
            try:
                processed_image = result['processed_image']
                # 将处理后的图片拉伸回原始尺寸
                restored_image = cv2.resize(
                    processed_image, 
                    (original_width, original_height), 
                    interpolation=cv2.INTER_LINEAR
                )
                result['processed_image'] = restored_image
                logger.debug("图片已恢复为原始尺寸 %dx%d", original_width, original_height)
            except Exception as e:
                logger.warning("图片尺寸恢复失败: %s", e)
                # 如果恢复失败，保持处理后的正方形图片
        
        return result
    
    def reload_config(self):
        """重新加载配置文件"""
        self.models = {}
        self.adapters = {}
        self._load_config()
        logger.debug("配置已重新加载")


# 测试代码
if __name__ == "__main__":
    manager = ModelManager()
    
    print("\n可用模型:")
    for model_id, model_name in manager.get_available_models().items():
        print(f"  - {model_id}: {model_name}")
    
    print("\n默认Prompts:")
    for model_id in manager.get_available_models().keys():
        prompt = manager.get_default_prompt(model_id)
        print(f"\n{model_id}:")
        print(f"  {prompt[:100]}...")

