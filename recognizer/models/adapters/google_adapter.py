"""
Google Gemini 适配器
封装 GeminiImageProcessor
"""

from .base_adapter import BaseModelAdapter


class GoogleAdapter(BaseModelAdapter):
    """Google Gemini 模型适配器"""
    
    def process_image(self, image, prompt):
        """
        使用 Google Gemini 处理图像
        
        Args:
            image: OpenCV格式图像 (numpy.ndarray, BGR)
            prompt: 处理提示词
        
        Returns:
            dict: 处理结果
        """
        try:
            # 导入 GeminiImageProcessor
            from gemini_image_processor import GeminiImageProcessor
            
            # 创建处理器（会自动读取 API key）
            processor = GeminiImageProcessor()
            
            # 如果配置中指定了 API key，设置到环境变量
            api_key = self.config.get('api_key')
            if api_key and api_key != "your_gemini_api_key_here":
                import os
                os.environ['GOOGLE_API_KEY'] = api_key
            
            # 调用处理
            result = processor.process_image(image, prompt)
            
            return result
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"[GoogleAdapter] 处理失败: {str(e)}")
            print(error_detail)
            
            return self._create_error_result(
                f'Gemini处理失败: {str(e)}',
                error_detail
            )

