"""
Google Gemini图片处理模块
用于对坐标系图片进行预处理/增强
"""

import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Dict, Optional, Tuple
from google.oauth2 import service_account

from google.genai.types import (
    HttpOptions,
    Tool,
    ToolCodeExecution,
    GenerateContentConfig,
    ThinkingConfig, Part
)
from google import genai
from google.genai import types


_CREDENTIALS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "config", "gcp_credentials.json"
)


def _load_gcp_credentials():
    """Load GCP credentials from external config file."""
    if not os.path.exists(_CREDENTIALS_PATH):
        return None
    import json
    with open(_CREDENTIALS_PATH, "r", encoding="utf-8") as f:
        creds_data = json.load(f)
    creds = service_account.Credentials.from_service_account_info(creds_data)
    return creds.with_scopes(["https://www.googleapis.com/auth/cloud-platform"])


class GeminiImageProcessor:
    """使用Google Gemini API处理图片"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化Gemini处理器
        
        Args:
            api_key: Google API Key，如果不提供则从环境变量读取
        """
        self.api_key = api_key or os.environ.get('GOOGLE_API_KEY')
        self.client = None
        
    def _init_client(self):
        """懒加载初始化Gemini客户端"""
        if self.client is None:
            try:
                from google import genai
                creds = _load_gcp_credentials()
                if creds is None:
                    raise ValueError("GCP credentials not found. Place gcp_credentials.json in config/.")
                self.client = genai.Client(
                    http_options=HttpOptions(api_version="v1"),
                    credentials=creds,
                    vertexai=True,
                )
            except ImportError:
                raise ImportError("请安装google-genai库: pip install google-genai")
    
    def process_image(self, image: np.ndarray, prompt: str, 
                     model: str = "gemini-2.5-flash-image") -> Dict:
        """
        使用Gemini处理图片
        
        Args:
            image: OpenCV格式的图片(BGR)
            prompt: 处理提示词
            model: 使用的模型名称
        
        Returns:
            结果字典，包含：
            - success: 是否成功
            - processed_image: 处理后的图片(OpenCV BGR格式)，如果失败则为None
            - original_image: 原始图片
            - error: 错误信息（如果失败）
            - response_text: API返回的文本描述（如果有）
        """
        result = {
            'success': False,
            'processed_image': None,
            'original_image': image.copy(),
            'error': None,
            'response_text': None
        }
        
        try:
            # 初始化客户端
            self._init_client()
            
            # 转换OpenCV图片到PIL Image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            print(f"[Gemini] 调用API，模型: {model}")
            print(f"[Gemini] Prompt: {prompt[:100]}...")
            
            # 调用Gemini API
            response = self.client.models.generate_content(
                model=model,
                contents=[pil_image,prompt],
                config=GenerateContentConfig(
                    temperature=0,
                )
            )
            
            # 解析响应
            generated_image = None
            response_texts = []
            
            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    response_texts.append(part.text)
                    print(f"[Gemini] 返回文本: {part.text[:200]}...")
                elif part.inline_data is not None:
                    # 提取生成的图片
                    generated_image = Image.open(BytesIO(part.inline_data.data))
                    print(f"[Gemini] 返回图片，尺寸: {generated_image.size}")
            
            # 保存文本响应
            if response_texts:
                result['response_text'] = '\n'.join(response_texts)
            
            # 处理生成的图片
            if generated_image is not None:
                # 转换PIL Image到OpenCV格式
                generated_np = np.array(generated_image)
                
                # 处理不同的颜色格式
                if len(generated_np.shape) == 2:
                    # 灰度图
                    processed_cv = cv2.cvtColor(generated_np, cv2.COLOR_GRAY2BGR)
                elif generated_np.shape[2] == 3:
                    # RGB
                    processed_cv = cv2.cvtColor(generated_np, cv2.COLOR_RGB2BGR)
                elif generated_np.shape[2] == 4:
                    # RGBA
                    processed_cv = cv2.cvtColor(generated_np, cv2.COLOR_RGBA2BGR)
                else:
                    raise ValueError(f"不支持的图片格式: {generated_np.shape}")
                
                result['processed_image'] = processed_cv
                result['success'] = True
                print(f"[Gemini] ✅ 处理成功，输出图片尺寸: {processed_cv.shape}")
            else:
                result['error'] = "API没有返回图片，只返回了文本"
                print(f"[Gemini] ⚠️ {result['error']}")
            
        except ImportError as e:
            result['error'] = f"缺少依赖库: {str(e)}"
            print(f"[Gemini] ❌ {result['error']}")
        except ValueError as e:
            result['error'] = f"配置错误: {str(e)}"
            print(f"[Gemini] ❌ {result['error']}")
        except Exception as e:
            result['error'] = f"API调用失败: {str(e)}"
            print(f"[Gemini] ❌ {result['error']}")
            import traceback
            print(traceback.format_exc())
        
        return result

