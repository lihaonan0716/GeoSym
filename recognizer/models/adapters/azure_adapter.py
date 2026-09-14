"""
Azure OpenAI 适配器
支持 Azure 部署的 Flux-Kontext-Pro
"""

import cv2
import numpy as np
import requests
import base64
from io import BytesIO
from PIL import Image
from .base_adapter import BaseModelAdapter

import sys as _sys
from pathlib import Path as _Path
_exp_root = _Path(__file__).resolve().parent.parent.parent.parent
if str(_exp_root) not in _sys.path:
    _sys.path.insert(0, str(_exp_root))

# Try to import serverconfig (optional, for standalone use)
try:
    from ... import serverconfig
except ImportError:
    try:
        import serverconfig
    except ImportError:
        serverconfig = None

class AzureAdapter(BaseModelAdapter):
    """Azure OpenAI 模型适配器"""
    
    def process_image(self, image, prompt):
        """
        使用 Azure OpenAI API 处理图像
        
        Args:
            image: OpenCV格式图像 (numpy.ndarray, BGR)
            prompt: 处理提示词
        
        Returns:
            dict: 处理结果
        """
        try:
            endpoint = self.config.get('endpoint')
            deployment = self.config.get('deployment')
            api_version = self.config.get('api_version')
            api_key = self.config.get('api_key')
            timeout = self.config.get('timeout', 120)
            
            if not all([endpoint, deployment, api_version, api_key]):
                return self._create_error_result('Azure configuration is incomplete; check the configuration file')
            
            print(f"[Azure/{deployment}] textprocessing image...")
            
            # 保存输入图像（调试用）
            try:
                import os
                import time
                debug_dir = getattr(serverconfig, "OUTPUT_DIR", None) or "debug_stage05"
                os.makedirs(debug_dir, exist_ok=True)
                timestamp = int(time.time() * 1000)

                # 保存原始输入图像
                cv2.imwrite(f"{debug_dir}/1_input_before_ai_{timestamp}.jpg", image)
                print(f"[Debug] Stage 0.5 input image saved: {debug_dir}/1_input_before_ai_{timestamp}.jpg")
            except Exception as e:
                print(f"[Debug] Failed to save input image: {str(e)}")
            
            # 构建 API URL（使用 /edits 端点进行图像编辑）
            # 确保 endpoint 有尾部斜杠
            endpoint_clean = endpoint.rstrip('/') + '/'
            base_path = f'openai/deployments/{deployment}/images'
            params = f'?api-version={api_version}'
            edit_url = f"{endpoint_clean}{base_path}/edits{params}"
            
            print(f"[Azure/{deployment}] API endpoint: {edit_url}")
            
            # 调用 Azure API
            processed_image = self._call_azure_api(
                edit_url, 
                image, 
                prompt, 
                api_key, 
                timeout
            )
            
            if processed_image is None:
                return self._create_error_result('Azure API call failed')
            
            # 保存AI增强后的图像（调试用）
            try:
                cv2.imwrite(f"{debug_dir}/2_output_after_ai_{timestamp}.jpg", processed_image)
                print(f"[Debug] Stage 0.5 output image saved: {debug_dir}/2_output_after_ai_{timestamp}.jpg")
                
                # 保存对比图
                if image.shape == processed_image.shape:
                    # 创建左右对比图
                    comparison = np.hstack([image, processed_image])
                    cv2.imwrite(f"{debug_dir}/3_comparison_{timestamp}.jpg", comparison)
                    print(f"[Debug] Comparison image saved: {debug_dir}/3_comparison_{timestamp}.jpg")
            except Exception as e:
                print(f"[Debug] Failed to save output image: {str(e)}")
            
            print(f"[Azure/{deployment}] ✅ Processing complete！")
            
            return self._create_success_result(
                processed_image,
                "处理成功"
            )
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"[AzureAdapter] Processing exception: {str(e)}")
            print(error_detail)
            
            return self._create_error_result(
                f'Azure处理异常: {str(e)}',
                error_detail
            )
    
    def _call_azure_api(self, edit_url, image, prompt, api_key, timeout):
        """
        调用 Azure API 进行图像编辑
        
        Args:
            edit_url: API 端点 URL
            image: OpenCV格式图像
            prompt: 处理提示词
            api_key: API 密钥
            timeout: 超时时间
        
        Returns:
            numpy.ndarray: 处理后的图像，失败返回 None
        """
        try:
            # 将 OpenCV 图像转换为 PNG 字节流
            # success, encoded_image = cv2.imencode('.png', image)
            # if not success:
            #     print("[AzureAdapter] Image encoding failed")
            #     return None
            
            # image_bytes = encoded_image.tobytes()
            
            # # 获取配置的参数
            # params = self.config.get('params', {})
            
            # # 构建请求数据（multipart/form-data）
            # files = {
            #     'image': ('image.png', image_bytes, 'image/png'),
            # }
            
            # 将OpenCV图像转换为JPEG字节
            success, encoded_image = cv2.imencode('.jpg', image)
            if not success:
                print("[ReplicateAdapter] Image encoding failed")
                return None
            
            image_bytes = encoded_image.tobytes()
            
            # 上传到Replicate
            files = {
                'image': ('image.jpg', image_bytes, 'image/jpeg')
            }

            # 获取配置的参数
            params = self.config.get('params', {})

            # 构建请求参数（使用 aspect_ratio 而不是 size）
            data = {
                'prompt': prompt,
                # 'n': params.get('n', 1),
                'aspect_ratio': params.get('aspect_ratio', 'match_input_image'),  # 关键修改
                'model': params.get('model', 'flux.1-kontext-pro'),
                'output_format': params.get('output_format', 'jpg')  # 改为 jpg
            }
            
            headers = {
                'Api-Key': api_key
            }
            
            print(f"[AzureAdapter] Sending request...")
            # print(f"[AzureAdapter] Prompt: {prompt[:100]}...")
            print(f"[AzureAdapter] Prompt: {prompt}")
            
            # 发送请求
            response = requests.post(
                edit_url,
                headers=headers,
                data=data,
                files=files,
                timeout=timeout
            )
            
            if response.status_code != 200:
                print(f"[AzureAdapter] API call failed: {response.status_code}")
                print(f"[AzureAdapter] Response: {response.text}")
                return None
            
            # 解析响应
            result = response.json()
            print(f"[AzureAdapter] APIResponse: {result}")
            
            # 提取 base64 编码的图像
            data = result.get('data', [])
            if not data:
                print("[AzureAdapter] Responsetext")
                return None
            
            b64_json = data[0].get('b64_json')
            if not b64_json:
                print("[AzureAdapter] Responsetext b64_json field")
                return None
            
            # 解码 base64 图像
            processed_image = self._decode_base64_image(b64_json)
            
            if processed_image is None:
                print("[AzureAdapter] imagedecode failed")
                return None
            
            return processed_image
            
        except requests.Timeout:
            print(f"[AzureAdapter] Request timed out ({timeout}s)")
            return None
        except Exception as e:
            print(f"[AzureAdapter] API call exception: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _decode_base64_image(self, b64_data):
        """
        解码 base64 图像数据为 OpenCV 格式
        
        Args:
            b64_data: base64 编码的图像字符串
        
        Returns:
            numpy.ndarray: OpenCV格式图像 (BGR)，失败返回 None
        """
        try:
            # 解码 base64
            image_data = base64.b64decode(b64_data)
            
            # 使用 PIL 打开图像
            pil_image = Image.open(BytesIO(image_data))
            
            # 转换为 numpy 数组
            image_array = np.array(pil_image)
            
            # 如果是 RGBA，转换为 RGB
            if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            
            # 转换为 OpenCV 的 BGR 格式
            if len(image_array.shape) == 3:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                # 灰度图
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            
            return image_bgr
            
        except Exception as e:
            print(f"[AzureAdapter] Decode exception: {str(e)}")
            import traceback

            traceback.print_exc()
            return None

