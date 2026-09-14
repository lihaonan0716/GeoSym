"""
Replicate API 适配器
支持 Flux-Kontext-Max 和 Flux-Kontext-Pro
"""

import cv2
import numpy as np
import requests
from .base_adapter import BaseModelAdapter


class ReplicateAdapter(BaseModelAdapter):
    """Replicate API 模型适配器"""
    
    def process_image(self, image, prompt):
        """
        使用 Replicate API 处理图像
        
        Args:
            image: OpenCV格式图像 (numpy.ndarray, BGR)
            prompt: 处理提示词
        
        Returns:
            dict: 处理结果
        """
        try:
            api_token = self.config.get('api_token')
            model_path = self.config.get('model_path')
            timeout = self.config.get('timeout', 120)
            
            if not api_token:
                return self._create_error_result('Replicate API token is not configured')
            
            print(f"[Replicate/{model_path}] textprocessing image...")
            
            # 步骤1: 上传图像到Replicate
            print(f"[Replicate/{model_path}] Step 1: uploading image...")
            image_url = self._upload_image(image, api_token)
            if not image_url:
                return self._create_error_result('Image upload failed')
            
            print(f"[Replicate/{model_path}] Image uploaded successfully: {image_url}")
            
            # 步骤2: 调用Replicate API
            print(f"[Replicate/{model_path}] Step 2: calling API...")
            output_url, logs = self._call_api(image_url, prompt, api_token, model_path, timeout)
            if not output_url:
                return self._create_error_result('API call failed', logs)
            
            print(f"[Replicate/{model_path}] API processing succeeded: {output_url}")
            
            # 步骤3: 下载处理后的图像
            print(f"[Replicate/{model_path}] Step 3: downloading processed image...")
            processed_image = self._download_image(output_url)
            if processed_image is None:
                return self._create_error_result('Image download failed', logs)
            
            print(f"[Replicate/{model_path}] ✅ Processing complete！")
            
            return self._create_success_result(
                processed_image,
                logs if logs else "处理成功"
            )
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"[ReplicateAdapter] Processing exception: {str(e)}")
            print(error_detail)
            
            return self._create_error_result(
                f'Replicate处理异常: {str(e)}',
                error_detail
            )
    
    def _upload_image(self, image, api_token):
        """上传图像到Replicate文件服务"""
        try:
            # 将OpenCV图像转换为JPEG字节
            success, encoded_image = cv2.imencode('.jpg', image)
            if not success:
                print("[ReplicateAdapter] Image encoding failed")
                return None
            
            image_bytes = encoded_image.tobytes()
            
            # 上传到Replicate
            files = {
                'content': ('image.jpg', image_bytes, 'image/jpeg')
            }
            
            headers = {
                'Authorization': f'Bearer {api_token}'
            }
            
            response = requests.post(
                'https://api.replicate.com/v1/files',
                headers=headers,
                files=files,
                timeout=30
            )
            
            if response.status_code != 201:
                print(f"[ReplicateAdapter] Upload failed: {response.status_code}")
                print(f"[ReplicateAdapter] Response: {response.text}")
                return None
            
            result = response.json()
            image_url = result.get('urls', {}).get('get')
            
            return image_url
            
        except Exception as e:
            print(f"[ReplicateAdapter] Upload exception: {str(e)}")
            return None
    
    def _call_api(self, image_url, prompt, api_token, model_path, timeout):
        """调用Replicate API处理图像"""
        try:
            headers = {
                'Authorization': f'Bearer {api_token}',
                'Content-Type': 'application/json',
                'Prefer': 'wait'  # 等待处理完成
            }
            
            # 获取参数配置
            params = self.config.get('params', {})
            
            data = {
                'input': {
                    'prompt': prompt,
                    'input_image': image_url,
                    'aspect_ratio': params.get('aspect_ratio', 'match_input_image'),
                    'output_format': params.get('output_format', 'jpg'),
                    'safety_tolerance': params.get('safety_tolerance', 2),
                    'prompt_upsampling': params.get('prompt_upsampling', False)
                }
            }
            
            api_url = f'https://api.replicate.com/v1/models/{model_path}/predictions'
            print(f"[ReplicateAdapter] Calling model: {model_path}")
            
            response = requests.post(
                api_url,
                headers=headers,
                json=data,
                timeout=timeout
            )
            
            if response.status_code not in [200, 201]:
                error_msg = f"API call failed: {response.status_code} - {response.text}"
                print(f"[ReplicateAdapter] {error_msg}")
                return None, error_msg
            
            result = response.json()
            
            # 检查状态
            status = result.get('status')
            if status == 'succeeded':
                output_url = result.get('output')
                logs = result.get('logs', '')
                return output_url, logs
            else:
                error = result.get('error', f'处理失败，状态: {status}')
                print(f"[ReplicateAdapter] {error}")
                return None, error
            
        except requests.Timeout:
            error_msg = f"API call timed out ({timeout}s)"
            print(f"[ReplicateAdapter] {error_msg}")
            return None, error_msg
        except Exception as e:
            error_msg = f"API call exception: {str(e)}"
            print(f"[ReplicateAdapter] {error_msg}")
            return None, error_msg
    
    def _download_image(self, url):
        """从URL下载图像"""
        try:
            response = requests.get(url, timeout=30)
            
            if response.status_code != 200:
                print(f"[ReplicateAdapter] Download failed: {response.status_code}")
                return None
            
            # 将字节转换为OpenCV图像
            image_array = np.frombuffer(response.content, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                print("[ReplicateAdapter] imagedecode failed")
                return None
            
            return image
            
        except Exception as e:
            print(f"[ReplicateAdapter] Download exception: {str(e)}")
            return None

