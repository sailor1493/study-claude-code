"""VLLM client for async requests to VLLM servers."""

import asyncio
import aiohttp
import base64
import io
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import numpy as np


class VLLMClient:
    """
    Client for making async requests to VLLM servers.
    
    Supports vision-language models for object detection tasks.
    """
    
    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """
        Initialize VLLM client.
        
        Args:
            base_url: Base URL of VLLM server (e.g., "http://localhost:8000")
            model_name: Model name on VLLM server
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Enter async context manager."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager."""
        if self.session:
            await self.session.close()
    
    def _encode_image(self, image: Union[Image.Image, np.ndarray]) -> str:
        """
        Encode image to base64 string.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Base64 encoded image string
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')
    
    async def generate(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, np.ndarray]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        Generate text response from VLLM server.
        
        Args:
            prompt: Text prompt for the model
            image: Optional image for vision-language models
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        # Add image if provided
        if image is not None:
            payload["image"] = self._encode_image(image)
        
        url = f"{self.base_url}/v1/completions"
        
        for attempt in range(self.max_retries):
            try:
                async with self.session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['text']
                    elif response.status >= 500:
                        # Server error, retry
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                    
                    # Client error or last retry
                    error_text = await response.text()
                    raise RuntimeError(
                        f"VLLM request failed with status {response.status}: {error_text}"
                    )
            
            except asyncio.TimeoutError:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"VLLM request timed out after {self.timeout}s")
            
            except aiohttp.ClientError as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"VLLM request failed: {e}")
        
        raise RuntimeError("Max retries exceeded")
    
    async def batch_generate(
        self,
        prompts: List[str],
        images: Optional[List[Union[Image.Image, np.ndarray]]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs
    ) -> List[str]:
        """
        Generate text responses for multiple prompts in parallel.
        
        Args:
            prompts: List of text prompts
            images: Optional list of images (must match prompts length)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated text responses
        """
        if images is not None and len(images) != len(prompts):
            raise ValueError("Number of images must match number of prompts")
        
        tasks = []
        for i, prompt in enumerate(prompts):
            image = images[i] if images else None
            task = self.generate(
                prompt=prompt,
                image=image,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    async def health_check(self) -> bool:
        """
        Check if VLLM server is healthy.
        
        Returns:
            True if server is healthy, False otherwise
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
        
        try:
            url = f"{self.base_url}/health"
            async with self.session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=5.0)
            ) as response:
                return response.status == 200
        except Exception:
            return False


def create_vllm_client(
    base_url: str,
    model_name: str,
    **kwargs
) -> VLLMClient:
    """
    Factory function to create VLLM client.
    
    Args:
        base_url: Base URL of VLLM server
        model_name: Model name on VLLM server
        **kwargs: Additional client parameters
        
    Returns:
        VLLMClient instance
    """
    return VLLMClient(base_url=base_url, model_name=model_name, **kwargs)
