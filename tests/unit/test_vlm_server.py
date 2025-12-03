"""Test VLM client and server detector."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
from PIL import Image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from object_detection.api.vllm_client import VLLMClient
from object_detection.models.vlm_server import VLMServerDetector
from object_detection.models.base import DetectionResult


def test_vllm_client_creation():
    """Test VLLMClient creation."""
    client = VLLMClient(
        base_url="http://localhost:8000",
        model_name="test-model"
    )
    
    assert client.base_url == "http://localhost:8000"
    assert client.model_name == "test-model"


def test_vllm_client_encode_image():
    """Test image encoding."""
    client = VLLMClient(
        base_url="http://localhost:8000",
        model_name="test-model"
    )
    
    # Test with PIL Image
    image = Image.new('RGB', (100, 100), color='red')
    encoded = client._encode_image(image)
    assert isinstance(encoded, str)
    assert len(encoded) > 0
    
    # Test with numpy array
    np_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    encoded = client._encode_image(np_image)
    assert isinstance(encoded, str)
    assert len(encoded) > 0


@pytest.mark.asyncio
async def test_vllm_client_generate_mock():
    """Test VLLMClient generate with mock."""
    with patch('aiohttp.ClientSession') as mock_session_class:
        # Setup mock
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            'choices': [{'text': 'person|0.9|0.1,0.2,0.3,0.4'}]
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()
        
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()
        mock_session_class.return_value = mock_session
        
        # Test
        client = VLLMClient(
            base_url="http://localhost:8000",
            model_name="test-model"
        )
        
        async with client:
            result = await client.generate(
                prompt="Detect objects",
                image=Image.new('RGB', (100, 100))
            )
        
        assert isinstance(result, str)
        assert 'person' in result


def test_vlm_server_detector_creation():
    """Test VLMServerDetector creation."""
    detector = VLMServerDetector(
        model_name="test-model",
        vllm_server_url="http://localhost:8000"
    )
    
    assert detector.model_name == "test-model"
    assert detector.vllm_server_url == "http://localhost:8000"
    assert detector.client is not None


def test_vlm_parse_detection_response():
    """Test parsing VLM detection responses."""
    detector = VLMServerDetector(
        model_name="test-model",
        vllm_server_url="http://localhost:8000",
        confidence_threshold=0.5
    )
    
    # Test standard format
    response = "person|0.9|0.1,0.2,0.3,0.4\ncar|0.8|0.5,0.6,0.7,0.8"
    result = detector._parse_detection_response(response, (480, 640))
    
    assert len(result.boxes) == 2
    assert len(result.scores) == 2
    assert result.scores[0] == 0.9
    assert result.scores[1] == 0.8
    
    # Check box coordinates are converted to absolute
    assert result.boxes[0][0] == 0.1 * 640  # x1
    assert result.boxes[0][1] == 0.2 * 480  # y1


def test_vlm_parse_detection_response_with_threshold():
    """Test filtering by confidence threshold."""
    detector = VLMServerDetector(
        model_name="test-model",
        vllm_server_url="http://localhost:8000",
        confidence_threshold=0.7
    )
    
    # Mix of high and low confidence
    response = "person|0.9|0.1,0.2,0.3,0.4\ncar|0.5|0.5,0.6,0.7,0.8\ndog|0.8|0.2,0.3,0.4,0.5"
    result = detector._parse_detection_response(response, (480, 640))
    
    # Should only include person (0.9) and dog (0.8), not car (0.5)
    assert len(result.boxes) == 2
    assert result.scores[0] >= 0.7
    assert result.scores[1] >= 0.7


def test_vlm_parse_empty_response():
    """Test parsing empty VLM response."""
    detector = VLMServerDetector(
        model_name="test-model",
        vllm_server_url="http://localhost:8000"
    )
    
    response = ""
    result = detector._parse_detection_response(response, (480, 640))
    
    assert len(result.boxes) == 0
    assert len(result.scores) == 0
    assert len(result.labels) == 0


def test_vlm_parse_malformed_response():
    """Test handling of malformed responses."""
    detector = VLMServerDetector(
        model_name="test-model",
        vllm_server_url="http://localhost:8000"
    )
    
    # Malformed lines should be skipped
    response = "person|0.9|0.1,0.2,0.3,0.4\ninvalid line\ncar|invalid|0.5,0.6,0.7,0.8"
    result = detector._parse_detection_response(response, (480, 640))
    
    # Should only parse the valid first line
    assert len(result.boxes) == 1
    assert result.scores[0] == 0.9


def test_vlm_parse_json_format():
    """Test parsing JSON-like format."""
    detector = VLMServerDetector(
        model_name="test-model",
        vllm_server_url="http://localhost:8000"
    )
    
    response = '{"class": "person", "bbox": [0.1, 0.2, 0.3, 0.4], "score": 0.9}'
    result = detector._parse_detection_response(response, (480, 640))
    
    assert len(result.boxes) == 1
    assert result.scores[0] == 0.9


def test_vlm_detector_info():
    """Test getting detector information."""
    detector = VLMServerDetector(
        model_name="test-model",
        vllm_server_url="http://localhost:8000",
        confidence_threshold=0.7
    )
    
    info = detector.get_info()
    
    assert info["model_name"] == "test-model"
    assert info["confidence_threshold"] == 0.7
