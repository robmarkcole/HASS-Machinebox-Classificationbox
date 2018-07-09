"""The tests for the classificationbox component."""
from unittest.mock import Mock, patch

import pytest
import requests
import requests_mock

from homeassistant.core import callback
from homeassistant.const import (
    ATTR_ID, ATTR_ENTITY_ID, ATTR_NAME, CONF_FRIENDLY_NAME,
    CONF_IP_ADDRESS, CONF_PORT, STATE_UNKNOWN)
from homeassistant.setup import async_setup_component
import homeassistant.components.image_processing as ip
import homeassistant.components.image_processing.classificationbox as cb

MOCK_IP = '192.168.0.1'
MOCK_PORT = '8080'

MOCK_FILE_PATH = '/images/mock.jpg'

MOCK_JSON = {'success': True,
             'classes': [{'id': 'birds', 'score': 0.915892},
                         {'id': 'not_birds', 'score': 0.084108}]}

MOCK_NO_MODELS = {'success': True, 'models': []}
MOCK_MODELS = [{'id': '12345', 'name': '12345'}]
MOCK_MODEL_ID = '12345'
MOCK_NAME = 'mock_name'

# Classes data after parsing.
PARSED_CLASSES = [{ATTR_ID: 'birds', ip.ATTR_CONFIDENCE: 91.59},
                  {ATTR_ID: 'not_birds', ip.ATTR_CONFIDENCE: 8.41}]

MATCHED_CLASSES = {'birds': 91.59, 'not_birds': 8.41}

VALID_ENTITY_ID = 'image_processing.classificationbox_demo_camera_12345'
VALID_CONFIG = {
    ip.DOMAIN: {
        'platform': 'classificationbox',
        CONF_IP_ADDRESS: MOCK_IP,
        CONF_PORT: MOCK_PORT,
        ip.CONF_SOURCE: {
            ip.CONF_ENTITY_ID: 'camera.demo_camera'}
        },
    'camera': {
        'platform': 'demo'
        }
    }


def test_encode_image():
    """Test that binary data is encoded correctly."""
    assert cb.encode_image(b'test') == 'dGVzdA=='


def test_get_matched_classes():
    """Test that matched_classes are parsed correctly."""
    assert cb.get_matched_classes(PARSED_CLASSES) == MATCHED_CLASSES


def test_parse_classes():
    """Test parsing of raw face data, and generation of matched_faces."""
    assert cb.parse_classes(MOCK_JSON['classes']) == PARSED_CLASSES


@pytest.fixture
def mock_image():
    """Return a mock camera image."""
    with patch('homeassistant.components.camera.demo.DemoCamera.camera_image',
               return_value=b'Test') as image:
        yield image


async def test_setup_platform(hass):
    """Setup platform with one entity."""
    with patch('homeassistant.components.image_processing.classificationbox.get_models',
               return_value=MOCK_MODELS):
        await async_setup_component(hass, ip.DOMAIN, VALID_CONFIG)
        await hass.async_block_till_done()
        assert hass.states.get(VALID_ENTITY_ID)
