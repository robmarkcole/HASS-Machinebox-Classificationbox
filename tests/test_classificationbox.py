"""The tests for the classificationbox component."""
from unittest.mock import Mock, patch

import pytest
import requests
import requests_mock

from homeassistant.core import callback
from homeassistant.const import (
    ATTR_ID, ATTR_ENTITY_ID, ATTR_NAME, CONF_FRIENDLY_NAME, CONF_PASSWORD,
    CONF_USERNAME, CONF_IP_ADDRESS, CONF_PORT, HTTP_BAD_REQUEST, HTTP_OK, 
    HTTP_UNAUTHORIZED, STATE_UNKNOWN)
from homeassistant.setup import async_setup_component
import homeassistant.components.image_processing as ip
import homeassistant.components.image_processing.classificationbox as cb

MOCK_BOX_ID = 'b893cc4f7fd6'
MOCK_IP = '192.168.0.1'
MOCK_PORT = '8080'

MOCK_FILE_PATH = '/images/mock.jpg'

MOCK_HEALTH = {'success': True,
               'hostname': 'b893cc4f7fd6',
               'metadata': {'boxname': 'facebox', 'build': 'development'},
               'errors': []}

MOCK_JSON = {'success': True,
             'classes': [{'id': 'birds', 'score': 0.915892},
                         {'id': 'not_birds', 'score': 0.084108}]}

MOCK_NO_MODELS = {'success': True, 'models': []}
MOCK_MODELS = [{'id': '12345', 'name': '12345'}]
MOCK_MODEL_ID = '12345'
MOCK_NAME = 'mock_name'
MOCK_USERNAME = 'mock_username'
MOCK_PASSWORD = 'mock_password'

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


@pytest.fixture
def mock_healthybox():
    """Mock cb.check_box_health."""
    check_box_health = 'homeassistant.components.image_processing.' \
        'classificationbox.check_box_health'
    with patch(check_box_health, return_value=MOCK_BOX_ID) as _mock_healthybox:
        yield _mock_healthybox


@pytest.fixture
def mock_image():
    """Return a mock camera image."""
    with patch('homeassistant.components.camera.demo.DemoCamera.camera_image',
               return_value=b'Test') as image:
        yield image


def test_check_box_health(caplog):
    """Test check box health."""
    with requests_mock.Mocker() as mock_req:
        url = "http://{}:{}/healthz".format(MOCK_IP, MOCK_PORT)
        mock_req.get(url, status_code=HTTP_OK, json=MOCK_HEALTH)
        assert cb.check_box_health(url, 'user', 'pass') == MOCK_BOX_ID

        mock_req.get(url, status_code=HTTP_UNAUTHORIZED)
        assert cb.check_box_health(url, None, None) is None
        assert "AuthenticationError on classificationbox" in caplog.text

        mock_req.get(url, exc=requests.exceptions.ConnectTimeout)
        cb.check_box_health(url, None, None)
        assert "ConnectionError: Is classificationbox running?" in caplog.text


def test_encode_image():
    """Test that binary data is encoded correctly."""
    assert cb.encode_image(b'test') == 'dGVzdA=='


def test_get_matched_classes():
    """Test that matched_classes are parsed correctly."""
    assert cb.get_matched_classes(PARSED_CLASSES) == MATCHED_CLASSES


def test_parse_classes():
    """Test parsing of raw API data"""
    assert cb.parse_classes(MOCK_JSON['classes']) == PARSED_CLASSES


async def test_setup_platform(hass, mock_healthybox):
    """Setup platform with one entity."""
    with patch('homeassistant.components.image_processing.classificationbox.get_models',
               return_value=MOCK_MODELS):
        await async_setup_component(hass, ip.DOMAIN, VALID_CONFIG)
        await hass.async_block_till_done()
        assert hass.states.get(VALID_ENTITY_ID)


async def test_setup_platform_with_auth(hass, mock_healthybox):
    """Setup platform with one entity and auth."""
    valid_config_auth = VALID_CONFIG.copy()
    valid_config_auth[ip.DOMAIN][CONF_USERNAME] = MOCK_USERNAME
    valid_config_auth[ip.DOMAIN][CONF_PASSWORD] = MOCK_PASSWORD
    with patch('homeassistant.components.image_processing.classificationbox.get_models',
                return_value=MOCK_MODELS):
        await async_setup_component(hass, ip.DOMAIN, valid_config_auth)
        assert hass.states.get(VALID_ENTITY_ID)


async def test_process_image(hass, mock_image, mock_healthybox):
    """Test processing of an image."""
    with patch('homeassistant.components.image_processing.classificationbox.get_models',
               return_value=MOCK_MODELS):
        await async_setup_component(hass, ip.DOMAIN, VALID_CONFIG)
        await hass.async_block_till_done()
        assert hass.states.get(VALID_ENTITY_ID)

    classification_events = []

    @callback
    def mock_classification_event(event):
        """Mock event."""
        classification_events.append(event)

    hass.bus.async_listen('image_processing.image_classification',
                          mock_classification_event)

    with requests_mock.Mocker() as mock_req:
        url = 'http://{}:{}/{}/models/{}/predict'.format(
            MOCK_IP,
            MOCK_PORT,
            cb.CLASSIFIER,
            MOCK_MODEL_ID)
        mock_req.post(url, json=MOCK_JSON)
        data = {ATTR_ENTITY_ID: VALID_ENTITY_ID}
        await hass.services.async_call(ip.DOMAIN,
                                       ip.SERVICE_SCAN,
                                       service_data=data)
        await hass.async_block_till_done()

    state = hass.states.get(VALID_ENTITY_ID)
    assert state.state == 'birds'
    assert state.attributes.get(ip.ATTR_CONFIDENCE) == ip.DEFAULT_CONFIDENCE

    assert state.attributes.get(cb.ATTR_MODEL_ID) == MOCK_MODEL_ID
    assert (state.attributes.get(CONF_FRIENDLY_NAME) ==
            'classificationbox demo_camera 12345')

    assert len(classification_events) == 1
    assert classification_events[0].data[ATTR_ID] == 'birds'
    assert classification_events[0].data[ip.ATTR_CONFIDENCE] == 91.59
