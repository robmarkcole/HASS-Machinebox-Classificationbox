"""
Component that will perform classification of images via classiifcationbox.

For more details about this platform, please refer to the documentation at
https://home-assistant.io/components/image_processing.classificationbox
"""
import base64
import logging

import requests
import voluptuous as vol

from homeassistant.core import split_entity_id
import homeassistant.helpers.config_validation as cv
from homeassistant.components.image_processing import (
    PLATFORM_SCHEMA, ImageProcessingEntity, CONF_SOURCE, CONF_ENTITY_ID,
    CONF_NAME)
from homeassistant.const import (CONF_IP_ADDRESS, CONF_PORT)

_LOGGER = logging.getLogger(__name__)

CLASSIFIER = 'classificationbox'

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend({
    vol.Required(CONF_IP_ADDRESS): cv.string,
    vol.Required(CONF_PORT): cv.port,
})


def encode_image(image):
    """base64 encode an image stream."""
    base64_img = base64.b64encode(image).decode('ascii')
    return {"base64": base64_img}


def setup_platform(hass, config, add_devices, discovery_info=None):
    """Set up the classifier."""
    entities = []
    IP = config[CONF_IP_ADDRESS]
    PORT = config[CONF_PORT]
    MODELS_LIST_URL = 'http://{}:{}/{}/models'.format(IP, PORT, CLASSIFIER)

    models_query = requests.get(MODELS_LIST_URL).json()  # Check for models.
    if models_query['success']:
        for model in models_query['models']:
            for camera in config[CONF_SOURCE]:
                entities.append(ClassificationboxEntity(
                    config[CONF_IP_ADDRESS],
                    config[CONF_PORT],
                    camera[CONF_ENTITY_ID],
                    model['id'],
                    model['name']
                ))
    add_devices(entities)


class ClassificationboxEntity(ImageProcessingEntity):
    """Perform an image classification."""

    def __init__(self, ip, port, camera_entity, model_id, model_name):
        """Init with the camera and model info."""
        super().__init__()
        self._base_url = "http://{}:{}/{}/".format(ip, port, CLASSIFIER)
        self._camera = camera_entity
        self._model_id = model_id
        self._model_name = model_name
        camera_name = split_entity_id(camera_entity)[1]
        self._name = "{} {} {}".format(
            CLASSIFIER, camera_name, model_name)
        self._state = None

    def process_image(self, image):
        """Process an image."""
        response = {}
        try:
            response = requests.post(
                self._url,
                json=encode_image(image),
                timeout=9
                ).json()
        except requests.exceptions.ConnectionError:
            _LOGGER.error("ConnectionError: Is %s running?", CLASSIFIER)
            response['success'] = False

        if response['success']:
            self._state = True
        else:
            self._state = None

    @property
    def camera_entity(self):
        """Return camera entity id from process pictures."""
        return self._camera

    @property
    def name(self):
        """Return the name of the sensor."""
        return self._name

    @property
    def state(self):
        """Return the state of the entity."""
        return self._state

    @property
    def device_state_attributes(self):
        """Return the classifier attributes."""
        return {
            'model_id': self._model_id,
            'model_name': self._model_name
            }
