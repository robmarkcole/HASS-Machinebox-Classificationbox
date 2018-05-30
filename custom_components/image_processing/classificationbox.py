"""
Component that will perform classification of images via classiifcationbox.

For more details about this platform, please refer to the documentation at
https://home-assistant.io/components/image_processing.classificationbox
"""
import base64
import logging
from urllib.parse import urljoin

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
    """Encode an image to a base64 string."""
    img_str = base64.b64encode(image).decode('ascii')
    return img_str


def get_classes(classes_json):
    """Return the classes data."""
    return {class_result['id']: round(class_result['score'], 2)
            for class_result in classes_json}


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
                    model['name'],
                ))
    add_devices(entities)


class ClassificationboxEntity(ImageProcessingEntity):
    """Perform an image classification."""

    def __init__(self, ip, port, camera_entity,
                 model_id, model_name):
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
        self._prediction = {}

    def process_image(self, image):
        """Process an image."""
        predict_url = urljoin(
            self._base_url, "models/{}/predict".format(self._model_id))

        input_json = {
            "inputs": [{
                "key": "image",
                "type": "image_base64",
                "value": encode_image(image)}
                       ]}

        response = {}
        try:
            response = requests.post(
                predict_url,
                json=input_json,
                timeout=9
                ).json()
        except requests.exceptions.ConnectionError:
            _LOGGER.error("ConnectionError: Is %s running?", CLASSIFIER)
            response['success'] = False

        if response['success']:
            self._state = response['classes'][0]['id']
            self._prediction = get_classes(response['classes'])
        else:
            self._state = None
            self._prediction = {}

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
        attr = {
            'model_id': self._model_id,
            'model_name': self._model_name
            }
        attr.update(self._prediction)
        return attr
