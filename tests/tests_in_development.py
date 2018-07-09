async def test_setup_platform(hass):
    """Setup platform with one entity."""
    with requests_mock.Mocker() as mock_req:
        url = 'http://{}:{}/{}/models'.format(
            MOCK_IP,
            MOCK_PORT,
            cb.CLASSIFIER)
        mock_req.post(url, json=MOCK_MODELS)
        await async_setup_component(hass, ip.DOMAIN, VALID_CONFIG)
        await hass.async_block_till_done()
        assert hass.states.get(VALID_ENTITY_ID)


async def test_process_image(hass, mock_image):
    """Test processing of an image."""
    await async_setup_component(hass, ip.DOMAIN, VALID_CONFIG)
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
    assert state.state == 'bird'
    assert state.attributes.get(ip.ATTR_CONFIDENCE) == ip.DEFAULT_CONFIDENCE

    assert state.attributes.get(cb.ATTR_MODEL_ID) == MOCK_MODEL_ID
    assert (state.attributes.get(CONF_FRIENDLY_NAME) ==
            'classificationbox demo_camera 12345')

    assert len(classification_events) == 1
    assert classification_events[0].data[ATTR_ID] == 'birds'
    assert classification_events[0].data[ip.ATTR_CONFIDENCE] == 91.59


async def test_setup_platform_with_name(hass):
    """Setup platform with one entity and a name."""
    MOCK_NAME = 'mock_name'
    NAMED_ENTITY_ID = 'image_processing.{}'.format(MOCK_NAME)

    VALID_CONFIG_NAMED = VALID_CONFIG.copy()
    VALID_CONFIG_NAMED[ip.DOMAIN][ip.CONF_SOURCE][ip.CONF_NAME] = MOCK_NAME

    await async_setup_component(hass, ip.DOMAIN, VALID_CONFIG_NAMED)
    assert hass.states.get(NAMED_ENTITY_ID)
    state = hass.states.get(NAMED_ENTITY_ID)
    assert state.attributes.get(CONF_FRIENDLY_NAME) == MOCK_NAME
