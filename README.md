Home-Assistant image classification using [Classificationbox](https://machinebox.io/docs/classificationbox). Follow [this guide](https://blog.machinebox.io/how-anyone-can-build-a-machine-learning-image-classifier-from-photos-on-your-hard-drive-very-5c20c6f2764f ) to create a model/models on Classificationbox. This component adds an `image_processing` entity for each model you have created on Classificationbox, where the state of the entity is the most likely classification of an image using that model. An event is fired when the confidence in classification is above the threshold set by `confidence`, which defaults to 80%.

Place the `custom_components` folder in your configuration directory (or add its contents to an existing custom_components folder).

Add to your HA config:
```yaml
image_processing:
  - platform: classificationbox
    ip_address: localhost
    port: 8080
    confidence: 50
    source:
      - entity_id: camera.local_file
```

Configuration variables:
- **ip_address**: the ip of your Tagbox instance
- **port**: the port of your Tagbox instance
- **confidence** (Optional): The minimum of confidence in percent to fire an event. Defaults to 80.
- **source**: must be a camera.

## Automations using events

Events can be used as a trigger for automations, and in the example automation below are used to trigger a notification with the image and the classification:

```yaml
- action:
  - data_template:
      message: 'Class {{ trigger.event.data.class_id }} with probability {{ trigger.event.data.score }}'
      title: New image classified
      data:
        file: ' {{states.camera.dummy.attributes.file_path}} '
    service: notify.pushbullet
  alias: Send classification
  condition: []
  id: '1120092824611'
  trigger:
  - event_data:
      event_type: image_classification
    event_type: image_processing
    platform: event
```

<p align="center">
<img src="https://github.com/robmarkcole/HASS-Machinebox-Classificationbox/blob/master/usage.png" width="650">
</p>

### Classificationbox
Get/update Classificationbox [from Dockerhub](https://hub.docker.com/r/machinebox/classificationbox/) by running:
```
sudo docker pull machinebox/classificationbox
```

Run the container with:
```
MB_KEY="INSERT-YOUR-KEY-HERE"
sudo docker run -p 8080:8080 -e "MB_KEY=$MB_KEY" machinebox/classificationbox
```

Classification box is trained on https://storage.googleapis.com/openimages/web/index.html

#### Limiting computation
[Image-classifier components](https://www.home-assistant.io/components/image_processing/) process the image from a camera at a fixed period given by the `scan_interval`. This leads to excessive computation if the image on the camera hasn't changed (for example if you are using a [local file camera](https://www.home-assistant.io/components/camera.local_file/) to display an image captured by a motion triggered system and this doesn't change often). The default `scan_interval` [is 10 seconds](https://github.com/home-assistant/home-assistant/blob/98e4d514a5130b747112cc0788fc2ef1d8e687c9/homeassistant/components/image_processing/__init__.py#L27). You can override this by adding to your config `scan_interval: 10000` (setting the interval to 10,000 seconds), and then call the `scan` [service](https://github.com/home-assistant/home-assistant/blob/98e4d514a5130b747112cc0788fc2ef1d8e687c9/homeassistant/components/image_processing/__init__.py#L62) when you actually want to process a camera image. So in my setup, I use an automation to call `scan` when a new motion triggered image has been saved and displayed on my local file camera.


## Local file camera
Note that for development I am using a [file camera](https://www.home-assistant.io/components/camera.local_file/).
```yaml
camera:
  - platform: local_file
    file_path: /images/bird.jpg
```
