### Bird classification project
Aim of project: automatically capture and classify images of birds visiting my bird-feeder. A long term goal of this project is to contribute the captured data to a nationwide study of bird populations.

Tools used:
* **Motion** for capturing motion triggered images
* **Classificationbox** for classifying bird/not-bird images
* **Home-Assistant** for coordinating the image capture/processing, storing data and sending notifications

This write-up will first present the image classification work using Classificationbox, then describe the practical implementation within an automated system.


### The problem definition
Being interested in bird watching, I attached a bird feeder to a window of my flat and within a few days various species of bird started visiting the feeder. I decided it would be fun to rig up a motion triggered camera to capture images of the birds, and I used Home-Assistant and a £10 USB webcam to capture images via motion trigger, and setup Home-Assistant to send an image notification to my phone when an image was captured. This setup is shown below:

<p align="center">
<img src="https://github.com/robmarkcole/HASS-Machinebox-Classificationbox/blob/master/bird_project/setup.png" width="700">
</p>

However I quickly discovered that all kinds of motion could trigger an image capture. The result was hundreds of images of all kinds of motion, such as planes flying in the distance or even funky light effects. Approximately less than half the images actually contained a bird, so I decided it was necessary to filter out the non-bird images. I have been interested in image classification for a while, and whilst searching online I came across [this article on Classificationbox](https://blog.machinebox.io/how-anyone-can-build-a-machine-learning-image-classifier-from-photos-on-your-hard-drive-very-5c20c6f2764f), which looked ideal for this project.


### Introduction to Classificationbox
[Classificationbox](https://machineboxio.com/docs/classificationbox) provides a ready-to-train classifier, deployed in a Docker container and exposed via a REST API. It uses [online learning](https://en.wikipedia.org/wiki/Online_machine_learning) to train a classifier that can be used to automatically classify various types of data, such as text, images, structured and unstructured data. The publishers of Classificationbox (a company called [Machinebox](https://machineboxio.com/), based in London, UK) advise that the accuracy of a Classificationbox classifier improves with the number and quality of images supplied, where accuracy is the percentage of images correctly classified. If you have less than 30 images of each class (bird/not-bird in this case, so 60 images total), you may achieve better accuracy using [Tagbox](https://machineboxio.com/docs/tagbox), which uses [one-shot learning](https://en.wikipedia.org/wiki/One-shot_learning). I initially experimented with Tagbox but found that in many cases it could not identify a bird in the images, presumably because the illumination in the images is poor and often the bird appears as a colourless silhouette. Luckily I had over 1000 images so could proceed to use Classificationbox.

Assuming you have [Docker installed](https://www.docker.com/community-edition), first download Classificationbox [from Dockerhub](https://hub.docker.com/r/machinebox/classificationbox/) by entering in the terminal:

```
sudo docker pull machinebox/classificationbox
```

Then run the container and expose on port **8080** with:
```
MB_KEY="INSERT-YOUR-KEY-HERE"
sudo docker run -p 8080:8080 -e "MB_KEY=$MB_KEY" machinebox/classificationbox
```

There are a number of ways you can interact with Classificationbox from the terminal, for example using [cURL](https://curl.haxx.se/), [HTTP](https://en.wikipedia.org/wiki/POST_(HTTP)) or python libraries such as [requests](http://docs.python-requests.org/en/master/). A useful website for converting cURL to requests is [curl.trillworks.com](https://curl.trillworks.com/).

To check that Classificationbox is running correctly using cURL, and assuming you are on the same machine that Classificationbox is running on (`localhost`), at the terminal enter:

```curl
curl http://localhost:8080/healthz
```
You should see a response similar to:

```
{
	"success": true,
	"hostname": "d6e51ge096c9",
	"metadata": {
		"boxname": "classificationbox",
		"build": "3ba550e"
	}
```

The if you don't get `"success": true` investigate the issue, otherwise you can proceed to training.

### Training Classificationbox
[This article](https://blog.machinebox.io/how-anyone-can-build-a-machine-learning-image-classifier-from-photos-on-your-hard-drive-very-5c20c6f2764f) explains the training of Classificationbox, and provides a GO script to perform training. However if you have difficulty getting GO installed on your system (it took me a few tries!) I've also published a training script in python [teach_classificationbox.py](https://github.com/robmarkcole/classificationbox_python). One advantage of the GO script is that it will print out the accuracy of your model, which is a feature I will add to the python script in time. Whichever script you use, the first step is to decide what and how many classes you want to identify. For this project I wanted two classes, bird/not-bird images, with examples shown below.

<p align="center">
<img src="https://github.com/robmarkcole/HASS-Machinebox-Classificationbox/blob/master/bird_project/bird_not_bird_examples.png" width="700">
</p>

I had a total of over 1000 images that I manually sorted in two folders of bird/not-bird images, with each folder containing 500 images (this number may well be excessive, and in future work on this project I will experiment on reducing this number, since its quite prohibitive to require so many images). Make sure that the images you use for training are representative of all the situations you will encounter in use, so for example if you were capturing images at day and night, you want your teaching images to also include images at day and night. With the images sorted into the two folders, I ran the  `teach_classificationbox.py` script to train Classificationbox. For 1000 images, teaching took about 30 minutes on my Macbook Pro with 8 Gb RAM, but obviously this time will vary depending on your project. In also re-ran the teaching with the GO script mentioned earlier and calculated that the model achieved 92% accuracy, pretty respectable! You will want to know the model ID, and can use cURL to check the ID:
```cURL
curl http://localhost:8080/classificationbox/models
```
This should return something like:
```
{
	"success": true,
	"models": [
		{
			"id": "5b0ce5d8023d4e35",
			"name": "5b0ce5d8023d4e35"
		}
	]
```
It is straightforward to download the model as a binary file, which you can then transfer to another machine. In my case I performed teaching on my Macbook but actually want to use the model on another machine (a Synology NAS) and again I can use cURL to upload the model file to that machine. To download the model file I used:

```cURL
curl http://localhost:8080/classificationbox/state/5b0ce5d8023d4e35 --output 5b0ce5d8023d4e35.classificationbox
```

You will want to replace my model ID (`5b0ce5d8023d4e35`) with your own. The downloaded file is 60 kb, so small enough to be shared on Github and other online hosts. This is useful if you want others to be able to reproduce your work. There is a variety of ways to post the model file to another instance of Classificationbox, for example if you want to use the model on another machine as I am doing. Search the Classificationbox docs for `Uploading state`. Personally I used pythons requests library to post the model file from my Macbook to a Synology using:

```python
import base64
import requests

MODEL_STATE_URL = 'http://{}:{}/classificationbox/state/{}'.format(IP, PORT, MODEL_ID)
filename = '/path/to/model/5b0ce5d8023d4e35.classificationbox'
with open(file_path, "rb") as f:
        file_data = base64.b64encode(f.read()).decode('ascii')
model_data  = {"base64": file_data}
requests.post(STATE_POST_URL, json=model_data).json()
```

You should see a response like:
```
{'success': True,
 'id': '5b0ce5d8023d4e35',
 'name': '5b0ce5d8023d4e35',
 'options': {},
 'predict_only': False}
 ```

### Using Classificationbox with Home-Assistant
There is not a cURL command we can use to perform a classification on an image using Classificationbox, so instead I have written code to use Classificationbox with Home-Assistant. Home-Assistant is an open source, python 3 home automation hub, and if you are reading this article then I assume you are familiar with it. If not I refer you to the documents online. Note that there are a couple of different ways to run Home-Assistant. In this project I am using the Hassio approach with you should [read about here](https://www.home-assistant.io/hassio/). However it doesn't matter how you have Home-Assistant running, this project will work with all approaches.

In this project we use Home-Assistant to post images from my motion triggered webcam to Classificationbox, then if a bird image is classified, we are sent a mobile phone notification with the image. A diagram of the system is shown below:

<p align="center">
<img src="https://github.com/robmarkcole/HASS-Machinebox-Classificationbox/blob/master/bird_project/system_overview.png" width="700">
</p>

#### Hardware
* **Webcam**: I picked up a [cheap webcam on Amazon](https://www.amazon.co.uk/gp/product/B000Q3VECE/ref=oh_aui_search_detailpage?ie=UTF8&psc=1). However you can use [any camera](https://www.home-assistant.io/components/#camera) that is compatible with Home-Assistant.
* **Pi 3**: I have the camera connected via USB to a raspberry pi 3 running Home-Assistant.
* **Synology NAS**: The pi 3 doesn't have sufficient RAM to run Classificationbox (2 GB min required) so instead I am running it on my [Synology DS216+II](https://www.amazon.co.uk/gp/product/B01G3HYR6G/ref=oh_aui_search_detailpage?ie=UTF8&psc=1) that I have [upgraded to have 8 GB RAM](http://blog.fedorov.com.au/2016/02/how-to-upgrade-memory-in-synology-ds216.html).
* **Bird feeder**: My mum bought this, but there are similar online, just search for `windown mounted birdfeeder`.

#### Home-Assistant Configuration
##### Classificationbox custom component
First of all you need to [get the code from here](https://github.com/robmarkcole/HASS-Machinebox-Classificationbox) to use Classificationbox with Home-Assistant. This code is called a `custom component` by Home-Assistant users, and it is added by placing the contents of the `custom_components` folder in your Home-Assistant configuration directory (or adding its contents to an existing custom_components folder). The `yaml` code-blocks that follow are all code to be entered in the Home-Assistant `configuration.yaml` file.

```yaml
image_processing:
  - platform: classificationbox
    ip_address: 192.168.0.18 # Change to the IP hosting Classificationbox
    port: 8080
    scan_interval: 100000
    source:
      - entity_id: camera.dummy
```
With the long `scan_interval` I am ensuring that image classification will only be performed when I trigger it. Next I configure the camera which will have the entity_id `camera.dummy`.


##### Motion detection with a USB camera
I have a cheap usb webcam that captures images on motion detection [using](https://community.home-assistant.io/t/usb-webcam-on-hassio/37297/7) the [motion](https://motion-project.github.io/) Hassio addon. The final view of the camera feed in Home-Assistant is shown below, with the live view camera image just cropped off the bottom of the image.

<p align="center">
<img src="https://github.com/robmarkcole/robins-hassio-config/blob/master/images/HA_motion_camera_view.png" width="500">
</p>

I've configured the motion add-on via its Hassio tab with the following settings:

```yaml
{
  "config": "",
  "videodevice": "/dev/video0",
  "input": 0,
  "width": 640,
  "height": 480,
  "framerate": 2,
  "text_right": "%Y-%m-%d %T-%q",
  "target_dir": "/share/motion",
  "snapshot_interval": 1,
  "snapshot_name": "latest",
  "picture_output": "best",
  "picture_name": "%v-%Y_%m_%d_%H_%M_%S-motion-capture",
  "webcontrol_local": "on",
  "webcontrol_html": "on"
}
```
This setup captures an image every second, saved as `latest.jpg`, and is over-written every second. Additionally, on motion detection a time-stamped image is captured with format `%v-%Y_%m_%d_%H_%M_%S-motion-capture.jpg`.

The image `latest.jpg` is displayed on the HA front-end using a [local-file camera](https://home-assistant.io/components/camera.local_file/). I will also display the last time-stamped image with a second `local_file` camera. **Note** that the image files (here `latest.jpg` and `dummy.jpg`) must be present when Home-Assistant starts as the component makes a check that the file exists, and therefore if running for the first time just copy some appropriately named images into the `/share/motion` folder. In `configuration.yaml`:

```yaml
camera:
  - platform: local_file
    file_path: /share/motion/latest.jpg
    name: "Live view"
  - platform: local_file
    file_path: /share/motion/dummy.jpg
    name: "dummy"
```
I use the [folder_watcher component](https://www.home-assistant.io/components/folder_watcher/) to detect when new time-stamped images are saved to disk on the raspberry Pi:

```yaml
folder_watcher:
  - folder: /share/motion
    patterns:
      - '*capture.jpg'
```

The `folder_watcher` fires an event with data including the image path to the added file. I use an automation to display the new image on the `dummy` camera using the `camera.update_file_path` service:

```yaml
- action:
    data_template:
      file_path: ' {{ trigger.event.data.path }} '
    entity_id: camera.dummy
    service: camera.local_file_update_file_path
  alias: Display new image
  condition: []
  id: '1520092824633'
  trigger:
  - event_data:
      event_type: created
    event_type: folder_watcher
    platform: event
```

I use a template sensor (in `sensors.yaml`) to break out the new file path:
```yaml
- platform: template
  sensors:
    last_added_file:
      friendly_name: Last added file
      value_template: "{{states.camera.dummy.attributes.file_path}}"
```
I use an automation triggered by the state change of the template sensor to trigger the `image_processing.scan` service which sends the new image for classification by Classificationbox:

```yaml
- id: '1527837198169'
  alias: Perform image classification
  trigger:
  - entity_id: sensor.last_added_file
    platform: state
  condition: []
  action:
  - data:
      entity_id: camera.dummy
    service: image_processing.scan
```

Finally I use the event fired by the image classification to trigger an automation to send me the new image and classification as a [Pushbullet](https://www.pushbullet.com/) notification:

```yaml
- action:
  - data_template:
      message: Class {{ trigger.event.data.class_id }} with probability {{ trigger.event.data.score }}
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
<img src="https://github.com/robmarkcole/robins-hassio-config/blob/master/images/iphone_notification.jpeg" width="300">
</p>

A photo of my birdfeeder setup is shown below.

<p align="center">
<img src="https://github.com/robmarkcole/robins-hassio-config/blob/master/images/camera_setup.jpg" width="500">
</p>

### Summary
In summary this write-up has described how to create an image classifier using Classificationbox, and how to deploy it for use with Home-Assistant. A cheap webcam is used to capture motion triggered images, which are posted to Classificationbox, and if there are birds in the image then the image is sent to my phone as a notification. Future work on this project is to train the classifier to identify different species of birds arriving at the bird feeder. One slight issue I have is that a magpi has been trying to rip the feeder off the window, so I need to do some work to make it magpi proof! I hope this project inspires you to try out using image classifiers in your projects.

### Links
* Classificationbox: https://machineboxio.com/docs/classificationbox) provides
* Home-Assistant: https://www.home-assistant.io/
* Hassio: https://www.home-assistant.io/hassio/
* Docker: https://www.docker.com/community-edition
* Classificationbox custom component for Home-Assistant: https://github.com/robmarkcole/HASS-Machinebox-Classificationbox
* Pushbullet: https://www.pushbullet.com/
