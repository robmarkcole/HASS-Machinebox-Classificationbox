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
<img src="https://github.com/robmarkcole/HASS-Machinebox-Classificationbox/blob/master/bird_project/setup.png" width="800">
</p>

However I quickly discovered that all kinds of motion could trigger an image capture. The result was hundreds of images of all kinds of motion, such as planes flying in the distance or even funky light effects. Approximately less than half the images actually contained a bird, so I decided it was necessary to filter out the non-bird images. I have been interested in image classification for a while, and whilst searching online I came across [this article on Classificationbox](https://blog.machinebox.io/how-anyone-can-build-a-machine-learning-image-classifier-from-photos-on-your-hard-drive-very-5c20c6f2764f), which looked ideal for this project.


### Introduction to Classificationbox
[Classificationbox](https://machineboxio.com/docs/classificationbox) provides a ready-to-train classifier, deployed in a Docker container and exposed via a REST API. It uses [online learning](https://en.wikipedia.org/wiki/Online_machine_learning) to train a classifier that can be used to automatically classify various types of data, such as text, images, structured and unstructured data. The publishers of Classificationbox (a company called [Machinebox](https://machineboxio.com/), based in London, UK) advise that the accuracy of a Classificationbox classifier improves with the number and quality of images supplied, where accuracy is the percentage of images correctly classified. If you have less than 30 images of each class (bird/not-bird in this case, so 60 images total), you may achieve better accuracy using [Tagbox](https://machineboxio.com/docs/tagbox), which uses [one-shot learning](https://en.wikipedia.org/wiki/One-shot_learning). I initially experimented with Tagbox but found that in many cases it could not identify a bird in the images, presumably because the illumination in the images is poor and often the bird appears as a colourless silhouette. Luckily I had over 1000 images so could proceed to use Classificationbox.

Assuming you have [Docker installed](https://www.docker.com/community-edition), first download Classificationbox [from Dockerhub](https://hub.docker.com/r/machinebox/classificationbox/) by entering in the terminal:

```
sudo docker pull machinebox/classificationbox
```

Then run the container and expose on port `8080` with:
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
<img src="https://github.com/robmarkcole/HASS-Machinebox-Classificationbox/blob/master/bird_project/bird_not_bird_examples.png" width="800">
</p>

I had a total of over 1000 images that I manually sorted in two folders of bird/not-bird images, with each folder containing 500 images (this number may well be excessive, and in future work on this project I will experiment on reducing this number, since its quite prohibitive to require so many images). Make sure that the images you use for training are representative of all the situations you will encounter in use, so for example if you were capturing images at day and night, you want your teaching images to also include images at day and night. With the images sorted into the two folders, I ran the  `teach_classificationbox.py` script to train Classificationbox. For 1000 images, teaching took about 30 minutes on my Macbook Pro with 8 GB RAM, but obviously this time will vary depending on your project. In also re-ran the teaching with the GO script mentioned earlier and calculated that the model achieved 92% accuracy, pretty respectable! You will want to know the model ID, and can use cURL to check the ID:
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
Now that the model is created we can use another cURL command to perform a classification on an image `bird.jpg`. I enter:
```
export FOO=`base64 -in /absolute/path/to/bird.jpg`

curl -X POST -H "Content-Type: application/json" -d  '{ "inputs": [ {"type": "image_base64", "key": "image", "value": "'$FOO'" } ] }' http://localhost:8080/classificationbox/models/5b0ce5d8023d4e35/predict
```
I then see:
```
{
	"success": true,
	"classes": [
		{
			"id": "birds",
			"score": 0.915892
		},
		{
			"id": "not_birds",
			"score": 0.084108
		}
	]
```

Now that we have confirmed the model is performing correctly, we can download the model as a binary file. This is important if you are on the free tier of Machinebox as the model will be deleted every time you restart the Docker container. Once we have the model binary file we can upload it after restarting the Docker container, or transfer it another machine. In my case I performed teaching on my Macbook but actually want to use the model in production on a Synology NAS. To download the model file I used:

```cURL
curl http://localhost:8080/classificationbox/state/5b0ce5d8023d4e35 --output 5b0ce5d8023d4e35.classificationbox
```

You will want to replace my model ID (`MODEL_ID 5b0ce5d8023d4e35`) with your own. The downloaded file is 60 kb, so small enough to be shared on Github and other online hosts. This is useful if you want others to be able to reproduce your work. There is a variety of ways to post the model file to another instance of Classificationbox, for example if you want to use the model on another machine as I am doing. Search the Classificationbox docs for `Uploading state`. Personally I used the python requests library to post the model file from my Macbook to my  Synology using:

```python
import base64
import requests

MODEL_STATE_URL = 'http://{}:{}/classificationbox/state/{}'.format(IP, PORT, MODEL_ID)
filename = '/path/to/model/5b0ce5d8023d4e35.classificationbox'
with open(file_path, "rb") as f:
        file_data = base64.b64encode(f.read()).decode('ascii')
model_data  = {"base64": file_data}
requests.post(STATE_POST_URL, json=model_data)
```

Replace the `IP` and `PORT` with those of your target machine, which for my Synology was `IP 192.168.0.18` You should see a response like:
```
{'success': True,
 'id': '5b0ce5d8023d4e35',
 'name': '5b0ce5d8023d4e35',
 'options': {},
 'predict_only': False}
 ```

### Using Classificationbox with Home-Assistant
I have written code to use Classificationbox with Home-Assistant. Home-Assistant is an open source, python 3 home automation hub, and if you are reading this article then I assume you are familiar with it. If not I refer you to the [documents online](https://www.home-assistant.io/). Note that there are a couple of different ways to run Home-Assistant. In this project I am using the Hassio approach which you should [read about here](https://www.home-assistant.io/hassio/), running on a Raspberry Pi 3. However it doesn't matter how you have Home-Assistant running, this project should work with all common approaches.

In this project we use Home-Assistant to post images from my motion triggered usb camera to Classificationbox, then if a bird image is classified, we are sent a mobile phone notification with the image. A diagram of the system is shown below:

<p align="center">
<img src="https://github.com/robmarkcole/HASS-Machinebox-Classificationbox/blob/master/bird_project/system_overview.png" width="900">
</p>

#### Hardware
* **Webcam**: I picked up a [cheap webcam on Amazon](https://www.amazon.co.uk/gp/product/B000Q3VECE/ref=oh_aui_search_detailpage?ie=UTF8&psc=1). However you can use [any camera](https://www.home-assistant.io/components/#camera) that is compatible with Home-Assistant.
* **Pi 3**: I have the camera connected via USB to a raspberry pi 3 running Home-Assistant.
* **Synology NAS**: The pi 3 doesn't have sufficient RAM to run Classificationbox (2 GB min required) so instead I am running it on my [Synology DS216+II](https://www.amazon.co.uk/gp/product/B01G3HYR6G/ref=oh_aui_search_detailpage?ie=UTF8&psc=1) that I have [upgraded to have 8 GB RAM](http://blog.fedorov.com.au/2016/02/how-to-upgrade-memory-in-synology-ds216.html).
* **Bird feeder**: My mum bought this, but there are similar online, just search for `windown mounted birdfeeder`.

#### Motion triggered image capture via Motion addon
I connected the usb webcam to the raspberry pi and pointed the webcam at the birdfeeder. I have a number of options for viewing the camera feed in Home-Assistant, but since I am using Hassio and want motion detection, I decided to try out an approach which uses the [Motion](https://motion-project.github.io/) software under the hood. When using Hassio it is straightforward to extend the functionality of Home-Assistant by installing ['Hassio addons'](https://www.home-assistant.io/addons/), and these addons are installed via a page on the Home-Assistant interface, shown below:

<p align="center">
<img src="https://github.com/robmarkcole/HASS-Machinebox-Classificationbox/blob/master/bird_project/hassio_addons.png" width="900">
</p>

The addon I am using is written by [@HerrHofrat](https://github.com/HerrHofrat) and is called `Motion`, and it is available at https://github.com/HerrHofrat/hassio-addons/tree/master/motion. You will need to add this repository as a location accessible to Hassio (search for the box that states `Add new repository by URL`). This addon will both continually capture still images, and capture timestamped images when motion is detected. I experimented with the addon settings but settled on the configuration below. The addon is configured by the Hassio tab for the addon:

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

The addon captures an image every second, saved as `latest.jpg`, and this image is continually over-written. On motion detection a timestamped image is captured with format `%v-%Y_%m_%d_%H_%M_%S-motion-capture.jpg`. All images are saved to the `/share/motion` folder on the Pi. These defaults should work regardless of the usb camera you are using, but if you have several usb cameras attached to your pi you may need to use the terminal to check the camera connection (here `/dev/video0`).

#### Displaying images on Home-Assistant
I display the images captured by the addon using a pair of [local-file cameras](https://home-assistant.io/components/camera.local_file/).
The continually updated `latest.jpg` is displayed on a camera with the name `Live view` and the most recent timestamped image captured will be displayed on a camera called `dummy`. The configuration for both cameras is added to `configuration.yaml`, shown below:

```yaml
camera:
  - platform: local_file
    file_path: /share/motion/latest.jpg
    name: "Live view"
  - platform: local_file
    file_path: /share/motion/dummy.jpg
    name: "dummy"
```
**Note** that the image files (here `latest.jpg` and `dummy.jpg`) must be present when Home-Assistant starts as the component makes a check that the file exists, and therefore if running for the first time just copy some appropriately named images into the `/share/motion` folder. In `configuration.yaml`:

The final view of the camera feed in Home-Assistant is shown below.

<p align="center">
<img src="https://github.com/robmarkcole/HASS-Machinebox-Classificationbox/blob/master/bird_project/HA_motion_camera_view.png" width="400">
</p>

#### Classificationbox custom component
To make Classificationbox accessible to Home-Assistant you will first need to get the Classificationbox custom component code from https://github.com/robmarkcole/HASS-Machinebox-Classificationbox. This code is added to Home-Assistant by placing the contents of the `custom_components` folder in your Home-Assistant configuration directory (or adding its contents to an existing custom_components folder). The yaml code-blocks that follow are all code to be entered in the Home-Assistant `configuration.yaml` file.

```yaml
image_processing:
  - platform: classificationbox
    ip_address: 192.168.0.18 # Change to the IP hosting Classificationbox
    port: 8080
    scan_interval: 100000
    source:
      - entity_id: camera.dummy
```
Not that by default the image will be classified every 10 seconds, but with the long `scan_interval` I am ensuring that image classification will only be performed when I trigger it using the `image_processing.scan` service described later. Note that the source is `camera.dummy`, which will be the motion triggered image.

#### Tying it all together
Now that image capture is configured and Classificationbox is available to use, we must link them together using a sequence of automations in Home-Assistant. The sequence that we setup is illustrated in the diagram below, where the blue arrows represent automations:

<p align="center">
<img src="https://github.com/robmarkcole/HASS-Machinebox-Classificationbox/blob/master/bird_project/sequence.png" width="300">
</p>

Out of the box, Home-Assistant has no knowledge of when the Motion addon captures a new motion triggered image, so I use the [folder_watcher component](https://www.home-assistant.io/components/folder_watcher/) to alert Home-Assistant to new images in the `/share/motion` directory. The configuration of folder_watcher in `configuration.yaml` is:

```yaml
folder_watcher:
  - folder: /share/motion
    patterns:
      - '*capture.jpg'
```

The `folder_watcher` component fires an event every time a new timestamped image is saved in `/share/motion` when the file name matches the pattern `*capture.jpg` (as the timestamped images file names do).  The `folder_watcher` event data including the file name and path to the added image. I use an automation to display the new image on the `camera.dummy` using the `camera.update_file_path` service. The configuration for the automation is shown below, added to `automations.yaml`:

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

I use a [template sensor](https://www.home-assistant.io/components/sensor.template/) to display the new image file path, which is available as an attribute on `camera.dummy`. The template sensor is configured in `sensors.yaml`:

```yaml
- platform: template
  sensors:
    last_added_file:
      friendly_name: Last added file
      value_template: "{{states.camera.dummy.attributes.file_path}}"
```

I now use an automation to trigger the `image_processing.scan` service on `camera.dummy`. The `scan` service instructs Home-Assistant to send the image displayed by `camera.dummy` for classification by Classificationbox, and this automation is triggered by the state change of the `file_path` sensor. I add to `automations.yaml`:

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

Finally I use the `image_classification` event fired by the Classificationbox component to trigger an automation to send me the image and classification as a [Pushbullet](https://www.pushbullet.com/) notification:

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

TO DO - ADD CONDITION TO THIS AUTOMATION. UPDATE CLASSIFICATIONBOX COMPONENT AND ADD VERSION TO THIS ARTICLE.

The notification is shown below.

<p align="center">
<img src="https://github.com/robmarkcole/HASS-Machinebox-Classificationbox/blob/master/bird_project/iphone_notification.jpeg" width="400">
</p>

### Summary
In summary this write-up has described how to create an image classifier using Classificationbox, and how to deploy it for use with Home-Assistant. A cheap webcam is used to capture motion triggered images, which are posted to Classificationbox, and if there are birds in the image then the image is sent to my phone as a notification. Future work on this project is to train the classifier to identify different species of birds arriving at the bird feeder. One slight issue I have is that a magpi has been trying to rip the feeder off the window (shown below), so I need to do some work to make it magpi proof (magpi desctruction efforts shown below)! I hope this project inspires you to try out using image classifiers in your projects.

<p align="center">
<img src="https://github.com/robmarkcole/HASS-Machinebox-Classificationbox/blob/master/bird_project/magpi.png" width="900">
</p>

### Links
* Classificationbox: https://machineboxio.com/docs/classificationbox) provides
* Home-Assistant: https://www.home-assistant.io/
* Hassio: https://www.home-assistant.io/hassio/
* Hassio Motion addon: https://github.com/HerrHofrat/hassio-addons/tree/master/motion
* Docker: https://www.docker.com/community-edition
* Classificationbox custom component for Home-Assistant: https://github.com/robmarkcole/HASS-Machinebox-Classificationbox
* Pushbullet: https://www.pushbullet.com/
