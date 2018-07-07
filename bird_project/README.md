### Bird classification project
**Summary of project:** automatically capture and classify images of birds visiting my bird-feeder. A long term goal of this project is to contribute the data to a nationwide study of bird populations.

Being interested in bird watching, I attached a bird feeder to a window of my flat and within a few days various species of bird started visiting the feeder. I decided it would be fun to rig up a motion triggered camera to capture images of the birds, and I used Home-Assistant and a £10 USB webcam to capture images via motion trigger, and setup Home-Assistant to send an image notification to my phone when an image was captured. This setup is shown below:

<p align="center">
<img src="https://github.com/robmarkcole/HASS-Machinebox-Classificationbox/blob/master/bird_project/setup.png" width="800">
</p>

However I quickly discovered that all kinds of motion could trigger an image capture. The result was hundreds of images of all kinds of motion, such as planes flying in the distance or even funky light effects. Approximately less than half the images actually contained a bird, so I decided it was necessary to filter out the non-bird images. I have been interested in image classification for a while, and whilst searching online I came across [this article on Classificationbox](https://blog.machinebox.io/how-anyone-can-build-a-machine-learning-image-classifier-from-photos-on-your-hard-drive-very-5c20c6f2764f), which looked ideal for this project.

This write-up will first present the image classification work using Classificationbox, then describe the practical implementation within an automated system.

Tools used:
* **Motion** software for capturing motion triggered images
* **Classificationbox** deep learning classifier for classifying bird/not-bird images
* **Home-Assistant** software for automated image capture and performing classification, recording data and sending notifications


### Introduction to Classificationbox
[Classificationbox](https://machineboxio.com/docs/classificationbox) provides a ready-to-train deep-learning classifier, deployed in a Docker container and exposed via a REST API. It uses [online learning](https://en.wikipedia.org/wiki/Online_machine_learning) to train a classifier that can be used to automatically classify various types of data, such as text, images, structured and unstructured data. The publishers of Classificationbox are a company called [Machinebox](https://machineboxio.com/), based in London, UK. Their docs advise that the accuracy of a Classificationbox classifier improves with the number and quality of images supplied, where accuracy is the percentage of images correctly classified. If you have less than 30 images of each class (bird/not-bird in this case, so 60 images total), you may achieve better accuracy using their alternative product called [Tagbox](https://machineboxio.com/docs/tagbox), which uses [one-shot learning](https://en.wikipedia.org/wiki/One-shot_learning). I initially experimented with Tagbox but found that in many cases it could not identify a bird in the images since the illumination in the images is poor and often the bird appears as a colourless silhouette. After a few weeks of image captures I had over 1000 images of bird/not-bird so could proceed to use Classificationbox.

Assuming you have [Docker installed](https://www.docker.com/community-edition), first download Classificationbox [from Dockerhub](https://hub.docker.com/r/machinebox/classificationbox/) by entering in the terminal:

```
sudo docker pull machinebox/classificationbox
```

Then run the Classificationbox container and expose it on port `8080` with:
```
MB_KEY="INSERT-YOUR-KEY-HERE"
sudo docker run -p 8080:8080 -e "MB_KEY=$MB_KEY" machinebox/classificationbox
```

There are a number of ways you can interact with Classificationbox from the terminal, for example using [cURL](https://curl.haxx.se/), [HTTP](https://en.wikipedia.org/wiki/POST_(HTTP)) or python libraries such as [requests](http://docs.python-requests.org/en/master/). To check that Classificationbox is running correctly using cURL, and assuming you are on the same machine that Classificationbox is running on (`localhost`), at the terminal enter:

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
[This article](https://blog.machinebox.io/how-anyone-can-build-a-machine-learning-image-classifier-from-photos-on-your-hard-drive-very-5c20c6f2764f) explains the training of Classificationbox, and links to a GO script which can be used to perform training. However if you have difficulty getting GO installed on your system (it took me a few tries!) I've also published a training script in python [teach_classificationbox.py](https://github.com/robmarkcole/classificationbox_python). One advantage of the GO script is that it will print out the accuracy of your model, which is a feature I will add to the python script in time. Whichever script you use, the first step is to decide on the classes you want to identify. For this project I wanted two classes, bird/not-bird images, with examples shown below.

<p align="center">
<img src="https://github.com/robmarkcole/HASS-Machinebox-Classificationbox/blob/master/bird_project/bird_not_bird_examples.png" width="800">
</p>

I had a total of over 1000 images that I manually sorted in two folders of bird/not-bird images, with each folder containing 500 images (this number may well be excessive, and in future work on this project I will experiment on reducing this number). Make sure that the images you use for training are representative of all the situations Classificationbox will encounter in use. For example, if you are capturing images at day and night, you want your teaching image set to also include images at day and night. With the images sorted into the two folders, I ran the GO script mentioned earlier and calculated that the model achieved 92% accuracy, pretty respectable! For 1000 images, teaching took about 30 minutes on my Macbook Pro with 8 GB RAM, but this will vary depending on you image set and hardware.

Classificationbox is capable of hosting multiple models, and you will want to know the model ID of the model you just created. You can use cURL to check the ID:
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
Now that the model is created we can use another cURL command to perform a classification on an test image `bird.jpg`. I enter:
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

Now that we have confirmed the model is performing correctly, we can download the model as a binary file. This is important if you are on the [free tier of Machinebox](https://machineboxio.com/#pricing) as the model will be erased every time you restart the Docker container. Once we have the model file we can upload it after restarting the Docker container, or transfer it another machine. In my case I performed teaching on my Macbook but actually want to use the model in production on a Synology NAS. To download the model file I used:

```cURL
curl http://localhost:8080/classificationbox/state/5b0ce5d8023d4e35 --output 5b0ce5d8023d4e35.classificationbox
```

You will want to replace my model ID (`5b0ce5d8023d4e35`) with your own. Note that just heading to the URL above in your browser will also download the file. The downloaded file is 60 kb, so small enough to be shared on Github and elsewhere online. This is useful if you want others to be able to reproduce your work.

To post the model file to Classificationbox use the cURL:
```
curl -X POST -F 'file=@/path/to/file/5b0ce5d8023d4e35.classificationbox' http://localhost:8080/classificationbox/state
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
Home-Assistant is an open source, python 3 home automation hub, and if you are reading this article then I assume you are familiar with it. If not I refer you to the [documents online](https://www.home-assistant.io/). Note that there are a couple of different ways to run Home-Assistant. In this project I am using the Hassio approach which you should [read about here](https://www.home-assistant.io/hassio/), running on a Raspberry Pi 3, and a home-Assistant version newer than 0.70. However it doesn't matter how you have Home-Assistant running, this project should work with all common approaches.

I have written code to use Classificationbox with Home-Assistant, and in this project we use this code with Home-Assistant to post images from my motion triggered USB camera to Classificationbox. If a bird image is classified, we are sent a mobile phone notification with the image. A diagram of the system is shown below:

<p align="center">
<img src="https://github.com/robmarkcole/HASS-Machinebox-Classificationbox/blob/master/bird_project/system_overview.png" width="900">
</p>

#### Hardware
* **Webcam**: I picked up a [cheap webcam on Amazon](https://www.amazon.co.uk/gp/product/B000Q3VECE/ref=oh_aui_search_detailpage?ie=UTF8&psc=1). However you can use [any camera](https://www.home-assistant.io/components/#camera) that is compatible with Home-Assistant.
* **Pi 3**: I have the camera connected via USB to a raspberry pi 3 running Home-Assistant.
* **Synology NAS**: The Raspberry Pi 3 doesn't have sufficient RAM to run Classificationbox (2 GB min required) so instead I am running it on my [Synology DS216+II](https://www.amazon.co.uk/gp/product/B01G3HYR6G/ref=oh_aui_search_detailpage?ie=UTF8&psc=1) that I have [upgraded to have 8 GB RAM](http://blog.fedorov.com.au/2016/02/how-to-upgrade-memory-in-synology-ds216.html). Alternatively you could use a spare laptop, or host Classificationbox on a cloud service such as [Google Cloud](https://blog.machinebox.io/deploy-docker-containers-in-google-cloud-platform-4b921c77476b).
* **Bird feeder**: My mum bought this, but there are similar online, just search for `window mounted birdfeeder`.

#### Motion triggered image capture via Motion addon
I connected the USB webcam to the Raspberry Pi running Home-Assistant and pointed the webcam at the birdfeeder. There are a number of options for viewing the camera feed in Home-Assistant, but since I am using Hassio and want motion detection, I decided to try out an approach which uses the [Motion](https://motion-project.github.io/) software deployed as a Hassio addon. [Hassio addons](https://www.home-assistant.io/addons/) are straightforward way to extend the functionality of Home-Assistant, and are installed via a page on the Home-Assistant interface, shown below:

<p align="center">
<img src="https://github.com/robmarkcole/HASS-Machinebox-Classificationbox/blob/master/bird_project/hassio_addons.png" width="900">
</p>

The addon I am using is written by [@HerrHofrat](https://github.com/HerrHofrat) and is called `Motion`, available at https://github.com/HerrHofrat/hassio-addons/tree/master/motion. You will need to add his repository as a location accessible to Hassio (search for the box that states `Add new repository by URL`). The addon will both continually capture still images, and capture timestamped images when motion is detected. I experimented with the addon settings but settled on the configuration below. The addon is configured by the Hassio tab for the addon:

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

The addon captures an image every second, saved as `latest.jpg`, and this image is continually over-written. On motion detection a timestamped image is captured with format `%v-%Y_%m_%d_%H_%M_%S-motion-capture.jpg`. All images are saved to the `/share/motion` folder on the Raspberry Pi. The above configuration should work regardless of the USB camera you are using, but if you have several USB cameras attached to your Pi you may need to use the terminal to check the camera interface (here `/dev/video0`).

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
To make Classificationbox accessible to Home-Assistant you will first need to get the Classificationbox custom component code from my Github repo, and this article requires you to use [release v0.3](https://github.com/robmarkcole/HASS-Machinebox-Classificationbox/releases/tag/v0.3). This code is added to Home-Assistant by placing the contents of the `custom_components` folder in your Home-Assistant configuration directory (or adding its contents to an existing custom_components folder). The yaml code-blocks that follow are code to be entered in the Home-Assistant `configuration.yaml` file, unless otherwise stated. To configure the Classificationbox component, add to the Home-Assistant `configuration.yaml` file:

```yaml
image_processing:
  - platform: classificationbox
    ip_address: localhost # Change to the IP hosting Classificationbox, e.g. 192.168.0.100
    port: 8080
    scan_interval: 100000
    source:
      - entity_id: camera.dummy
```
Note that by default the image will be classified every 10 seconds, but by setting a long `scan_interval` I am ensuring that image classification will only be performed when I trigger it using the `image_processing.scan` service described later. Note that the image source is `camera.dummy`, which will be the motion triggered image. The Classificationbox component fires an Home-Assistant `image_processing.image_classification` [event](https://www.home-assistant.io/docs/configuration/events/) when an image is classified with a probability greater than a threshold confidence of 80%, and we use this later to trigger a notification.

#### Tying it all together
Now that image capture is configured and Classificationbox is available to use, we must link them together using a sequence of automations in Home-Assistant. The sequence that we setup is illustrated in the diagram below, where the blue arrows represent automations:

<p align="center">
<img src="https://github.com/robmarkcole/HASS-Machinebox-Classificationbox/blob/master/bird_project/sequence.png" width="300">
</p>

Out of the box, Home-Assistant has no knowledge of when the Hassio addon captures a new motion triggered image, so I use the [folder_watcher component](https://www.home-assistant.io/components/folder_watcher/) to alert Home-Assistant to new images in the `/share/motion` directory. The configuration of folder_watcher in `configuration.yaml` is:

```yaml
folder_watcher:
  - folder: /share/motion
    patterns:
      - '*capture.jpg'
```

The `folder_watcher` component fires an event every time a new timestamped image is saved in `/share/motion` when the file name matches the pattern `*capture.jpg` (as the timestamped image file names do).  The event data includes the file name and path to the added image, and I use an automation to display the new image on `camera.dummy` using the `camera.update_file_path` service. The configuration for the automation is shown below, added to `automations.yaml`:

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

Finally I use the `image_processing.image_classification` event fired by the Classificationbox component to trigger an automation to send me any images of birds as a [Pushbullet](https://www.pushbullet.com/) notification:

```yaml
- action:
  - data_template:
      message: Class {{ trigger.event.data.id }} with probability {{ trigger.event.data.confidence
        }}
      title: New image classified
      data:
        file: ' {{states.camera.local_file.attributes.file_path}} '
    service: notify.pushbullet
  alias: Send classification
  condition: []
  id: '1120092824611'
  trigger:
  - event_data:
      id: birds
    event_type: image_processing.image_classification
    platform: event
```

The notification is shown below.

<p align="center">
<img src="https://github.com/robmarkcole/HASS-Machinebox-Classificationbox/blob/master/bird_project/iphone_notification.jpeg" width="400">
</p>

So finally we have achieved the aim of this project, and receive a notification when a bird image is captured by the motion triggered camera.

### Future Work
Now that I have basic recognition of when a bird image is captured, the next step for this project is to train the classifier to recognise particular species of birds. I have a range of species visiting the birdfeeder including Blue Tits, Robins, Dunnocks, Magpis and even Parakeets. However the vast majority of the visitors are Blue Tits, and I have so far captured relatively few images of the other species, which will make training a classifier on these species more tricky. I may have to use photos of these species from the internet, but another idea is to create a website where users can submit their own images which are then used to create a 'community' classifier that can be shared amongst all users of the website. This would allow local studies such as this one to be reproduced at a large scale. I will also investigate ways to capture the statistics on visiting birds using [counters](https://www.home-assistant.io/components/counter/), and create a custom component to allow people to automatically submit their bird data to studies such as the RSPB annual [Birdwatch](https://www.rspb.org.uk/get-involved/activities/birdwatch/) survey.


### Summary
In summary this article has described how to create an image classifier using Classificationbox, and how to deploy it for use within an automated system using Home-Assistant. A cheap USB webcam is used to capture motion triggered images, and these are posted to Classificationbox for classification. If there is a successful classification of a bird then the image is sent to my phone as a notification. Future work on this project is to train the classifier to identify different species of birds, and investigate ways to create a community classifier. One slight issue I have is that a Magpi has recently been trying to rip the feeder off the window (shown below), so I need to do some work to make it Magpi proof (Magpi destruction efforts shown below)! Additionally, now that it is Summer, this seasons crop of baby birds have fledged and my bird feeder has very few visitors as there is now abundant food in nature. However I will continue to refine the system and aim to deploy it next Spring. I hope this project inspires you to try out using deep-learning classifiers in your projects.

<p align="center">
<img src="https://github.com/robmarkcole/HASS-Machinebox-Classificationbox/blob/master/bird_project/magpi.png" width="900">
</p>
**Above:** A Magpi either intentionally or unintentionally almost rips the bird feeder off the window. The arriving Blue Tit is unable to feed normally owing to the position of the feeder, but finds another way to feed via a hole in the top of the feeder!

### Links
* Classificationbox: https://machineboxio.com/docs/classificationbox) provides
* Home-Assistant: https://www.home-assistant.io/
* Hassio: https://www.home-assistant.io/hassio/
* Hassio Motion addon: https://github.com/HerrHofrat/hassio-addons/tree/master/motion
* Docker: https://www.docker.com/community-edition
* Classificationbox custom component for Home-Assistant: https://github.com/robmarkcole/HASS-Machinebox-Classificationbox
* Pushbullet: https://www.pushbullet.com/
* RSPB Birdwatch: https://www.rspb.org.uk/get-involved/activities/birdwatch/
