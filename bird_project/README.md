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

You will want to replace my model ID (`5b0ce5d8023d4e35`) with your own. The downloaded file is 60 kb, so small enough to be shared on Github and other online hosts. This is useful if you want others to be able to reproduce your work.

### Using Classificationbox
You can perform a quick image classification using the model by using cURL. From within the same folder as the image `bird.jpg` I used the following:

```cURL
curl -X POST -F 'file=@bird.jpg' http://localhost:8080/classificationbox/models/5b0ce5d8023d4e35/predict
```
GETTING AN ERROR
