### Bird classification project: Introduction

Aim of project: automatically capture and classify images of birds visiting my bird-feeder. A long term goal of this project is to contribute the captured data to a nationwide study of bird populations.

Tools used:
* **Motion** for capturing motion triggered images
* **Classificationbox** for classifying bird/not-bird images
* **Home-Assistant** for coordinating the image capture/processing, storing data and sending notifications

This write-up will first present the image classification work using Classificationbox, then describe the practical implementation within an automated system.


### The problem definition

Being interested in bird watching, I attached a bird feeder to a window of my flat and within a few days various species of bird started visiting the feeder. I decided it would be fun to rig up a motion triggered camera to capture images of the birds, and I used Home-Assistant and a £10 USB webcam to capture images via motion trigger, and setup Home-Assistant to send an image notification to my phone when an image was captured. However I quickly discovered that all kinds of motion could trigger an image capture. The result was hundreds of images of all kinds of motion, such as planes flying in the distance or even funky light effects. Approximately less than half the images actually contained a bird, so I decided it was necessary to filter out the non-bird images. I have been interested in image classification for a while, and whilst searching online I came across [this article on Classificationbox](https://blog.machinebox.io/how-anyone-can-build-a-machine-learning-image-classifier-from-photos-on-your-hard-drive-very-5c20c6f2764f), which looked ideal for this project.


### Classificationbox
[Classificationbox](https://machineboxio.com/docs/classificationbox) provides a ready-to-train classifier, deployed in a Docker container and exposed via a REST API. It uses [online learning](https://en.wikipedia.org/wiki/Online_machine_learning) to quickly train a classifier that can be used to automatically classify various types of data, such as text, images, structured and unstructured data.

Assuming you have [Docker installed](https://www.docker.com/community-edition), first download Classificationbox [from Dockerhub](https://hub.docker.com/r/machinebox/classificationbox/) by entering in the terminal:
```
sudo docker pull machinebox/classificationbox
```

Then run the container with:
```
MB_KEY="INSERT-YOUR-KEY-HERE"
sudo docker run -p 8080:8080 -e "MB_KEY=$MB_KEY" machinebox/classificationbox
```

Now explain training.
