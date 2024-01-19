---
layout: page
title: Contact-GraspNet 
description: Dockerized Contact-GraspNet for ease of use.
img: assets/img/projects/thumbnail_dockerized-contact-graspnet.png
importance: 1
category: work
---

To quickly generate grasp proposals in research projects while avoiding a dependency hell, I have dockerized the amazing [Contact-GraspNet](https://github.com/NVlabs/contact_graspnet) network. 

You can find the repo [here](https://github.com/tlpss/contact_graspnet) and a prebuilt docker container [here](https://hub.docker.com/r/tlpss/contact-graspnet-flask).

The network has proven fairly robust during some tests in the lab while the docker overhead on the inference time is neglible. The limitations of using point clouds are still there of course.
<br>


<iframe width="800" height="500"
src="https://github.com/tlpss/contact_graspnet/assets/37955681/2236c3f0-8157-4cf0-a167-cc2ecae7a4df">
</iframe>
<br>


To quickly set up this demo in under 200 lines of code, I used the [airo-mono](https://github.com/airo-ugent/airo-mono) packages.