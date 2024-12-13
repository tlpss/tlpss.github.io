---
layout: page
title: GELLO Teleop
description: Robot teleoperation setup using GELLO arms
img: assets/img/projects/thumbnail-gello-teleop.jpg
importance: 1
category: work
---

To collect demonstrations for robot policy learning, we built two [GELLO](https://wuphilipp.github.io/gello_site/) UR replica's and created a dual-arm teleop setup. 

{% include figure.liquid path="assets/img/projects/thumbnail-gello-teleop.jpg" class="img-fluid mx-auto d-block rounded z-depth-1" %}


You can see an example of the fine-grained manipulation tasks that can be demonstrated with such a teleop setup in the video below:

<iframe width="560" height="315" src="https://www.youtube.com/embed/YMs0lpAdEcw?si=L-gd4iHaRtiIEaLn" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


Using Behavior Learning algorithms, such as [ACT](), we can train our robot to execute fine-grained manipulation tasks. Below is an example of how the robot is able to autonomously open our (instrumented) Nespresso machine;

<iframe width="560" height="315" src="https://www.youtube.com/embed/ddFOJ304XMc?si=O0Bd7obf_Aux7j4l" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

The codebase is available [here](https://github.com/tlpss/gello_software). A big thank you to [Philipp Wu](https://wuphilipp.github.io/) for open-sourcing GELLO.