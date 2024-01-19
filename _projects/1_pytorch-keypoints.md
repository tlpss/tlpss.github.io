---
layout: page
title: Keypoint Detection 
description: Framework for 2D keypoint detection in Pytorch 
img: assets/img/projects/thumbnail-keypoints.png
importance: 1
category: work
---

code on  [github](https://github.com/tlpss/keypoint-detection)

This is framework for one-stage, heatmap-based keypoint detection. We use it for state estimation in robot pipelines, trained on both real and synthetic data. It can deal with symmetric objects by combining multiple semantic landmarks into a single heatmap. Below you can see the output of a model to detect corners of towels, that has been trained entirely on synthetic data:

{% include figure.liquid path="assets/img/projects/cloth_keypoints_test.png" class="img-fluid mx-auto d-block rounded z-depth-1" %}


Most importantly this project has allowed me to learn more about *software engineering for ML systems* as I have been using it long enough to have me spend a lot of time on refactoring bad architectural choices made in the past, and as it is used by other people at my lab as well, implying there is a need for stability and better documentation. These are things you normally don't get to experience when working on a paper-project. 

