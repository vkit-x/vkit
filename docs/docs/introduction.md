---
slug: /
---

# Overview

## Introduction

<div align="center">

<img alt="logo.svg" width="230" src="img/logo.svg" />

[![release-nightly](https://github.com/vkit-x/vkit/actions/workflows/release-nightly.yaml/badge.svg)](https://github.com/vkit-x/vkit/actions/workflows/release-nightly.yaml)
[![license](https://img.shields.io/github/license/vkit-x/vkit)](https://github.com/vkit-x/vkit/blob/master/LICENSE)

</div>

[vkit](https://github.com/vkit-x/vkit) is a toolkit designed for CV (Computer Vision) developers, especially targeting document image analysis and optical character recognition workloads:

* Supporting rich data augmentation strategies:
  * Common [photometric distortion](feature/photometric-distortion/interface.md) strategies such as various colorspace manipulation methods and image noise related techniques
  * ⭐ Common [geometric distortion](feature/geometric-distortion/interface.md) strategies such as various affine transformations and non-linear transformations (e.g. [similarity MLS](feature/geometric-distortion/mls.md#similarity_mls), [camera-model based 3D surface curving](feature/geometric-distortion/camera.md#camera_cubic_curve), [folding effect](feature/geometric-distortion/camera.md#camera_plane_line_fold), etc.)
  * ⭐ Simultaneously transforming labeled data while performing geometric distortion. As an example, while an image was rotated, vkit will rotate the corresponding positional label (e.g. image mask, polygons) at the same time without manual intervention
* Supporting comprehensive data type encapsulation and the corresponding visualization:
  * [Image type](utility/image.md) (encapsulation based on PIL, supporting reading/writing various image file types)
  * [Labeled data type](utility/label.md): mask, score map, box, polygon and so on
* Industrial-grade code quality:
  * Auto-completion and type hint friendly, making it practical to be used in production
  * Matured package and dependency management
  * Automated code style enforcement (based on flake8) and static type checker (based on pyright)

Remarks: ⭐ Highlights (features that other similar projects have not, or not elegantly supported)

### Demo!

[camera_cubic_curve](feature/geometric-distortion/camera.md#camera_cubic_curve):

<div align="center">
    <video width="30%" height="30%" autoplay="true" muted="true" playsinline="true" loop="true" controls="ture">
        <source src="/homepage/home_page_camera_cubic_curve.mp4" type="video/mp4" />
    </video>
</div>

[rotate](feature/geometric-distortion/affine.md#rotate):

<div align="center">
    <video width="30%" height="30%" autoplay="true" muted="true" playsinline="true" loop="true" controls="ture">
        <source src="/homepage/home_page_rotate.mp4" type="video/mp4" />
    </video>
</div>

## Objectives

The author, as a CV/NLP engineer, wishes to bring the convenience to developers in the aforementioned disciplines through this project:

* To free developers from the tedious data governance tasks, therefore more time can be spent on actual high-value work such as the data governance strategies, model designing and fine tuning
* To consolidate common data augmentation techniques, aiming to aid document image analysis and recognition researches, and their industrial practices. The author wishes to make the "secret sauce", i.e. the industrial grade data augmentation methods, available to public
* To construct open-source industrial document image analysis and recognition solutions powered by vkit:
  * Distortion correction
  * Hyper resolution
  * OCR
  * Layout Analysis
  * ...

## Installation

Python version requirement: 3.8 and 3.9 (No plan to support Python version lower than 3.8 due to third-party package dependency complications)

To install the stable release:

```bash
pip install vkit
```

To install the nightly version (tracking the latest commit in main branch):

```bash
pip install vkit-nightly
```

(click [here](https://vkit-nightly.vkit-x.com/) to visit the nightly documentation)

## Recent release plans

* 0.2.0
  - [ ] Complete CI testing pipeline
  - [ ] Support font rendering
  - [ ] Support generating training set for OCR text detection tasks
  - [ ] Support generating training set for OCR text recognition tasks

## Recent stable releases

* 0.1.2
  - Remove strict dependency versioning
* 0.1.1
  - User manual (English version)
  - GitHub Page for serving user manual
* 0.1.0
  - Support Python 3.9
  - Support Python 3.8
  - Image type encapsulation
  - Labeled data type encapsulation
  - Common photometric distortion strategies
  - Common geometric distortion strategies
  - User manual

<!---
## Funding
TODO: Setup a patreon account.
-->

## Communication

* Question, or requesting for new feature: [Discussions](https://github.com/vkit-x/vkit/discussions)
* Bug reporting: [Issues](https://github.com/vkit-x/vkit/issues)

Your kind understanding will be greatly appreciated if the response is slow on these forums as the author is busy with his work while he cannot devote his full time into this project
