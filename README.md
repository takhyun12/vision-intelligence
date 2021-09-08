# vision-intelligence
![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9-blue)
![PyPI](https://badge.fury.io/py/tensorflow.svg)

This repository is still under construction.

## References
> [Github-AdaIN-style](https://github.com/xunhuang1995/AdaIN-style)

> [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)

## Install:
To install the current release:

```shell
$ pip install {}
```

## Table of Contents

* AdaIN Style transfer

![adain](https://user-images.githubusercontent.com/41291493/132440767-673332f9-6ec9-4fb9-aca6-236e4df64198.png)



## Usage:

* Usage for AdaIN Style transfer

``` python
>>> from AdaIN import AdaIN_v1
>>>
>>> content_image_path: str = 'AAA.jpg'
>>> style_image_path: str = 'BBB.jpg'
>>> alpha: float = 0.7  # 0.0 <= alpha <= 1.0
>>>
>>> adain = AdaIN_v1(content_image=content_image_path, style_image=style_image_path, alpha=alpha)
>>> adain.style_transfer()
```

## TODO / Known Issues:
