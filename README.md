# README

**Coming Next: Yolo detection using resnet50 and Pascal trained Darknet19**

## Scripts
Open terminal in the home directory of the project. Place relevant weights in the weights/ directory. Then you can:
* train darknet19 on ImageNet: `python src/imagenet/imagenet_train_darknet.py`
  * the implementation is multithread, though it might not be optimized enough. Sugguestions are welcome.
* train tensorflow slim resnet50 on Pascal: `python src/pascal/pascal_train_resnet.py`

## Resources
* tensorflow slim [resnet50](https://www.dropbox.com/sh/bsj9fuuv4co23qy/AABRjYECNkCTPzgjWyBZMvLRa?dl=0) further trained on Pascal
* [darknet19](https://www.dropbox.com/sh/7ncpmioirhr735e/AAALEw1nEJqQZRqtNGDvhiSHa?dl=0) trained on ImageNet for 88 epochs (validation accuracy ~60)

## References
* Darknet19 is built according to [YOLO1](https://arxiv.org/abs/1506.02640) and [YOLO2](https://arxiv.org/abs/1612.08242) paper by J. Redmon et al.