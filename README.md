# README

**Coming Next: better trained darknet19 model, and easier to use detection tool, (better explanation for dataset folder placement for training)**

## Scripts
Open terminal in the home directory of the project. Then you can:
* use darknet19 to detect: `python src/pascal/pascal_detect_darknet.py`
  * first, put [pascal trained darknet19 weights](https://www.dropbox.com/sh/gvrr4udelzflcc7/AAAr_Sg4BimiAfssdIhiGO7va?dl=0) into `weights/` directory (since the model has been only trained for 80k iterations, the accuracy might not be optimal)
  * change the `image_path` variable to the image you want to do detection on
* train darknet19 on ImageNet: `python src/imagenet/imagenet_train_darknet.py`
  * first, put ILSVRC dataset into `data/` folder
  * the implementation is multithread, though it might not be optimized enough. Sugguestions are welcome.
* train tensorflow slim resnet50 on Pascal: `python src/pascal/pascal_train_resnet.py`
  * first, put VOCdevkit dataset into `data/` folder
  * second, put [tensorflow slim resnet50 weights](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz) in the `weights/` directory

## Resources
Trained tensorflow model ckpts:
* tensorflow slim [resnet50](https://www.dropbox.com/sh/bsj9fuuv4co23qy/AABRjYECNkCTPzgjWyBZMvLRa?dl=0) further trained on Pascal VOC2007
* [darknet19](https://www.dropbox.com/sh/7ncpmioirhr735e/AAALEw1nEJqQZRqtNGDvhiSHa?dl=0) trained on ImageNet for 88 epochs (validation accuracy ~60)
* [darknet19](https://www.dropbox.com/sh/gvrr4udelzflcc7/AAAr_Sg4BimiAfssdIhiGO7va?dl=0) trained on Pascal VOC2007 (only for 80k iterations, the accuracy might not be optimal)

## References
* Darknet19 is built according to [YOLO1](https://arxiv.org/abs/1506.02640) and [YOLO2](https://arxiv.org/abs/1612.08242) paper by J. Redmon et al.