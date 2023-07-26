# vit_cifar10
vit cifar10 test with image size 32, 64


### vit test result with patch size 32

acc : 83.6

https://github.com/YuBeomGon/vit_cifar10/blob/master/notebooks/vit-scratch-s4.ipynb

****

### vit test result with patch size 32, v2 training

regard v2, 

![new training method](example/new_training_method.png)

https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/

acc : 88.0

https://github.com/YuBeomGon/vit_cifar10/blob/master/notebooks/vit-scratch-v2.ipynb

****

### vit fine tune to img_size 64

acc : 89.1

https://github.com/YuBeomGon/vit_cifar10/blob/master/notebooks/vit-64-from-32.ipynb

confusion matrix

![](example/cifar_confusion_mat.png)


attn map visualize

![](example/attn_map_plane.png)
![](example/attn_amp_dog_cat.png)





