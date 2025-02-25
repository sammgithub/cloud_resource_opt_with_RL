#!/bin/bash
set -e

echo "Starting training: gWaveNet on GWdata"
python train-GWdata-gWaveNet.py
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches

echo "Starting training: AlexNet on CIFAR10"
python train-cifar10-alexnet.py
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches

echo "Starting training: DenseNet121 on CIFAR10"
python train-cifar10-densenet121.py
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches

echo "Starting training: EfficientNetB5 on CIFAR10"
python train-cifar10-efficientnetB5.py
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches

echo "Starting training: InceptionV3 on CIFAR10"
python train-cifar10-inceptionV3.py
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches

echo "Starting training: MobileNetV3-Large on CIFAR10"
python train-cifar10-mobileNetV3lrg.py
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches

echo "Starting training: ResNet50 on CIFAR10"
python train-cifar10-resnet50.py
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches

echo "Starting training: VGG16 on CIFAR10"
python train-cifar10-vgg16.py
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches

echo "Starting training: Vision Transformer on CIFAR10"
python train-cifar10-vit.py
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches

echo "Starting training: AlexNet on ImageNet45"
python train-imagenet45-alexnet.py
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches

echo "Starting training: DenseNet121 on ImageNet45"
python train-imagenet45-densenet121.py
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches

echo "Starting training: EfficientNetB5 on ImageNet45"
python train-imagenet45-efficientnetB5.py
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches

echo "Starting training: InceptionV3 on ImageNet45"
python train-imagenet45-inceptionV3.py
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches

echo "Starting training: MobileNetV3-Large on ImageNet45"
python train-imagenet45-mobileNetV3lrg.py
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches

echo "Starting training: ResNet50 on ImageNet45"
python train-imagenet45-resnet50.py
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches

echo "Starting training: VGG16 on ImageNet45"
python train-imagenet45-vgg16.py
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches

echo "Starting training: Vision Transformer on ImageNet45"
python train-imagenet45-vit.py
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches

echo "All training runs are complete."
