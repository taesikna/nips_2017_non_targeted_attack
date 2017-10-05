# nips_2017_non_targeted_attack

This repository is created to share my final submission for
[NIPS 2017 Non-targeted Adversarial Attack Competition](https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack)


### Prerequisites

Please run download_checkpoints.sh to download pre-trained ImageNet models used in this implementation.

```
./download_checkpoints.sh
```

### Attach Method

In particular, I've used 2 adversarially trained ImageNet models (Inception ResNet v2 and Inception v3 trained with ensemble adversarial training) and 1 purely trained model (Inception v3) as source networks for adversarial images generation.
You can see the full list of adversarially trained ImageNet models in [adv_imagenet_models](https://github.com/tensorflow/models/tree/master/research/adv_imagenet_models)
and image classification models in TF-Slim in [slim](https://github.com/tensorflow/models/tree/master/research/slim).

I've used iterative basic method introduced in [Adversarial Machine Learning at Scale](https://arxiv.org/abs/1611.01236) and generated adversarial images per each model.
Then, I've used average adversarial noises to create final adversarial images.
I've set adversarial noise to be max_e if the average noise > max_e/8, and -max_e if the average noise < -max_e/8.

### Generating Adversarial Images

After you have downloaded pre-trained models, please run run_attack.sh to generate adversarial images.

```
mkdir output
run_attack.sh $PATH_TO_SOURCE_IMAGES output 8
```

Please feel free to enable debug_flag in attack_ens_iter_fgsm.py to print prediction for debugging.
Enjoy !!!


