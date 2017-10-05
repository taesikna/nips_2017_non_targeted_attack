# nips_2017_non_targeted_attack

This repository is created to share my final submission for
[NIPS 2017 Non-targeted Adversarial Attack Competition](https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack)


### Prerequisites

Please run download_checkpoints.sh to download pre-trained ImageNet models used in this implementation.

```
./download_checkpoints.sh
```

### Generating Adversarial Images

After you have downloaded pre-trained models, please run run_attack.sh to generate adversarial images.

```
mkdir output
run_attack.sh $PATH_TO_SOURCE_IMAGES output 8
```

Please feel free to enable debug_flag in attack_ens_iter_fgsm.py to print prediction for debugging.
Enjoy !!!


