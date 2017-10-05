#!/bin/bash
#
# Scripts which download checkpoints for provided models.
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

mkdir "${SCRIPT_DIR}/inception_v3"
mkdir "${SCRIPT_DIR}/ens4_adv_inception_v3"
mkdir "${SCRIPT_DIR}/ens_adv_inception_resnet_v2"

# Download inception v3 checkpoint for fgsm attack.
wget http://download.tensorflow.org/models/ens4_adv_inception_v3_2017_08_18.tar.gz
tar -xvzf ens4_adv_inception_v3_2017_08_18.tar.gz -C ens4_adv_inception_v3/
rm ens4_adv_inception_v3_2017_08_18.tar.gz

wget http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz
tar -xvzf ens_adv_inception_resnet_v2_2017_08_18.tar.gz -C ens_adv_inception_resnet_v2/
rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz

wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar -xvzf inception_v3_2016_08_28.tar.gz -C inception_v3
rm inception_v3_2016_08_28.tar.gz

python tf_rename_variables.py --checkpoint_dir=inception_v3 --replace_from=InceptionV3 --replace_to=Pure/InceptionV3 

