"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

#from cleverhans.attacks import FastGradientMethod
#from cleverhans.attacks import BasicIterativeMethod
from attacks import BasicIterativeMethod
import numpy as np
from PIL import Image

import tensorflow as tf
import inception_resnet_v2
from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath) as f:
      image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images


def save_images(images, filenames, output_dir):
  """Saves images to the output directory.

  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
      img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
      Image.fromarray(img).save(f, format='PNG')

# this function is from RalphMao commented on 17 Mar
# at https://github.com/tensorflow/tensorflow/issues/312
def optimistic_restore(session, save_file, include_global_step=True):
  reader = tf.train.NewCheckpointReader(save_file)
  saved_shapes = reader.get_variable_to_shape_map()
  var_names = sorted([(var.name, var.name.split(':')[0])
                      for var in tf.global_variables()
                      if var.name.split(':')[0] in saved_shapes])
  restore_vars = []
  name2var = dict(zip(map(lambda x:x.name.split(':')[0],
                          tf.global_variables()), tf.global_variables()))
  with tf.variable_scope('', reuse=True):
    for var_name, saved_var_name in var_names:
      curr_var = name2var[saved_var_name]
      var_shape = curr_var.get_shape().as_list()
      if var_shape == saved_shapes[saved_var_name]:
        if 'global_step' in saved_var_name:
          if include_global_step:
            restore_vars.append(curr_var)
        else:
            restore_vars.append(curr_var)
  saver = tf.train.Saver(restore_vars)
  saver.restore(session, save_file)


class InceptionPureModel(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with tf.variable_scope('Pure') as scope:
      with slim.arg_scope(inception.inception_v3_arg_scope()):
        _, end_points = inception.inception_v3(
            x_input, num_classes=self.num_classes, is_training=False,
            reuse=reuse)
    self.built = True
    output = end_points['Predictions']
    # Strip off the extra reshape op at the output
    probs = output.op.inputs[0]
    return probs

class InceptionModel(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(inception.inception_v3_arg_scope()):
      _, end_points = inception.inception_v3(
          x_input, num_classes=self.num_classes, is_training=False,
          reuse=reuse)
    self.built = True
    output = end_points['Predictions']
    # Strip off the extra reshape op at the output
    probs = output.op.inputs[0]
    return probs

class InceptionResModel(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
      _, end_points = inception_resnet_v2.inception_resnet_v2(
          x_input, num_classes=self.num_classes, is_training=False,
          reuse=reuse)
    self.built = True
    output = end_points['Predictions']
    # Strip off the extra reshape op at the output
    probs = output.op.inputs[0]
    return probs


def main(_):
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
  debug_flag = False
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  eps_iter = 1.0 / 255.0
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1001

  tf.logging.set_verbosity(tf.logging.INFO)

  checkpoints = ['./ens4_adv_inception_v3/ens4_adv_inception_v3.ckpt',
                 './ens_adv_inception_resnet_v2/ens_adv_inception_resnet_v2.ckpt',
                 './inception_v3/inception_v3.ckpt' ]

  model_names = [InceptionModel, InceptionResModel, InceptionPureModel]

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)


    models = []
    for model_name in model_names:
      models.append(model_name(num_classes))

    iter_fgsms = []
    for model in models:
      iter_fgsms.append(BasicIterativeMethod(model))

    x_advs = []
    model_preds = []
    for iter_fgsm in iter_fgsms:
      x_adv, model_pred, y_target = iter_fgsm.generate(x_input,
                                 eps=eps, clip_min=-1., clip_max=1.,
                                 eps_iter=eps_iter,
                                 nb_iter=int(max(FLAGS.max_epsilon+4,
                                                 1.25*FLAGS.max_epsilon)*2) )
      x_advs.append(x_adv)
      model_preds.append(model_pred)

    # Run computation
    saver = tf.train.Saver()
    run_config = tf.ConfigProto()
    with tf.Session(config=run_config) as sess:

      for checkpoint in checkpoints:
        optimistic_restore(sess, checkpoint)

      i = 0
      for filenames, images in load_images(FLAGS.input_dir, batch_shape):
        adv_images, preds = sess.run([x_advs, model_preds],
                                     feed_dict={x_input: images})
        diff_images = []
        for adv_images_per_model in adv_images:
          diff_images.append(adv_images_per_model - images)

        ens_diffs = np.mean(diff_images, axis=0)
        ens_diffs[np.where(ens_diffs > eps/8)] = eps
        ens_diffs[np.where(ens_diffs < -eps/8)] = -eps

        ens_adv_images = np.clip(ens_diffs + images, -1., 1.)

        save_images(ens_adv_images, filenames, FLAGS.output_dir)

        if debug_flag:
          adv_preds = sess.run(model_preds, feed_dict={ x_input: adv_images[0]})
          adv_preds2 = sess.run(model_preds, feed_dict={ x_input: adv_images[1]})
          adv_preds3 = sess.run(model_preds, feed_dict={ x_input: adv_images[2]})

          ens_adv_preds = sess.run(model_preds, feed_dict={ x_input: ens_adv_images})
          filenames1 = []
          filenames2 = []
          filenames3 = []
          for j, filename in enumerate(filenames):
            filename = filename.split('.')[0]
            filenames1.append(filename + '_v3.png')
            filenames2.append(filename + '_res2.png')
            filenames3.append(filename + '_pure_v3.png')
          save_images(adv_images[0], filenames1, FLAGS.output_dir)
          save_images(adv_images[1], filenames2, FLAGS.output_dir)
          save_images(adv_images[2], filenames3, FLAGS.output_dir)
          for j, (pred, adv_pred, adv_pred2, adv_pred3, ens_adv_pred) \
            in enumerate(zip(preds, adv_preds, adv_preds2, adv_preds3, ens_adv_preds)):
            print ('Test for model ', j)
            print ('clean prediction: \n',
                    np.argmax(pred, axis=1), ' ', np.max(pred, axis=1))
            print ('adv images from model 0 prediction: \n',
                    np.argmax(adv_pred, axis=1),  ' ', np.max(adv_pred, axis=1))
            print ('adv images from model 1 prediction: \n',
                    np.argmax(adv_pred2, axis=1), ' ', np.max(adv_pred2, axis=1))
            print ('adv images from model 2 prediction: \n',
                    np.argmax(adv_pred3, axis=1), ' ', np.max(adv_pred3, axis=1))
            print ('ens_adv images prediction: \n',
                    np.argmax(ens_adv_pred, axis=1), ' ',
                    np.max(ens_adv_pred, axis=1))

        print ('%d images are being processed' % ((i+1)*FLAGS.batch_size))
        i+=1


if __name__ == '__main__':
  tf.app.run()
