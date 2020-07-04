r"""Modified from vggish_inference_demo.py

Usage:
  $ python audio_to_prediction.py <audio>
"""

from __future__ import print_function

import sys
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow as tf2

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

flags = tf.app.flags

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')


FLAGS = flags.FLAGS

frame_n = 2
saved_model_path = 'saved_model/0001'


def main():
	extract_and_predict('test.wav')


def extract_and_predict(wav):
    wav_file = wav
    examples_batch = vggish_input.wavfile_to_examples(wav_file)

    # Prepare a postprocessor to munge the model embeddings.
    pproc = vggish_postprocess.Postprocessor()

    with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)

        # Run inference and postprocessing.
        [embedding_batch] = sess.run([embedding_tensor],
                                    feed_dict={features_tensor: examples_batch})
        postprocessed_batch = pproc.postprocess(embedding_batch)
        postprocessed_batch = [postprocessed_batch[i] for i in range(len(postprocessed_batch))]
    pred_each_n_seconds = predict_with_saved_model(postprocessed_batch)
    print( str(pred_each_n_seconds))

def predict_with_saved_model(feature):
	# [0, 1] is snoring
	# [1, 0] is non-snoring
	tf.enable_v2_behavior()
	loaded_model = tf2.saved_model.load(saved_model_path)
	pred = loaded_model(feature).numpy()
	# Take out snoring possibilities
	# ~1 is more likely snoring; ~0 is not snoring
	pred = pred[:,1]
	reframed = frame_multi_seconds(pred)
	return np.round(reframed, 3)

def frame_multi_seconds(pred_single_second):
    res = []
    if frame_n >= len(pred_single_second):
        res.append(sum(pred_single_second) / len(pred_single_second))
    else:
        for i in range(len(pred_single_second)-frame_n+1):
            sub = pred_single_second[i: i+frame_n]
            res.append(sum(sub) / len(sub))
    return res

if __name__ == '__main__':
	# tf.app.run()
	main()
