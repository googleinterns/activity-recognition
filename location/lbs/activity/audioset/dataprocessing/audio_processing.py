"""
Script to process a csv file from AudioSet into a csv file with features.

Each row of the inputted csv is parsed for the video_id, start_time_seconds,
end_time_seconds, and labels.
The downloaded audios would be clipped into the labelled 10-seconds segments.
The audio segments are then processed using Librosa to extract features.

Usage
Standalone script:
python audio_processing.py
    --src_dir PATH_TO_INPUT_FILES --dest_dir PATH_TO_OUTPUT_FILES
    --filename CSV FILE --labels LABELS
    --redo BOOLEAN TO REDOWNLOAD THE YOUTUBE FILES

Example:
python audio_processing.py
    --src_dir location/lbs/activity/audioset/dataprocessing/example_src_dir
    --dest_dir location/lbs/activity/audioset/dataprocessing/example_dest_dir
    --filename balanced_train_segments --labels "Gunshot, gunfire" --False

Typical usage example (Bazel):
bazel build location/lbs/activity/audioset/dataprocessing:audio_processing
bazel-bin/location/lbs/activity/audioset/dataprocessing/audio_processing
    --src_dir location/lbs/activity/audioset/dataprocessing/example_src_dir
    --dest_dir location/lbs/activity/audioset/dataprocessing/example_dest_dir
    --audioset_csv balanced_train_segments --labels "Gunshot, gunfire" --False

    Performance: to confirm the presence of, extract features from, and output
    to a csv file, takes about 3 hours for around 20,000 samples.
"""
from __future__ import unicode_literals
from os.path import isfile, join
import librosa
from absl import app
from absl import flags
from absl import logging
import datetime
import pandas as pd
import numpy as np
import audioset_helper
import downloader

FLAGS = flags.FLAGS
flags.DEFINE_string('filename', 'balanced_train_segments',
                    'The csv file of a dataset provided by the audioset')
flags.DEFINE_multi_string('labels', ['Gunshot, gunfire'],
                          'The set of labels identified by the audioset to '
                          'treat as positive')
flags.DEFINE_multi_string('features', ['mfcc'],
                          'The set of features to extract from the audio files')
flags.DEFINE_bool('redo', False,
                  'Whether to redownload the YouTube videos in the included'
                  ' csv file')
flags.DEFINE_string('src_dir',
                    'location/lbs/activity/audioset/dataprocessing'
                    '/example_src_dir',
                    'The location of the audioset csv file(s) and the '
                    'ontology.json file')
flags.DEFINE_string('dest_dir',
                    'location/lbs/activity/audioset/dataprocessing'
                    '/example_dest_dir',
                    'The location of the downloaded YouTube videos and the '
                    'outputted csv file containing the extracted features '
                    'and labels')
flags.DEFINE_bool('scaled', True, 'Whether to average the extracted features')


def extract_feature(dest_dir, video_id, feature, scaled):
    """Extracts a feature from a specific audio file given a video_id.

   Given a specific of video_id, it extracts features using the Librosa library
   from the corresponding audio file and stores the features in a dictionary
   with a video_id, dictionary key-value pair, with the dictionary value being
   comprised of feature name, feature list key-value pairs. Using Librosa it
   extracts one of the following features: chroma_stft, chroma_cqt, chroma_cens,
   melspectogram, mfcc, rms, spectral_centroid, spectral_bandwidth,
   spectral_contrast, spectral_flatness, spectral_rolloff, poly_features,
   tonnetz, zero_crossing_rate

   Args:
       dest_dir: Path to the parent directory where the downloaded videos are
           stored.
       video_id: The video_id of specific audio file from which features are to
           be extracted.
       feature: A dictionary with video_id, dictionary pairs with the
           dictionary values being feature name, feature list key-value pairs.
       scaled: whether the extracted feature should be averaged

   Returns:
       A boolean of whether the extracting features was successful

   Raises: ValueError: A feature not supported by Librosa has been inputted.
   """
    path = join(dest_dir, 'yt_videos', 'sliced_' + video_id + '.wav')
    if not isfile(path):
        return None
    try:
        # get audio (y) and sampling rate (sr)
        y, sr = librosa.load(dest_dir + '/yt_videos/sliced_'
                             + video_id + '.wav')
    except ValueError as error:
        logging.error(error)
        return None
    switcher = {
        'chroma_stft': librosa.feature.chroma_stft,
        'chroma_cqt': librosa.feature.chroma_cqt,
        'chroma_cens': librosa.feature.chroma_cens,
        'melspectrogram': librosa.feature.melspectrogram,
        'mfcc': librosa.feature.mfcc,
        'rms': librosa.feature.rms,
        'spectral_centroid': librosa.feature.spectral_centroid,
        'spectral_bandwidth': librosa.feature.spectral_bandwidth,
        'spectral_contrast': librosa.feature.spectral_contrast,
        'spectral_flatness': librosa.feature.spectral_flatness,
        'spectral_rolloff': librosa.feature.spectral_rolloff,
        'poly_features': librosa.feature.poly_features,
        'tonnetz': librosa.feature.tonnetz,
        'zero_crossing_rate': librosa.feature.zero_crossing_rate
    }
    sample_rate_void_feature_names = {'rms', 'spectral_flatness'}
    feature_extraction_func = switcher.get(feature)
    if feature_extraction_func is None:
        raise ValueError
    if feature in sample_rate_void_feature_names:
        extracted_feature = feature_extraction_func(y)
    else:
        extracted_feature = feature_extraction_func(y, sr)
    if scaled:
        extracted_feature = np.mean(extracted_feature.T, axis=0)
    logging.info('extracted features')
    return extracted_feature


def is_positive_example(labels_list, labels_set):
    """Returns a 1 if any label in the labels_list is in labels_set else a 0.

    Iterates through the labels_list and returns a 1 if any label is in the
    labels_set, and returns a 0 otherwise.

    Args:
        labels_list: a list of labels in the format found in the audioset csv
        labels_set: a set of labels past in via command line arguments in the
                    format found in the audioset csv

    Returns:
        A integer, a 0 or 1, depending on if any label in labels_list is present
        in labels_set
    """
    for label in labels_list:
        if label in labels_set:
            return True
        else:
            return False


def delete_outer_list(alist):
    """Removes outer lists from a nested list.

    Removes outer lists from a nested list until the list has more than one
    element.

    Args:
        alist: Inputted list from which to remove any unnecessary outer lists.

    Returns:
        A list with all unnecessary outer lists removed. For example:

        [[1,2,3,4],[1,3,5,2]] or [1,5,3,2,6,9]
    """
    if len(alist) == 1 and isinstance(alist, list):
        return delete_outer_list(alist[0])
    return alist


def print_flags():
    """Prints out the flags inputted via command line arguments.

    Prints a filename, a list of labels, the redo boolean, a source
    directory, a destination directory, the scaled boolean, and a list of
    features, based on the command line input or their default values.
    """
    print(FLAGS.filename)
    print(FLAGS.labels)
    print(FLAGS.redo)
    print(FLAGS.src_dir)
    print(FLAGS.dest_dir)
    print(FLAGS.scaled)
    print(FLAGS.features)


def output_df(src_dir, dest_dir, filename, labels, features_to_extract,
              scaled=True, redo=False):
    """Creates dataframe object from inputted csv files, features, and labels.

        Parses through a csv file and extracts all the metadata from it. Then
        iterates through the list of metadata confirming that each video
        associated with every video_id is downloaded and attempt to download if
        the video is missing. The specified features are then extracted from the
        downloaded videos, and the features are then put into a dataframe with
        a label of 1 or 0 depending on if the video's label is in the list of
        labels passed in.

        Args:
            src_dir: Path to where all the input files are expected to be.
            dest_dir: Path to where all the output files will be stored.
            filename: Name of the audioset csv file not including '.csv'
            labels: A list of labels to treat as positive examples.
            features_to_extract: A list of features to extract.
            scaled: A boolean on whether to average the extracted features.
            redo: A boolean on whether to re-download all the videos.

        Returns:
            A pandas dataframe object with the following format:

            LABEL   FEATURE1   FEATURE2   ...
            0       [1, 3, 5]  [2, 4, 5]  ...
            1       [1, 1, 5]  [4, 4, 5]  ...
            0       [1, 3, 5]  [1, 4, 5]  ...
        """
    begin_time = datetime.datetime.now()
    path = join(dest_dir, filename + '_features.csv')
    if isfile(path) and not redo:
        datasetdf = pd.read_csv(path)
        return datasetdf
    dataset = []
    label_dict = audioset_helper.get_label_dict(src_dir)
    labels_set = audioset_helper.label_list_to_set(labels, label_dict)
    audio_dict = audioset_helper.parse_metadata(src_dir, filename)
    count = 0
    vid_count = 0
    downloader.download_from_list(dest_dir, audio_dict, redo)
    download_time = datetime.datetime.now() - begin_time
    logging.info('Time to download: {}'.format(download_time.total_seconds()))
    for video_id, entry in audio_dict.items():
        example = [1 if is_positive_example(entry.labels, labels_set) else 0]
        count += 1 if example[0] == 1 else 0
        for feature in features_to_extract:
            extracted_feature = extract_feature(
                dest_dir, video_id, feature, scaled)
            if extracted_feature is None:
                continue
            example.append(extracted_feature)
        current_time = (datetime.datetime.now() - begin_time).total_seconds()
        logging.info((vid_count, current_time))
        vid_count += 1
        dataset.append(example)
    logging.info('There are {} positive examples'.format(count))
    current_time = datetime.datetime.now() - begin_time
    extract_time = (current_time - download_time)
    logging.info(
        'Time to extract features: {}'.format(extract_time.total_seconds()))
    columns = ['label'] + features_to_extract
    datasetdf = pd.DataFrame(dataset, columns=columns)
    current_time = datetime.datetime.now() - begin_time
    df_time = (current_time - extract_time)
    logging.info('Time to create dataframe: {}'.format(df_time.total_seconds()))
    end_time = datetime.datetime.now()
    logging.info('Total time to produce dataframe: {}'.format(
        (end_time - begin_time).total_seconds()))
    return datasetdf


def output_csv(src_dir, dest_dir, filename, labels, features_to_extract,
               scaled=True, redo=False):
    df = output_df(src_dir, dest_dir, filename, labels, features_to_extract,
                   scaled, redo)
    df_to_csv(dest_dir, filename, df)


def df_to_csv(dest_dir, filename, dataframe):
    path = join(dest_dir, filename + '_features.csv')
    dataframe.to_csv(path)


def main(argv):
    """Configures the output location using command line arguments.

    Uses command line arguments to configure this script to output a csv file
    based on specific source directory and a specified output directory. It also
    tells the script to redownload the YouTube videos based on the user input.

    Args:
        argv: A list containing the path this script after the build process.
    """
    output_csv(FLAGS.src_dir, FLAGS.dest_dir, FLAGS.filename, FLAGS.labels,
               FLAGS.features, FLAGS.scaled, FLAGS.redo)


if __name__ == "__main__":
    app.run(main)
