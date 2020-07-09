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
import youtube_dl
import csv
import json
from pydub import AudioSegment
import os
import shutil
from os import listdir
from os.path import isfile, isdir, join
import librosa
from absl import app
from absl import flags
from absl import logging
import datetime
import pandas as pd
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string('filename', 'balanced_train_segments',
                    'The csv file of a dataset provided by the audioset')
flags.DEFINE_multi_string('labels', 'Gunshot, gunfire',
                          'The set of labels identified by the audioset to '
                          'treat as positive')
flags.DEFINE_multi_string('features', 'mfcc',
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


class AudioSetEntry:
    """Class to hold the metadata of each example in the dataset.

    Attributes:
        video_id: video id of the YouTube video.
        start_time: start time in seconds of the labelled segment.
        end_time: end time in seconds of the labelled segment.
        labels: list of labels that the audio segment is classified by.
    """

    def __init__(self, video_id, start_time, end_time, labels):
        """Inits AudioSetEntry with video_id, start and end times, labels."""
        self.video_id = video_id
        self.start_time = start_time
        self.end_time = end_time
        self.labels = labels


def parse_metadata(src_dir, filename):
    """Parses metadata and labels from the audioset csv.

    Iterates through the csv files from audioset and extracts the metadata:
    video_id, start_time, end_time, and labels, creating an AudioSetEntry object
    with the metadata as instance data.

    Args:
        src_dir: Path to a directory where the audioset csv file is expected to
            be found.
        filename: A csv file without the .csv file extension

    Returns:
        A dictionary with video_id, AudioSetEntry key, value pairs.
    """
    csv_path = join(src_dir, filename + '.csv')
    audio_dict = {}
    with open(csv_path) as csv_file:
        fieldnames = ['video_id', 'start_time', 'end_time']
        csv_rows = csv.DictReader(filter(lambda row: row[0] != '#', csv_file),
                                  fieldnames=fieldnames)
        for row in csv_rows:
            label_list = []
            video_id = row.get('video_id')
            start_time = float(row.get('start_time').strip())
            end_time = float(row.get('end_time').strip())
            for label in row.get(None):
                label = label.strip()
                if label[0] == '"':
                    label = label[1:]
                if label[-1] == '"':
                    label = label[:-1]
                label_list.append(label)
            audio_dict[video_id] = AudioSetEntry(video_id, start_time, end_time,
                                                 label_list)
    return audio_dict


def get_label_dict(src_dir):
    """Builds a dictionary of ids to labels as seen in the ontology.json file.

    Args:
        src_dir: Path to a directory where the ontology.json is expected to be.

    Returns:
        A dictionary of plain english ids to labels as seen in ontology.json.
    """
    label_dict = {}
    with open(src_dir + '/' + "ontology.json") as f:
        label_json = json.load(f)
    for label in label_json:
        name = label["name"]
        id = label["id"]
        label_dict[name] = id
    return label_dict


def get_failed_downloads(dest_dir):
    """Converts file with a list of video_ids that failed to download to a set.

    Reads through a failed_downloads.txt and converts the list of videos_ids of
    YouTube videos that failed to download into a set.

    Args:
        dest_dir: Path to a directory where the failed_downloads.txt is expected
            to be

    Returns:
        A set of video_ids that failed to download.
    """
    check_dir(dest_dir)
    failed_download_set = set()
    path = join(dest_dir, 'failed_downloads.txt')
    if isfile(path):
        with open(path, 'r') as file:
            for line in file:
                video_id = line.strip()
                failed_download_set.add(video_id)
    return failed_download_set


def store_failed_download(dest_dir, video_id):
    """Stores the video_id of a YouTube video that failed to download in a file.

    Finds or creates a text file dubbed failed_downloads.txt and writes a line
    containing the video_id of a YouTube video that failed to download.

    Args:
        dest_dir: Path to a directory where the failed_downloads.txt is expected
            to be.
        video_id: The video_id of a YouTube video that failed to download.
    """
    check_dir(dest_dir)
    path = join(dest_dir, 'failed_downloads.txt')
    with open(path, "a") as file:
        file.write(video_id)
        file.write('\n')


def check_dir(dir):
    """Checks if specified directory exists and creates it if it does not exist.

    Checks if a particular directory exists. If the specified directory does not
    exist, it creates the directory. All errors associated with creating the
    directory are logged.

    Args:
        dir: Path to a directory that is checked to see if it exists.
    """
    if not isdir(dir):
        try:
            os.mkdir(dir)
        except OSError as error:
            logging.error(error)


def download_from_list(dest_dir, audio_dict, redo):
    """Downloads and trims YouTube audio to the label start and end time.

    Iterates through the key, value pairs in the dictionary and downloads each
    YouTube video unless the video is unavailable or made private. In those
    cases, it skips over the failed download. For each failed download, the
    video_id is stored in a list, failed_downloads. Then each video_id is
    removed from the audio_dict dictionary.

    Args:
        dest_dir: Path to the parent directory where the downloaded videos are
            to be stored.
        audio_dict: A dictionary with video_id, AudioSetEntry key, value pairs
        redo: A boolean of specifying whether to re-download all the YouTube
            videos.
    """
    failed_download_set = get_failed_downloads(dest_dir)
    failed_downloads = []
    for video_id, entry in audio_dict.items():
        if video_id in failed_download_set:
            failed_downloads.append(video_id)
            continue
        success = download(dest_dir, video_id, failed_downloads, redo)
        start_time = entry.start_time
        end_time = entry.end_time
        if success:
            chop_audio(dest_dir, video_id, start_time, end_time)
    for video_id in failed_downloads:
        del audio_dict[video_id]


def download(dest_dir, video_id, failed_downloads, redo):
    """Downloads a YouTube video using its video_id

    Calls the youtube-dl tool to download a YouTube video by its video_id and
    convert the video into an audio file. If the download is successful,
    it stores the video_id in a list, downloaded_video.

    Args:
        dest_dir: Path to the parent directory where the downloaded videos are
            to be stored.
        video_id: the video_id of the YouTube video to be downloaded.
        failed_downloads: a list where failed downloaded YouTube videos'
            video_id will be stored.
        redo: A boolean of specifying whether to re-download all the YouTube
            videos.

    Returns:
        A boolean of whether the download was successful or not. Returns false
        if the video was already downloaded and redo is False.
    """
    check_dir(dest_dir)
    # check to see if the video_id has already been downloaded and skip the
    # download if it is already downloaded unless the -r, --redo flag has been
    # passed
    video_path = join(dest_dir, 'yt_videos', 'sliced_' + video_id + '.wav')
    already_downloaded = isfile(video_path)
    if already_downloaded and not redo:
        logging.info('Already Downloaded')
        return False

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredquality': '192',
        }],
        # Force the file naming of outputs.
        'outtmpl': join(dest_dir, 'tmp', video_id + '.%(ext)s')
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download(['https://www.youtube.com/watch?v=' + video_id])
            logging.info('Download Complete')
            return True
        except youtube_dl.DownloadError:
            store_failed_download(dest_dir, video_id)
            failed_downloads.append(video_id)
            logging.info('Downloading Failed.')
            return False


def chop_audio(dest_dir, video_id, start_time, end_time):
    """Chops an audio file into a segment by a given start_time and end_time.

    Using a specific start_time and end_time in seconds, it chops the audio file
    from the downloaded YouTube video, and then removes the original audio file.

    Args:
        dest_dir: Path to the parent directory where the downloaded videos are
            are to be stored.
        video_id: The video_id of the audio from the downloaded YouTube video.
        start_time: The start time in seconds of the labelled video segment.
        end_time: The end time in seconds of the labelled video segment.
    """
    tmp_path = join(dest_dir, 'tmp')
    if isdir(tmp_path):
        length = len(listdir(tmp_path))
        m4a_path = join(tmp_path, video_id + '.m4a')
        opus_path = join(tmp_path, video_id + '.opus')
        ogg_path = join(tmp_path, video_id + '.ogg')
        wav_path = join(dest_dir, 'yt_videos', 'sliced_' + video_id + '.wav')
        if isfile(m4a_path):
            video = AudioSegment.from_file(m4a_path)
            temp_path = m4a_path
        elif isfile(opus_path):
            video = AudioSegment.from_file(opus_path)
            temp_path = opus_path
        elif isfile(ogg_path):
            video = AudioSegment.from_file(ogg_path)
            temp_path = ogg_path
        else:
            if length == 0:
                shutil.rmtree(tmp_path)
            return None
        sliced = video[start_time * 1000: end_time * 1000]
        sliced.export(wav_path, format='wav')
        os.remove(temp_path)
        if length <= 1:
            shutil.rmtree(tmp_path)
        logging.info('chopped_audio')


def extract_feature(dest_dir, video_id, feature, scaled):
    """Extracts a feature from a specific audio file given a video_id.

   Given a specific of video_id, it extracts features using the Librosa libray
   from the corresponding audio file and stores the features in a dictionary
   with a video_id, dictionary key, value pair, with the dictionary value being
   comprised of feature name, feature list key, value pairs. Using Librosa it
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
           dictionary values being feature name, feature list key, value pairs.
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
        return False
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
    odd_ones = {'rms', 'spectral_flatness'}
    func = switcher.get(feature)
    if func is None:
        raise ValueError
    if feature in odd_ones:
        extracted_feature = func(y)
    else:
        extracted_feature = func(y, sr)
    if scaled:
        extracted_feature = np.mean(extracted_feature.T, axis=0)
    logging.info('extracted features')
    return extracted_feature


def label_selection(labels_list, labels_set):
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
            return 1
        else:
            return 0


def label_list_to_set(labels_list, label_dict):
    """Converts a list of labels to a set of labels.

    Translates a list of plain English labels into the syntax used in the input
    csv file from audioset. Then, converts the translated list of labels to a
    set of labels, then returns that set.

    Args:
        labels_list: a list of labels in plain English passed in as
            command line arguments
        label_dict: a dictionary of plain English labels to audioset syntax.

    Returns:
        A set of labels in the format found in the audioset csv
    """
    translated_labels_list = []
    for label in labels_list:
        translated_labels_list.append(label_dict.get(label))
    labels_set = set(translated_labels_list)
    return labels_set


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
    label_dict = get_label_dict(src_dir)  # plain English to audioset syntax
    labels_set = label_list_to_set(labels, label_dict)  # user specified labels
    audio_dict = parse_metadata(src_dir, filename)  # video_id to AudioSetEntry
    count = 0
    vid_count = 0
    download_from_list(dest_dir, audio_dict, redo)
    download_time = datetime.datetime.now() - begin_time
    logging.info('Time to download: {}'.format(download_time.total_seconds()))
    for video_id, entry in audio_dict.items():
        label = label_selection(entry.labels, labels_set)
        if label == 1:
            count += 1
        example = [label]
        for feature in features_to_extract:
            extracted_feature = extract_feature(dest_dir, video_id, feature,
                                                scaled)
            if extracted_feature is None:
                continue
            example.append(extracted_feature)
        curent_time = (datetime.datetime.now() - begin_time).total_seconds()
        logging.info((vid_count, curent_time))
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
               FLAGS.features_to_extract, FLAGS.scaled, FLAGS.redo)


if __name__ == "__main__":
    app.run(main)
