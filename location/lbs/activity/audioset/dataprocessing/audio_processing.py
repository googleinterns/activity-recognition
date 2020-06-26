"""
Script to process a csv file from AudioSet into a csv file with features.

Each row of the inputted csv is parsed for the video_id, start_time_seconds,
end_time_seconds, and labels
The downloaded audios would be clipped into the labelled 10 seconds segments
The audio segments are then processed using Librosa to extract features

Usage
Standalone script:
python audio_processing.py --src_dir PATH_TO_INPUT_FILES --des_dir DEST_DIR PATH_TO_OUTPUT_FILES_--audioset_csv CSV FILE --labels LABELS --redo BOOLEAN TO REDOWNLOAD THE YOUTUBE FILES
example:
python audio_processing.py --src_dir location/lbs/activity/audioset/dataprocessing/example_src_dir --des_dir location/lbs/activity/audioset/dataprocessing/example_dest_dir_--audioset_csv balanced_train_segments --labels "Gunshot, gunfire" --False

Build with bazel:
bazel build location/lbs/activity/audioset/dataprocessing:audio_processing
bazel-bin/location/lbs/activity/audioset/dataprocessing/audio_processing --src_dir location/lbs/activity/audioset/dataprocessing/example_src_dir --des_dir location/lbs/activity/audioset/dataprocessing/example_dest_dir_--audioset_csv balanced_train_segments --labels "Gunshot, gunfire" --False


--src_dir is the path to the directory containing the input files
(audioset csv file ontology.json)
--dest_dir is the path to the directory containing the output files
(output csv file)
--labels included in the AudioSet dataset can be used (i.e "Gunshot, gunfire")

Performance: to confirm the presence of, extract features from, and output to a
    csv file, takes about ___minutes for around 20,000 samples.
"""

from __future__ import unicode_literals
import youtube_dl
import csv
import json
from pydub import AudioSegment
import os
import shutil
from os import listdir
from os.path import isfile, join
import librosa
from absl import app
from absl import flags
from absl import logging
import datetime

FLAGS = flags.FLAGS


def setup_abseil():
    """Sets up command line arguments using Abseil.

    Defines --audioset_csv, --labels, --redo, --src_dir, and --dest_dir as flags
    """
    flags.DEFINE_string('audioset_csv', 'balanced_train_segments',
                        'The csv file of a dataset provided by the audioset')
    flags.DEFINE_multi_string('labels', 'Gunshot, gunfire',
                              'The set of labels identified by the audioset to '
                              'treat as positive')
    flags.DEFINE_bool('redo', False,
                      'Whether to redownload the YouTube videos in the included'
                      ' csv file')
    flags.DEFINE_string('src_dir',
                        'location/lbs/activity/audioset/dataprocessing/example_src_dir',
                        'The location of the audioset csv file(s) and the '
                        'ontology.json file')
    flags.DEFINE_string('dest_dir',
                        'location/lbs/activity/audioset/dataprocessing/example_dest_dir',
                        'The location of the csv file and downloaded YouTube '
                        'videos')


def parse_metadata():
    """Parses metadata and labels from the audioset csv.

    Iterates through the metadata

    Returns:
        A tuple with the first element being a list of lists containing a
        video_id, start_time (in seconds), and end_time (in seconds). This list
        is generated from the inputted csv. The second element is a dictionary
        with video_id, list of labels key, value pairs
        For example:

        ([[video_id1, start_time1, end_time1],
        [video_id2, start_time2, end_time2],
        [video_id3, start_time3, end_time3]],
        {video_id1 : [label1, label3],
        video_id12: [label2]
        video_id3 : [label2, label3]})
    """
    csv_path = FLAGS.src_dir + '/' + FLAGS.audioset_csv + '.csv'
    audio_list = []
    label_dict = {}
    with open(csv_path) as csvfile:
        csv_rows = csv.reader(csvfile, delimiter=',')
        for row in csv_rows:
            if row[0][0] == '#':
                continue
            new_row = []
            for element in row:
                new_row.append(element.strip())
            label_list = []
            count = 0
            for element in new_row:
                if count == 0:
                    video_id = element
                elif count == 1:
                    start_time = float(element)
                elif count == 2:
                    end_time = float(element)
                elif count >= 3:
                    if element[0] == '"':
                        element = element[1:]
                    if element[-1] == '"':
                        element = element[:-1]
                    label_list.append(element)
                count += 1
            audio_list.append([video_id, start_time, end_time])
            label_dict[video_id] = label_list
    return audio_list, label_dict


def label_id_search(src_dir, label):
    """Translates a label in plain English to the syntax used by audioset

    Searches the ontology.json file in the source directory for the label in
    plain English and returns its audioset syntax.

    Args:
        src_dir: Path to a directory where the ontology.json is expected to be
        label: A label in plain English

    Returns:
        A string of the label in audioset syntax (the syntax used in the
        csv file from the audioset).
    """
    with open(src_dir + '/' + "ontology.json") as f:
        label_info = json.load(f)
    for info in label_info:
        if info["name"] == label:
            return info["id"]
    return None


def download_from_list(audio_list, downloaded_audio):
    """Downloads YouTube videos in an inputted list

    Iterates through the list and downloads each YouTube video unless the video
    is unavailable or made private. In those cases, is skips over the failed
    download. For each successful download, the video_id is stored in a list,
    downloaded_audio.

    Args:
        audio_list: a list of of YouTube video_ids to download
        downloaded_audio: a list where successfully downloaded YouTube videos'
            video_id will be stored.
    """
    for video_id, start_time, end_time in audio_list:
        success = download(video_id, downloaded_audio)
        if success:
            chop_audio(video_id, start_time, end_time)


def download(video_id, downloaded_audio):
    """Downloads a YouTube video using its video_id

    Calls the youtube-dl tool to download a YouTube video by its video_id and
    convert the video into an audio file. If the download is successful,
    it stores the video_id in a list, downloaded_video.

    Args:
        video_id: the video_id of the YouTube video to be downloaded.
        downloaded_audio: a list where successfully downloaded YouTube videos'
            video_id will be stored.

    Returns:
        A boolean of whether the download was successful or not. Returns false
        if the video was already downloaded and redo is False.
    """
    try:
        os.mkdir(FLAGS.dest_dir)
    except OSError as error:
        logging.error(error)

    # check to see if the video_id has already been downloaded and skip the
    # download if it is already downloaded unless the -r, --redo flag has been
    # passed
    already_downloaded = isfile(FLAGS.dest_dir + '/yt_videos'
                                + '/sliced_' + video_id + '.wav')
    if already_downloaded and not FLAGS.redo:
        downloaded_audio.append(video_id)
        logging.info('Already Downloaded')
        return False

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredquality': '192',
        }],
        # Force the file naming of outputs.
        'outtmpl': FLAGS.dest_dir + '/tmp/' + video_id + '.%(ext)s'
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download(['https://www.youtube.com/watch?v=' + video_id])
            downloaded_audio.append(video_id)
            logging.info('Download Complete')
            return True
        except:
            logging.info('Downloading Failed.')
            return False


def chop_audio(video_id, start_time, end_time):
    """Chops an audio file into a segment by a given start_time and end_time.

    Using a specific start_time and end_time in seconds, it chops the audio file
    from the downloaded YouTube video, and then removes the original audio file.

    Args:
        video_id: The video_id of the audio from the downloaded YouTube video.
        start_time: The start time in seconds of the labelled video segment.
        end_time: The end time in seconds of the labelled video segment.

    TODO: breaking if there is more than one file in /tmp
    """
    onlyfile = [f for f in listdir(FLAGS.dest_dir + '/tmp') if isfile(
        join(FLAGS.dest_dir + '/tmp', f))][0]
    if onlyfile.endswith('.m4a'):
        total = AudioSegment.from_file(
            FLAGS.dest_dir + '/tmp/' + video_id + '.m4a', 'm4a')
    elif onlyfile.endswith('.opus'):
        total = AudioSegment.from_file(
            FLAGS.dest_dir + '/tmp/' + video_id + '.opus', codec='opus')
    else:
        shutil.rmtree(FLAGS.dest_dir + '/tmp/')
        return None
    sliced = total[start_time * 1000: end_time * 1000]
    sliced.export(FLAGS.dest_dir + '/yt_videos/sliced_' + video_id
                  + '.wav', format='wav')
    shutil.rmtree(FLAGS.dest_dir + '/tmp/')
    logging.info("chopped audio")


def extract_features_from_list(downloaded_audio, vid_id_to_features):
    """Extracts features from each audio file given a list of video_ids.

    Given a list of video_ids, it extracts features using the Librosa libray
    from each corresponding audio file and stores the features in a dictionary
    with a video_id, dictionary key, value pair, with the dictionary value being
    comprised of feature name, feature list key, value pairs. Using Librosa it
    extracts the following features: chroma_stft, chroma_cqt, chroma_cens,
    melspectogram, mfcc, rms, spectral_centroid, spectral_bandwidth,
    spectral_contrast, spectral_flatness, spectral_rolloff, poly_features,
    tonnetz, zero_crossing_rate

    Args:
        downloaded_audio: A list of video_ids of YouTube videos that were
            successfully downloaded.
        vid_id_to_features: A dictionary with video_id, dictionary pairs with
            the dictionary values being feature name, feature list key, value
            pairs.
    """
    for video_id in downloaded_audio:
        extract_features(video_id, vid_id_to_features)


def extract_features(video_id, vid_id_to_features):
    """Extracts features from a specific audio file given a video_id.

   Given a specific of video_id, it extracts features using the Librosa libray
   from the corresponding audio file and stores the features in a dictionary
   with a video_id, dictionary key, value pair, with the dictionary value being
   comprised of feature name, feature list key, value pairs. Using Librosa it
   extracts the following features: chroma_stft, chroma_cqt, chroma_cens,
   melspectogram, mfcc, rms, spectral_centroid, spectral_bandwidth,
   spectral_contrast, spectral_flatness, spectral_rolloff, poly_features,
   tonnetz, zero_crossing_rate

   Args:
       video_id: The video_id of specific audio file from which features are to
           be extracted.
       vid_id_to_features: A dictionary with video_id, dictionary pairs with the
           dictionary values being feature name, feature list key, value pairs.

   Returns:
       A boolean of whether the extracting features was successful
   """
    feature_dict = {}
    if not isfile(FLAGS.dest_dir + '/yt_videos/sliced_' + video_id + '.wav'):
        return False
    y, sr = librosa.load(FLAGS.dest_dir + '/yt_videos/sliced_'
                         + video_id + '.wav')
    feature_dict["chroma_stft"] = librosa.feature.chroma_stft(y, sr)
    feature_dict["chroma_cqt"] = librosa.feature.chroma_cqt(y, sr)
    feature_dict["chroma_cens"] = librosa.feature.chroma_cens(y, sr)
    feature_dict["melspectrogram"] = librosa.feature.melspectrogram(y, sr)
    feature_dict["mfcc"] = librosa.feature.mfcc(y, sr)
    feature_dict["rms"] = librosa.feature.rms(y)
    feature_dict["spectral_centroid"] = librosa.feature.spectral_centroid(y, sr)
    feature_dict["spectral_bandwidth"] = librosa.feature.spectral_bandwidth(y,
                                                                            sr)
    feature_dict["spectral_contrast"] = librosa.feature.spectral_contrast(y, sr)
    feature_dict["spectral_flatness"] = librosa.feature.spectral_flatness(y)
    feature_dict["spectral_rolloff"] = librosa.feature.spectral_rolloff(y, sr)
    feature_dict["poly_features"] = librosa.feature.poly_features(y, sr)
    feature_dict["tonnetz"] = librosa.feature.tonnetz(y, sr)
    feature_dict["zero_crossing_rate"] = librosa.feature.zero_crossing_rate(y,
                                                                            sr)
    vid_id_to_features[video_id] = feature_dict
    logging.info("extracted features")
    return True


def create_csv_from_list(vid_id_to_features, labels_dict):
    """Creates a csv from a dictionary of video_id, feature dictionary

    Creates a csv file to input into TensorFlow with the following structure:
    is_gunshot, video_id, feature list.

    Args:
        vid_id_to_features: A dictionary with video_id, dictionary pairs with
            the dictionary values being feature name, feature list key,
            value pairs.
        labels_dict: a dictionary of video_id list of labels key value pairs
    """
    with open(FLAGS.dest_dir + "/tfdataset.csv", 'w') as outcsv:
        csv_writer = csv.writer(outcsv, delimiter=':',
                                quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(
            ['label', 'video_id', 'chroma_stft', 'chroma_cqt', 'chroma_cens',
             'melspectrogram', 'mfcc', 'rms', 'spectral_centroid',
             'spectral_bandwidth', 'spectral_contrast', 'spectral_flatness',
             'spectral_rolloff', 'poly_features', 'tonnetz',
             'zero_crossing_rate'])
        labels_set = label_list_to_set(FLAGS.labels)
        for key in vid_id_to_features:
            video_id = key
            feature_dict = vid_id_to_features.get(key)
            # TODO: Enable selection of labels to set as true
            label = label_selection(labels_dict.get(video_id), labels_set)
            csv_writer.writerow([label] + [video_id] + [delete_outer_list(
                feature_dict.get('chroma_stft').tolist())]
                                + [delete_outer_list(
                feature_dict.get('chroma_cqt').tolist())]
                                + [delete_outer_list(
                feature_dict.get('chroma_cens').tolist())]
                                + [delete_outer_list(
                feature_dict.get('melspectrogram').tolist())]
                                + [delete_outer_list(
                feature_dict.get('mfcc').tolist())]
                                + [delete_outer_list(
                feature_dict.get('rms').tolist())]
                                + [delete_outer_list(
                feature_dict.get('spectral_centroid').tolist())]
                                + [delete_outer_list(
                feature_dict.get('spectral_bandwidth').tolist())]
                                + [delete_outer_list(
                feature_dict.get('spectral_contrast').tolist())]
                                + [delete_outer_list(
                feature_dict.get('spectral_flatness').tolist())]
                                + [delete_outer_list(
                feature_dict.get('spectral_rolloff').tolist())]
                                + [delete_outer_list(
                feature_dict.get('poly_features').tolist())]
                                + [delete_outer_list(
                feature_dict.get('tonnetz').tolist())]
                                + [delete_outer_list(
                feature_dict.get('zero_crossing_rate').tolist())])


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
    translated_labels = []
    for label in labels_list:
        translated_labels.append(label_id_search(FLAGS.src_dir, label))
    for label in translated_labels:
        if label in labels_set:
            return 1
        else:
            return 0


def label_list_to_set(translated_labels_list):
    """Converts a list of labels to a set of labels.

    Converts a list of labels in the format found in the audioset csv to a set
    of labels, then returns that set.

    Args:
        translated_labels_list: a list of labels in the format found in the
            audioset csv passed in as command line arguments

    Returns:
        A set of labels in the format found in the audioset csv
    """
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


def output_csv_linear():
    """Outputs csv file with a label, metadata, and features from audioset csv.

    Parses the audioset csv, then downloads the YouTube videos and stores the
    audio from the segment from start time to end time. Then extracts features
    using Librosa from the successfully downloaded audio and stores the label,
    video_id, and features in a csv file.
    """
    begin_time = datetime.datetime.now()
    audio_list, label_dict = parse_metadata()
    downloaded_audio = []
    download_from_list(audio_list, downloaded_audio)
    vid_id_to_features = {}
    extract_features_from_list(downloaded_audio, vid_id_to_features)
    create_csv_from_list(vid_id_to_features, label_dict)
    logging.info(datetime.datetime.now() - begin_time)


def main(argv):
    """Configures the output location using command line arguments.

    Uses command line arguments to configure this script to output a csv file
    based on specific source directory and a specified output directory. It also
    tells the script to redownload the YouTube videos based on the user input.

    Args:
        argv: A list containing the path this script after the build process.
    """
    setup_abseil()
    output_csv_linear()


if __name__ == "__main__":
    app.run(main)
