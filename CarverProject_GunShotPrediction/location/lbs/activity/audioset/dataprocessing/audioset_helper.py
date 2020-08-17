"""Helper functions that are specific to the way AudioSet organizes datasets.

AudioSet dataset csv files contain a header and organizes its rows like so:
YouTube ID, start_seconds, end_seconds, positive_labels

The file contains functions and classes to parse for the video_id,
start_time_seconds, end_time_seconds, and labels, representing each row and its
metadata as an AudioSetEntry and its labels as Label objects.

This file also contains other helper methods to confirm and create directories,
and build sets of labels and video_id to Label lists key-value containing
dictionaries.
"""
import csv
import json
import os
from os.path import isdir, join
from absl import logging


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


class Label:
    """Class to hold the metadata of each label in the dataset.

    Attributes:
        name: plain English name of the label.
        label_id: id of the label as found in the ontology.json file.
        description: The description of the label.
        child_ids: a list of the label's children's ids.
    """

    def __init__(self, name, label_id, description, child_ids):
        """Inits Label with name, id, description, and child_ids."""
        self.name = name
        self.label_id = label_id
        self.description = description
        self.child_ids = child_ids


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
        A dictionary with video_id, AudioSetEntry key-value pairs.
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
                label = label.replace('"', '')
                label_list.append(label)
            audio_dict[video_id] = AudioSetEntry(
                video_id, start_time, end_time, label_list)
    logging.info('The set has {} examples'.format(len(audio_dict)))
    return audio_dict


def get_label_dict(src_dir):
    """Builds a dictionary of label names to Label objects.

    Args:
        src_dir: Path to a directory where the ontology.json is expected to be.

    Returns:
        A dictionary of label names to Label objects.
    """
    label_dict = {}
    with open(src_dir + '/' + "ontology.json") as f:
        label_json = json.load(f)
    for label in label_json:
        name = label["name"]
        label_id = label["id"]
        children = label["child_ids"]
        description = label["description"]
        label_object = Label(name, label_id, description, children)
        label_dict[name] = label_object
    return label_dict


def label_list_to_set(labels_list, label_dict):
    """Converts a list of labels to a set of labels.

    Translates a list of plain English labels names into label ids and adds the
    child_ids of each label to the list. Then, converts the list of label ids
    to a set of labels ids and returns that set.

    Args:
        labels_list: a list of label names passed in as command line arguments.
        label_dict: a dictionary of label names to Label objects.

    Returns:
        A set of labels in the format found in the audioset csv
    """
    translated_labels_list = []
    for name in labels_list:
        translated_labels_list.append(label_dict.get(name).label_id)
        for label_id in label_dict.get(name).child_ids:
            translated_labels_list.append(label_id)
    labels_set = set(translated_labels_list)
    return labels_set


def check_dir(directory):
    """Checks if specified directory exists and creates it if it does not exist.

    Checks if a particular directory exists. If the specified directory does not
    exist, it creates the directory. All errors associated with creating the
    directory are logged.

    Args:
        directory: Path to a directory that is checked to see if it exists.
    """
    if not isdir(directory):
        try:
            os.mkdir(directory)
        except OSError as error:
            logging.error(error)