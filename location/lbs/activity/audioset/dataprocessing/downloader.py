"""Helper functions for downloading audio from YouTube videos.

The functions in this script utilize youtube-dl to download videos from YouTube,
convert them into audio files, and chop them into segments based on specific
start and end times in seconds from the AudioSet csv file.
"""
import youtube_dl
from pydub import AudioSegment
import os
from os.path import join, isfile, isdir
import shutil
from os import listdir
from absl import logging
from audioset_helper import check_dir


def download_from_list(dest_dir, audio_dict, redo):
    """Downloads and trims YouTube audio to the label start and end time.

    Iterates through the key-value pairs in the dictionary and downloads each
    YouTube video unless the video is unavailable or made private. In those
    cases, it skips over the failed download. For each failed download, the
    video_id is stored in a list, failed_downloads. Then each video_id is
    removed from the audio_dict dictionary.

    Args:
        dest_dir: Path to the parent directory where the downloaded videos are
            to be stored.
        audio_dict: A dictionary with video_id, AudioSetEntry key-value pairs
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
    logging.info('{} examples successfully downloaded'.format(len(audio_dict)))


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
    print('downloading video')
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
