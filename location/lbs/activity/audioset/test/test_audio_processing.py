import datetime
import unittest
from unittest import TestCase
from ..dataprocessing import audio_processing as ap
from os import listdir
from os.path import isfile, isdir

class AudioProcessingTest(TestCase):

    def test_parse_metadata(self):
        src_dir = 'location/lbs/activity/audioset/dataprocessing/example_src_dir'
        dest_dir = 'location/lbs/activity/audioset/dataprocessing/example_dest_dir'
        filename = 'balanced_train_segments'
        audio_list, label_dict = ap.parse_metadata(src_dir, filename)

        video_id = audio_list[0][0]
        start_time = audio_list[0][1]
        end_time = audio_list[0][2]
        self.assertEqual(video_id, '--PJHxphWEs')
        self.assertEqual(start_time, 30)
        self.assertEqual(end_time, 40)

        video_id = audio_list[-1][0]
        start_time = audio_list[-1][1]
        end_time = audio_list[-1][2]
        self.assertEqual(video_id, 'zzya4dDVRLk')
        self.assertEqual(start_time, 30)
        self.assertEqual(end_time, 40)


    def test_download_from_list(self):
        begin_time = datetime.datetime.now()
        src_dir = 'location/lbs/activity/audioset/dataprocessing/example_src_dir'
        dest_dir = 'location/lbs/activity/audioset/dataprocessing/example_dest_dir'
        filename = 'balanced_train_segments'
        audio_list, label_dict = ap.parse_metadata(src_dir, filename)
        downloaded_audio = []
        ap.download_from_list(dest_dir, audio_list, downloaded_audio, False)
        self.assertGreater(len(downloaded_audio), 0)
        end_time = datetime.datetime.now()
        print(end_time - begin_time)


if __name__ == '__main__':
    unittest.main()