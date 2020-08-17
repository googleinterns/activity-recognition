import datetime
import unittest
from unittest import TestCase
from ..dataprocessing import audio_processing as ap


class AudioProcessingTest(TestCase):

    def test_parse_metadata(self):
        src_dir = 'location/lbs/activity/audioset/dataprocessing/example_src_dir'
        filename = 'balanced_train_segments'
        audio_dict = ap.parse_metadata(src_dir, filename)

        video_id = audio_dict.get('--PJHxphWEs').video_id
        start_time = audio_dict.get('--PJHxphWEs').start_time
        end_time = audio_dict.get('--PJHxphWEs').end_time
        self.assertEqual(video_id, '--PJHxphWEs')
        self.assertEqual(start_time, 30)
        self.assertEqual(end_time, 40)

        video_id = audio_dict.get('zzya4dDVRLk').video_id
        start_time = audio_dict.get('zzya4dDVRLk').start_time
        end_time = audio_dict.get('zzya4dDVRLk').end_time
        self.assertEqual(video_id, 'zzya4dDVRLk')
        self.assertEqual(start_time, 30)
        self.assertEqual(end_time, 40)

    def test_download_from_list(self):
        begin_time = datetime.datetime.now()
        src_dir = 'location/lbs/activity/audioset/dataprocessing/example_src_dir'
        dest_dir = 'location/lbs/activity/audioset/dataprocessing/example_dest_dir'
        filename = 'balanced_train_segments'
        audio_dict = ap.parse_metadata(src_dir, filename)
        ap.download_from_list(dest_dir, audio_dict, False)
        self.assertGreater(len(audio_dict), 0)
        end_time = datetime.datetime.now()
        print(end_time - begin_time)


if __name__ == '__main__':
    unittest.main()
