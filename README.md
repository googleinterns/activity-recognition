# Activity Recognition Open Source Project

Note: this is not an official Google product.

## Xingjian (Jerry) Wu's Project:
### Snoring Prediction
Please check detailed information inside folder _JerryProject\_SnorePrediction_.

## Carver Forbe's Project:
### Gunshot Prediction
1. To build download and install Bazel, Librosa, Python (version 3), and youtube-dl
2. Navigate to activity-recognition (cd activity-recognition)
3. bazel build //python/location/lbs/activity/audioset/dataprocessing:audio_processing
4. bazel-bin/python/location/lbs/activity/audioset/dataprocessing/audio_processing -d DATASETS [DATASETS ...] [-r] -l LABELS [LABELS ...]

