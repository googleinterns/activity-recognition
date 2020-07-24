# Snore Detection Android App
## Version: 1.1.0
### Update date: 2020.07.24
### Author: Jerry Wu

This is a sample android app that utilizes snoring detection model to measure and track snoring.
It allows user to record and predict snoring that returns a list of snoring confidence score.

<br>

**Update from 1.0.0**:
- Translated all python codes into java codes
  - Removed use of Chaquopy
- Implemented python functions in java, including:
  - Fft.rfft, real fast fourier transform
  - Np.stride change
  - Clip
  - etc......


<br>

In order to have this working, you need to download [vggish_feature_extraction_model.tflite](https://drive.google.com/file/d/10YgZ48mMkdLgJ_wzsMqG5nnNOZs9mkaC/view?usp=sharing) into app/src/main/assets folder.

**Apk** could also be downloaded directly from [link](https://drive.google.com/file/d/1SHrmHW1oSOlUO7xv4kW5wPjl9O-wUMBQ/view?usp=sharing).

Please use following command to install apk:
**adb install -t \<app name>**


-----------------------------------------------
Copyright © 2020 by Jerry Wu
All rights reserved.
