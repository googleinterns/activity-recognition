# Snore Detection Android App
## Version: 1.3.0
### Update date: 2020.08.17
### Author: Jerry Wu

This is a sample android app that utilizes snoring detection model to measure and track snoring.
It allows user to record and predict snoring that returns a list of snoring confidence score.

<br>

**Update from 1.2.0**:
- Fixed mis-labelling silence as snoring.
- Retrained model: Collected recorded audio data and add it into model training. New model becomes more robust in Android recorded environment.


<br>

In order to have this working, you need to download [vggish_feature_extraction_model.tflite](https://drive.google.com/file/d/10YgZ48mMkdLgJ_wzsMqG5nnNOZs9mkaC/view?usp=sharing) into app/src/main/assets folder.

**Apk** could also be downloaded directly from [link](https://drive.google.com/file/d/1ry3zAC2UpMDlNe-Sc-kR3dYZeunnoOjd/view?usp=sharing).

Please use following command to install apk:
**adb install -t \<app name>**


-----------------------------------------------
Copyright © 2020 by Jerry Wu
All rights reserved.
