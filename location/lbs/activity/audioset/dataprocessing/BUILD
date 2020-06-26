load("@rules_python//python:defs.bzl", "py_binary", "py_library")

py_binary(
  name = "audio_processing",
  srcs = ["audio_processing.py"],
  data = ["//python/location/lbs/activity/audioset/dataprocessing:ReferDoc/balanced_train_segments.csv",
   "//python/location/lbs/activity/audioset/dataprocessing:ReferDoc/eval_segments.csv",
   "//python/location/lbs/activity/audioset/dataprocessing:ReferDoc/ontology.json",
   "//python/location/lbs/activity/audioset/dataprocessing:ReferDoc/unbalanced_train_segments.csv"],
)
