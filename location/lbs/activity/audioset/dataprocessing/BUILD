load("@rules_python//python:defs.bzl", "py_binary", "py_library")

py_binary(
  name = "audio_processing",
  srcs = ["audio_processing.py"],
  data = ["//location/lbs/activity/audioset/dataprocessing:example_src_dir/balanced_train_segments.csv",
   "//location/lbs/activity/audioset/dataprocessing:example_src_dir/eval_segments.csv",
   "//location/lbs/activity/audioset/dataprocessing:example_src_dir/ontology.json",
   "//location/lbs/activity/audioset/dataprocessing:example_src_dir/unbalanced_train_segments.csv"],
)
