load("@rules_java//java:defs.bzl", "java_binary", "java_lite_proto_library", "java_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

proto_library(
  name = "incomplete_example_proto",
  srcs = ["java/com/google/location/lbs/activity/audioset/dataprocessing/incomplete-example.proto"],
  deps = ["@com_google_protobuf//:timestamp_proto"],
)

java_proto_library(
  name = "incomplete_example_java_proto",
  deps = [":incomplete_example_proto"],
)

java_binary(
  name = "csv_parser_java",
  srcs = ["java/com/google/location/lbs/activity/audioset/dataprocessing/CsvParser.java"],
  main_class = "CsvParser",
  deps = [":incomplete_example_java_proto"],
  data = ["//java/com/google/location/lbs/activity/audioset/dataprocessing/datasets:balanced_train_segments.csv", "//java/com/google/location/lbs/activity/audioset/dataprocessing/datasets:eval_segments.csv"],
)

java_binary(
  name = "video_downloader_java",
  srcs = ["java/com/google/location/lbs/activity/audioset/dataprocessing/VideoDownloader.java"],
  main_class = "VideoDownloader",
  deps = [":csv_parser_java"],
)

java_binary(
  name = "feature_extractor_java",
  srcs = ["java/com/google/location/lbs/activity/audioset/dataprocessing/FeatureExtractor.java"],
  main_class = "FeatureExtractor",
  deps = [":video_downloader_java"],
)