package com.google.location.lbs.activity.audioset.dataprocessing;

import java.io.File;
import java.io.IOException;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.io.FileWriter;
import com.google.location.lbs.activity.audioset.dataprocessing.IncompleteExample;

/**
 * CsvParser Class
 *
 * <p>CsvParser is a class for the reading of a dataset in a csv format.
 */
public final class CsvParser {

  private File csv;
  private File bashScript;
  private List<String> videoIdList;
  private Map<String, Example> idToProtoMap;

  private final boolean debug = true;

  /**
   * CsvParser constructor
   *
   * @param pathname the pathname for the csv file to be read
   * @throws IllegalArgumentException if the pathname is incorrect
   */
  public CsvParser(String pathname) {
    csv = new File(pathname);
    if (!csv.exists()) {
      throw new IllegalArgumentException(String.format("This file does not exist: %s", pathname));
    }
    bashScript = new File(pathname);
    if (!bashScript.exists()) {
      throw new IllegalArgumentException(String.format("This file does not exist: %s", pathname));
    }
    videoIdList = new ArrayList<>();
    idToProtoMap = new HashMap<>();
  }

  /** Tester Function to print 5 YouTube Links */
  public void printYouTubeLinks() {
    // prints youtube links
    Scanner scan = null;
    try {
      scan = new Scanner(csv);
      String line = scan.nextLine();
      while (line.charAt(0) == '#') {
        line = scan.nextLine();
      }
      // while (scan.hasNext()) {
      //   String videoId = line.split(",")[0];
      //   String url = "https://youtube.com/watch?v=" + videoId;
      //   System.out.println(url);
      //   line = scan.nextLine();
      // }
      for (int i = 0; i < 5; i++) {
        String videoId = line.split(",")[0];
        String url = "https://youtube.com/watch?v=" + videoId;
        System.out.println(url);
        line = scan.nextLine();
      }
    } catch (IOException io) {
      io.printStackTrace();
    } finally {
      if (scan != null) {
        scan.close();
      }
    }
  }

  /** Extracts metadata from dataset.csv file */
  public void csvToProtoMap() {
    if (debug) {System.out.println("in the csvToProtoMap method");}
    Scanner scan = null;
    try {
      scan = new Scanner(csv);
      int count = 0;
      String line = scan.nextLine();
      count++;
      while (line.charAt(0) == '#') {
        line = scan.nextLine();
        count++;
      }
      do {
        createProto(line);
        if (debug) {System.out.printf("At line %d\n", count);}
        line = scan.nextLine();
        count++;
      } while (scan.hasNext());
      createProto(line);
      if (debug) {System.out.printf("At line %d\n", count);}
    } catch (IOException io) {
      io.printStackTrace();
    } finally {
      if (scan != null) {
        scan.close();
      }
    }
  }

  /**
   * Creates a protocol buffer from the metadata in each line of the dataset.csv
   *
   * @param csvLine
   */
  public void createProto(String csvLine) {
    if (debug) {System.out.println("in the createProto method");}
    String[] lineArr = csvLine.split(",");
    String videoId = lineArr[0];
    videoIdList.add(videoId);
    String url = "https://youtube.com/watch?v=" + videoId;
    String bashLine = "youtube-dl --id -x --audio-format wav" + url + "\n";
    FileWriter writer = new FileWriter(bashScript);
    writer.write(bashLine);
    writer.flush();
    writer.close();
    int start = (int) Double.parseDouble(lineArr[1]);
    int end = (int) Double.parseDouble(lineArr[2]);
    String[] labelsList = Arrays.copyOfRange(lineArr, 3, lineArr.length);
    ArrayList<Example.Label> protoLabels = new ArrayList<>();
    int len = labelsList.length;
    for (int i = 0; i < len; i++) {
      labelsList[i] = labelsList[i].replace("\"", "");
      labelsList[i] = labelsList[i].trim();
      Example.Label protoLabel = Example.Label.newBuilder().setLabel(labelsList[i]).build();
      protoLabels.add(protoLabel);
    }
    Example proto =
      Example.newBuilder()
        .setVideoId(videoId)
        .setStart(start)
        .setEnd(end)
        .addAllLabels(protoLabels)
        .build();
    idToProtoMap.put(videoId, proto);
  }

  public Map<String, Example> getIdToProtoMap() {
    return idToProtoMap;
  }

  public List<String> getVideoIdList() {
    return videoIdList;
  }

  /**
   * Main function to test class' functionality (stop gap until I get junits working with bazel
   *
   * @param args command line arguments
   */
  public static void main(String[] args) {
    CsvParser parser;
    parser =
        new CsvParser(
            "java/com/google/location/lbs/activity/audioset/dataprocessing/datasets/balanced_train_segments.csv");
    parser.csvToProtoMap();
    String url = "https://youtube.com/watch?v=" + "--PJHxphWEs";
    if ("https://youtube.com/watch?v=--PJHxphWEs".equals(url)) {
      System.out.println("Passed");
    } else {
      System.out.println("Failed");
    }
    url = "https://youtube.com/watch?v=" + "2KwwQHit0-Q";
    if ("https://youtube.com/watch?v=2KwwQHit0-Q".equals(url)) {
      System.out.println("Passed");
    } else {
      System.out.println("Failed");
    }
    url = "https://youtube.com/watch?v=" + "zzya4dDVRLk";
    if ("https://youtube.com/watch?v=zzya4dDVRLk".equals(url)) {
      System.out.println("Passed");
    } else {
      System.out.println("Failed");
    }
    int start = parser.getIdToProtoMap().get("--PJHxphWEs").getStart();
    if (start == 30) {
      System.out.println("Passed");
    } else {
      System.out.println("Failed");
    }
    start = parser.getIdToProtoMap().get("2KwwQHit0-Q").getStart();
    if (start == 30) {
      System.out.println("Passed");
    } else {
      System.out.println("Failed");
    }
    start = parser.getIdToProtoMap().get("zzya4dDVRLk").getStart();
    if (start == 30) {
      System.out.println("Passed");
    } else {
      System.out.println("Failed");
    }
    int end = parser.getIdToProtoMap().get("--PJHxphWEs").getEnd();
    if (end == 40) {
      System.out.println("Passed");
    } else {
      System.out.println("Failed");
    }
    end = parser.getIdToProtoMap().get("2KwwQHit0-Q").getEnd();
    if (end == 40) {
      System.out.println("Passed");
    } else {
      System.out.println("Failed");
    }
    end = parser.getIdToProtoMap().get("zzya4dDVRLk").getEnd();
    if (end == 40) {
      System.out.println("Passed");
    } else {
      System.out.println("Failed");
    }
    int count = parser.getIdToProtoMap().get("--PJHxphWEs").getLabelsCount();
    String[] labelsList = new String[] {"/m/09x0r", "/t/dd00088"};
    for (int i = 0; i < count; i++) {
      if (labelsList[i].equals(
          parser.getIdToProtoMap().get("--PJHxphWEs").getLabels(i).getLabel())) {
        System.out.printf("Passed at element %d\n", i);
      } else {
        System.out.printf("Failed at element %d\n", i);
      }
    }
    count = parser.getIdToProtoMap().get("2KwwQHit0-Q").getLabelsCount();
    labelsList = new String[] {"/m/0l156k"};
    for (int i = 0; i < count; i++) {
      if (labelsList[i].equals(
          parser.getIdToProtoMap().get("2KwwQHit0-Q").getLabels(i).getLabel())) {
        System.out.printf("Passed at element %d\n", i);
      } else {
        System.out.printf("Failed at element %d\n", i);
      }
    }
    count = parser.getIdToProtoMap().get("zzya4dDVRLk").getLabelsCount();
    labelsList = new String[] {"/m/01v_m0", "/m/0hdsk"};
    for (int i = 0; i < count; i++) {
      if (labelsList[i].equals(
          parser.getIdToProtoMap().get("zzya4dDVRLk").getLabels(i).getLabel())) {
        System.out.printf("Passed at element %d\n", i);
      } else {
        System.out.printf("Failed at element %d\n", i);
      }
    }
    VideoDownloader.downloadList(parser.videoIdList);
  }
}
