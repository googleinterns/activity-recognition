import java.io.File;
import java.io.IOException;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Arrays;

/**
 * CSV Class
 *
 * <p>CSV is a class for the reading of a dataset in a csv format.
 *
 * @author Carver Forbes
 * @version 0.1
 */
public class CsvParser {

  private File file;
  private ArrayList<String> ytidList;
  private HashMap<String, IncompleteExample.Example> idToProtoMap;
  private HashMap<String, String> idToUrlMap;

  private final boolean debug = true;

  /**
   * CSV constructor
   *
   * @param pathname the pathname for the csv file to be read
   * @throws IllegalArgumentException if the pathname is incorrect
   */
  public CsvParser(String pathname) {
    file = new File(pathname);
    if (!file.exists()) {
      throw new IllegalArgumentException(String.format("This file does not exist: %s", pathname));
    }
    ytidList = new ArrayList<>();
    idToProtoMap = new HashMap<>();
    idToUrlMap = new HashMap<>();
  }

  /** Tester Function to print 5 YouTube Links */
  public void printYouTubeLinks() {
    // prints youtube links
    Scanner scan = null;
    try {
      scan = new Scanner(file);
      String line = scan.nextLine();
      while (line.charAt(0) == '#') {
        line = scan.nextLine();
      }
      // while (scan.hasNext()) {
      //   String ytid = line.split(",")[0];
      //   String url = "https://youtube.com/watch?v=" + ytid;
      //   System.out.println(url);
      //   line = scan.nextLine();
      // }
      for (int i = 0; i < 5; i++) {
        String ytid = line.split(",")[0];
        String url = "https://youtube.com/watch?v=" + ytid;
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
    if (debug) System.out.println("in the csvToProtoMap method");
    Scanner scan = null;
    try {
      scan = new Scanner(file);
      int count = 0;
      String line = scan.nextLine();
      count++;
      while (line.charAt(0) == '#') {
        line = scan.nextLine();
        count++;
      }
      do {
        createProto(line);
        if (debug) System.out.printf("At line %d\n", count);
        line = scan.nextLine();
        count++;
      } while (scan.hasNext());
      createProto(line);
      if (debug) System.out.printf("At line %d\n", count);
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
    if (debug) System.out.println("in the createProto method");
    String[] lineArr = csvLine.split(",");
    String ytid = lineArr[0];
    ytidList.add(ytid);
    String url = "https://youtube.com/watch?v=" + ytid;
    idToUrlMap.put(ytid, url);
    int start = (int) Double.parseDouble(lineArr[1]);
    int end = (int) Double.parseDouble(lineArr[2]);
    String[] labelsList = Arrays.copyOfRange(lineArr, 3, lineArr.length);
    ArrayList<IncompleteExample.Example.Label> protoLabels = new ArrayList<>();
    int len = labelsList.length;
    for (int i = 0; i < len; i++) {
      labelsList[i] = labelsList[i].replace("\"", "");
      labelsList[i] = labelsList[i].trim();
      IncompleteExample.Example.Label protoLabel =
          IncompleteExample.Example.Label.newBuilder().setLabel(labelsList[i]).build();
      protoLabels.add(protoLabel);
    }
    IncompleteExample.Example proto =
        IncompleteExample.Example.newBuilder()
            .setYtid(ytid)
            .setStart(start)
            .setEnd(end)
            .addAllLabels(protoLabels)
            .build();
    idToProtoMap.put(ytid, proto);
  }

  public HashMap<String, String> getIdToUrlMap() {
    return idToUrlMap;
  }

  public HashMap<String, IncompleteExample.Example> getIdToProtoMap() {
    return idToProtoMap;
  }

  public ArrayList<String> getYtidList() {
    return ytidList;
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
    String url = parser.getIdToUrlMap().get("--PJHxphWEs");
    if ("https://youtube.com/watch?v=--PJHxphWEs".equals(url)) {
      System.out.println("Passed");
    } else {
      System.out.println("Failed");
    }
    url = parser.getIdToUrlMap().get("2KwwQHit0-Q");
    if ("https://youtube.com/watch?v=2KwwQHit0-Q".equals(url)) {
      System.out.println("Passed");
    } else {
      System.out.println("Failed");
    }
    url = parser.getIdToUrlMap().get("zzya4dDVRLk");
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
  }
}
