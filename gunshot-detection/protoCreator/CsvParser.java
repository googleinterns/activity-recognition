import java.io.File;
import java.io.IOException;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.HashMap;

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

  /**
   * CSV constructor
   *
   * @param pathname the pathname for the csv file to be read
   * @throws IllegalArgumentException if the pathname is incorrect
   */
  public CsvParser(String pathname) {
    file = new File(pathname);
    if (!file.exists()) {
      throw new IllegalArgumentException("No such file exists");
    }
  }

  /**
   * Tester Function to print 5 YouTube Links
   */
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

  /**
   * Extracts metadata from dataset.csv file
   *
   * @param pb the ProtocolBuffer to extract the metadata to
   */
  public void setMetadata(ProtocolBuffer pb) {
    //TODO: add in error checking
    Scanner scan = null;
    try {
      scan = new Scanner(file);
      String line = scan.nextLine();
      while (line.charAt(0) == '#') {
        line = scan.nextLine();
      }
      do {
        String[] lineArr = line.split(",");
        String ytid = lineArr[0];
        int start = (int) Double.parseDouble(lineArr[1]);
        int end = (int) Double.parseDouble(lineArr[2]);
        String labels = lineArr[3];
        String[] labelsList = labels.split(",");
        String url = "https://youtube.com/watch?v=" + ytid;

        pb.getIdToUrlMap().put(ytid, url);
        pb.getIdToStartMap().put(ytid, start);
        pb.getIdToEndMap().put(ytid, end);
        ArrayList<String> al = new ArrayList<>();
        for (int i = 0; i < labelsList.length; i++) {
          al.add(labelsList[i]);
        }
        pb.getIdToLabelsMap().put(ytid, al);
          if (scan.hasNext()) {
              line = scan.nextLine();
          }
      } while (scan.hasNext());
    } catch (IOException io) {
      io.printStackTrace();
    } finally {
        if (scan != null) {
            scan.close();
        }
    }
  }

  /**
   * Extracts metadata from dataset.csv file
   */
  public void csvToProtoMap() {
    Scanner scan = null;
    try {
      scan = new Scanner(file);
      String line = scan.nextLine();
      while (line.charAt(0) == '#') {
        line = scan.nextLine();
      }
      do {
        createProto(line);
      } while (scan.hasNext());
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
    String[] lineArr = csvLine.split(",");
    String ytid = lineArr[0];
    ytidList.add(ytid);
    String url = "https://youtube.com/watch?v=" + ytid;
    idToUrlMap.put(ytid, url);
    int start = (int) Double.parseDouble(lineArr[1]);
    int end = (int) Double.parseDouble(lineArr[2]);
    String labels = lineArr[3];
    String[] labelsList = labels.split(",");
    ArrayList<IncompleteExample.Example.Label> protoLabels = new ArrayList<>();
    for (int i = 0; i < labelsList.length; i++) {
      IncompleteExample.Example.Label protoLabel = IncompleteExample.Example.Label.newBuilder()
          .setLabel(labelsList[i])
          .build();
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
}