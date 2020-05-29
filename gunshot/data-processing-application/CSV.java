import java.io.File;
import java.io.IOException;
import java.util.Scanner;
import java.util.ArrayList;

/**
 * CSV Class
 * 
 * <p>CSV is a class for the reading of a dataset in a csv format
 * 
 * @author Carver Forbes
 * @version 0.1
 */
public class CSV {
  private File file;

  /**
   * CSV constructor
   * 
   * @param pathname the pathname for the csv file to be read
   * @throws IllegalArgumentException if the pathname is incorrect
   */
  public CSV(String pathname) throws IllegalArgumentException {
    file = new File(pathname);
    //TODO: handle permission issues
    if (!file.exists()) {
        throw new IllegalArgumentException("No such file exists");
    }
  }   

  /** Tester Function to print 5 YouTube Links */
  public void printYouTubeLinks() {
    // prints youtube links
    try {
      Scanner scan = new Scanner(file);
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
      scan.close();
    } catch (IOException io) {
      io.printStackTrace();
    }
  }

  /**
   * Extracts metadata from dataset.csv file
   * 
   * @param pb the ProtocolBuffer to extract the metadata to
   */
  public void setMetadata(ProtocolBuffer pb) {
    //TODO: add in error checking
    try {
      Scanner scan = new Scanner(file);
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
        if (scan.hasNext()) line = scan.nextLine();
      } while (scan.hasNext());
      scan.close();
    } catch (IOException io) {
      io.printStackTrace();
    }
  }

  /**
   * Tester main function to run printYouTubeLinks
   * 
   * @param args list of command-line arguments
   */
  public static void main(String[] args) {
    // run csv parsing
    System.out.println("Hello, world");
    CSV balancedDataSet = new CSV("datasets/balanced_train_segments.csv");
    balancedDataSet.printYouTubeLinks();
  }
}
    