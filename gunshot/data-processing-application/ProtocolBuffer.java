import java.util.HashMap;
import java.util.ArrayList;
/**
 * ProtocolBuffer Class
 * 
 * <p>ProtocolBuffer is a class for protocol buffer creation from datasets in a csv format
 * 
 * @author  Carver Forbes
 * @version 0.1
 */
public class ProtocolBuffer {
    
  /* 
   * Hashmaps to hold mappings from the YTID (YouTube Video ID)
   * to YouTube video url, start time in seconds, end time in seconds,
   * list of labels for the audio in the specified time segment, and 
   * a list of audio features for the whole video
   */
  private HashMap<String, String> idToUrlMap;
  private HashMap<String, Integer> idToStartMap;
  private HashMap<String, Integer> idToEndMap;
  private HashMap<String, ArrayList<String>> idToLabelsMap;
  private HashMap<String, ArrayList<Integer>> idToFeaturesMap;

  /** ProtocolBuffer Constructor */
  public ProtocolBuffer() {
    // initialize HashMaps as instance variables
    idToUrlMap = new HashMap<>();
    idToStartMap = new HashMap<>();
    idToEndMap = new HashMap<>();
    idToLabelsMap = new HashMap<>();
    idToFeaturesMap = new HashMap<>();
  }

  // getters
  public HashMap<String, String> getIdToUrlMap() {
    return idToUrlMap;
  }

  public HashMap<String, Integer> getIdToStartMap() {
    return idToStartMap;
  }

  public HashMap<String, Integer> getIdToEndMap() {
    return idToEndMap;
  }

  public HashMap<String, ArrayList<String>> getIdToLabelsMap() {
    return idToLabelsMap;
  }

  public HashMap<String, ArrayList<Integer>> getIdToFeaturesMap() {
    return idToFeaturesMap;
  }

  // setters
  public void setIdToUrlMap(HashMap<String, String> map) {
    idToUrlMap = map;
  }

  public void setIdToStartMap(HashMap<String, Integer> map) {
    idToStartMap = map;
  }

  public void setIdToEndMap(HashMap<String, Integer> map) {
    idToEndMap = map;
  }

  public void setIdToLabelsMap(HashMap<String, ArrayList<String>> map) {
    idToLabelsMap = map;
  }

  public void setIdToFeaturesMap(HashMap<String, ArrayList<Integer>> map) {
    idToFeaturesMap = map;
  }

  /** Prints 5 examples of metadata associated with YTID */
  public void print5metadata() {
    int count = 0;
    for (String key : idToUrlMap.keySet()) {
      if (count >= 5) break;
      System.out.printf("%s, ", key);
      System.out.printf("%s, ", idToUrlMap.get(key));
      System.out.printf("%d, ", idToStartMap.get(key));
      System.out.printf("%d, ", idToEndMap.get(key));
      System.out.println();
      count++;
    }
  }

  /**
   * Creates a ProtocolBuffer from a dataset.csv
   * 
   * @param args a list of command-line arguments
   */
  public static void main(String[] args) {
    // 1. create protobuffer and csv
    // 2. extract metadata from dataset.csv
    // 3. extract metadata from additionalAnnotations.csv if available
    // 4. download videos in dataset
    // 5. process downloaded audio
    // 6. Extract features from downloaded audio
    // 7. Write to protobuffer and merge any examples referring to same video
    //    (positive && negative labels == positive when it comes to additional annotations)

    CSV balancedDataSet = new CSV("datasets/balanced_train_segments.csv");
    ProtocolBuffer pb = new ProtocolBuffer();
    balancedDataSet.setMetadata(pb);
    pb.print5metadata();
  }
}