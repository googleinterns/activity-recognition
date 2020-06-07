import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class CsvParserTest {

  CsvParser parser;

  @Before
  public void initialize() {
    parser = new CsvParser("datasets/balanced_train_segments.csv");
    parser.csvToProtoMap();
  }

  @Test
  public void urlFirstTest() {
    String url = parser.getIdToUrlMap().get("--PJHxphWEs");
    assertEquals("https://youtube.com/watch?v=--PJHxphWEs", url);
  }

  @Test
  public void urlMiddleTest() {
    String url = parser.getIdToUrlMap().get("2KwwQHit0-Q");
    assertEquals("https://youtube.com/watch?v=2KwwQHit0-Q", url);
  }

  @Test
  public void urlLastTest() {
    String url = parser.getIdToUrlMap().get("zzya4dDVRLk");
    assertEquals("https://youtube.com/watch?v=zzya4dDVRLk", url);
  }

  @Test
  public void startFirstTest() {
    int start = parser.getIdToProtoMap().get("--PJHxphWEs").getStart();
    assertEquals(30, start);
  }

  @Test
  public void startMiddleTest() {
    int start = parser.getIdToProtoMap().get("2KwwQHit0-Q").getStart();
    assertEquals(30, start);
  }

  @Test
  public void startLastTest() {
    int start = parser.getIdToProtoMap().get("zzya4dDVRLk").getStart();
    assertEquals(30, start);
  }

  @Test
  public endFirstTest() {
    int end = parser.getIdToProtoMap().get("--PJHxphWEs").getEnd();
    assertEquals(40, end);
  }

  @Test
  public void endMiddleTest() {
    int end = parser.getIdToProtoMap().get("2KwwQHit0-Q").getEnd();
    assertEquals(40, end);
  }

  @Test
  public void endLastTest() {
    int end = parser.getIdToProtoMap().get("zzya4dDVRLk").getEnd();
    assertEquals(40, end);
  }

  @Test
  public void labelsFirstTest() {
    int count = parser.getIdToProtoMap().get("--PJHxphWEs").getLabelsCount();
    String[] labelsList = {"m/09x9r", "/t/dd00088"};
    for (int i = 0; i < count; i++) {
      assertEquals(labelsList[i], parser.getIdToProtoMap().get("--PJHxphWEs").getLabels(i).getLabel());
    }
  }

  @Test
  public void labelsMiddleTest() {
    int count = parser.getIdToProtoMap().get("2KwwQHit0-Q").getLabelsCount();
    String[] labelsList = {"/m/0l156k"};
    for (int i = 0; i < count; i++) {
      assertEquals(labelsList[i], parser.getIdToProtoMap().get("2KwwQHit0-Q").getLabels(i).getLabel());
    }
  }

  @Test
  public void labelsEndTest() {
    int count = parser.getIdToProtoMap().get("zzya4dDVRLk").getLabelsCount();
    String[] labelsList = {"/m/01v_m0", "/m/0hdsk"};
    for (int i = 0; i < count; i++) {
      assertEquals(labelsList[i], parser.getIdToProtoMap().get("zzya4dDVRLk").getLabels(i).getLabel());
    }
  }
}