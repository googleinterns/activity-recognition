package com.google.location.lbs.activity.audioset.dataprocessing;

import java.util.List;
import java.util.ArrayList;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.IOException;
import java.io.File;
import java.io.FileWriter;

public final class VideoDownloader {
  public static void downloadList(List<String> videoIdList) {
    System.out.println("Starting main function now");
    Process p;
    try {
      System.out.println("Starting try block now");
      File script = new File("java/com/google/location/lbs/activity/audioset/dataprocessing/download.sh");
      script.createNewFile();
      FileWriter writer = new FileWriter("java/com/google/location/lbs/activity/audioset/dataprocessing/download.sh");
      writer.write("echo 'starting download script'");
      for (String videoId : videoIdList) {
        String command = "youtube-dl"
        writer.write("youtube-dl --id -x --audio-format wav https://www.youtube.com/watch?v=" + videoId);
      }
      writer.write("echo 'ending download script'");
      writer.close()
      List<String> cmdList = new ArrayList<String>();
      // adding command and args to the list
      cmdList.add("sh");
      cmdList.add(script.getAbsolutePath());
      ProcessBuilder pb = new ProcessBuilder(cmdList);
      p = pb.start();
      p.waitFor();
      BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
      String line;
      int count = 0;
      while((line = reader.readLine()) != null) {
        System.out.println(line);
        count++;
      }
      System.out.println(count);
      System.out.println("Ending try block now");
    } catch (IOException e) {
      // TODO Auto-generated catch block
      System.out.println("IOexception");
      e.printStackTrace();
    } catch (InterruptedException e) {
      System.out.println("InterruptedException");
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
  }
}