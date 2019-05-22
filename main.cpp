#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "VideoFaceDetector.h"

const cv::String WINDOW_NAME("Camera video");
const cv::String CASCADE_FILE("haarcascade_frontalface_default.xml");

int main() {
  // Try opening camera
  cv::VideoCapture camera(0);
  // cv::VideoCapture camera("D:\\video.mp4");
  if (!camera.isOpened()) {
    fprintf(stderr, "Error getting camera...\n");
    exit(1);
  }

  cv::namedWindow(WINDOW_NAME, cv::WINDOW_FULLSCREEN);
  cv::moveWindow(WINDOW_NAME, 0, 0);

  VideoFaceDetector detector(CASCADE_FILE, camera, "Pham Thanh Tu");
  cv::Mat frame;
  //  double fps = 0, time_per_frame;
  int prev_x1 = 0, prev_y1 = 0, prev_x2 = 0, prev_y2 = 0, delta = 20;
  while (true) {

    detector >> frame;

    if (detector.isFaceFound()) {
      auto box = detector.face();

      if (abs(box.x - prev_x1) < delta)
        box.x = prev_x1;
      else
        prev_x1 = box.x;

      if (abs(box.y - prev_y1) < delta)
        box.y = prev_y1;
      else
        prev_y1 = box.y;

      if (abs(box.br().x - prev_x2) < delta)
        box.width = prev_x2 - box.x;
      else
        prev_x2 = box.br().x;

      if (abs(box.br().y - prev_y2) < delta)
        box.height = prev_y2 - box.y;
      else
        prev_y2 = box.br().y;

      cv::rectangle(frame, box, cv::Scalar(255, 0, 0), 2);
      auto text = detector.name();
      auto box_t = cv::Rect(box.x, box.br().y, box.width, 40);
      cv::rectangle(frame, box_t, cv::Scalar(255, 0, 0), cv::FILLED);
      cv::putText(frame, text, cv::Point(box.x + 10, box_t.br().y - 10),
                  cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    }

    cv::imshow(WINDOW_NAME, frame);
    if (cv::waitKey(25) == 27)
      break;
  }

  return 0;
}
