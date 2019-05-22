#ifndef VIDEO_FACEDETECTOR_H
#define VIDEO_FACEDETECTOR_H

#define CPPHTTPLIB_OPENSSL_SUPPORT

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

//#include "httplib.h"

class VideoFaceDetector {
public:
  VideoFaceDetector(const std::string cascadeFilePath,
                    cv::VideoCapture &videoCapture, const std::string name,
                    bool learn = false);
  ~VideoFaceDetector();

  cv::Point getFrameAndDetect(cv::Mat &frame);
  cv::Point operator>>(cv::Mat &frame);
  void setVideoCapture(cv::VideoCapture &videoCapture);
  cv::VideoCapture *videoCapture() const;
  void setFaceCascade(const std::string cascadeFilePath);
  cv::CascadeClassifier *faceCascade() const;
  void setResizedWidth(const int width);
  int resizedWidth() const;
  bool isFaceFound() const;
  cv::Rect face() const;
  std::string name() const;
  cv::Point facePosition() const;
  void setTemplateMatchingMaxDuration(const double s);
  double templateMatchingMaxDuration() const;

private:
  static const double TICK_FREQUENCY;

  //  httplib::SSLClient *m_cli = nullptr;
  bool m_learn = false;
  cv::VideoCapture *m_videoCapture = nullptr;
  cv::CascadeClassifier *m_faceCascade = nullptr;
  std::vector<cv::Rect> m_allFaces;
  cv::Rect m_trackedFace;
  cv::Rect m_faceRoi;
  cv::Mat m_faceTemplate;
  cv::Mat m_matchingResult;
  bool m_templateMatchingRunning = false;
  int64 m_templateMatchingStartTime = 0;
  int64 m_templateMatchingCurrentTime = 0;
  bool m_foundFace = false;
  double m_scale;
  int m_resizedWidth = 320;
  const uint face_dim = 128;
  faiss::IndexFlatL2 *quantizer = nullptr;
  faiss::IndexIVFFlat *faiss_index = nullptr;
  cv::Point m_facePosition;
  double m_templateMatchingMaxDuration = 3;
  std::string m_name;
  std::vector<float> xb;
  int nb = 0;
  cv::Rect doubleRectSize(const cv::Rect &inputRect,
                          const cv::Rect &frameSize) const;
  cv::Rect biggestFace(std::vector<cv::Rect> &faces) const;
  cv::Point centerOfRect(const cv::Rect &rect) const;
  cv::Mat getFaceTemplate(const cv::Mat &frame, cv::Rect face);
  void detectFaceAllSizes(const cv::Mat &frame);
  void detectFaceAroundRoi(const cv::Mat &frame);
  void detectFacesTemplateMatching(const cv::Mat &frame);
  float search_face_local(const std::vector<float> &xq);
  void reindex();
  void add_faces(const std::string &folder);
  void searchFaceChip(const cv::Mat &chip_frame);
  void limitRect(const cv::Mat &frame, cv::Rect &rect);
};

#endif // VIDEO_FACEDETECTOR_H
