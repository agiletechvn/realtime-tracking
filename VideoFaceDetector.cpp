#include "VideoFaceDetector.h"
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <dlib/string.h>
#include <iostream>
//#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>

// using json = nlohmann::json;
using namespace dlib;

// ----------------------------------------------------------------------------------------

// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the
// introductory dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train
// this network. The dlib_face_recognition_resnet_model_v1 model used by this
// example was trained using essentially the code shown in
// dnn_metric_learning_on_images_ex.cpp except the mini-batches were made larger
// (35x15 instead of 5x5), the iterations without progress was set to 10000, and
// the training dataset consisted of about 3 million images instead of
// 55.  Also, the input layer was locked to images of size 150.
template <template <int, template <typename> class, int, typename> class block,
          int N, template <typename> class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template <typename> class, int, typename> class block,
          int N, template <typename> class BN, typename SUBNET>
using residual_down =
    add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block =
    BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET>
using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET>
using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET>
using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET>
using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET>
using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<
    128, avg_pool_everything<alevel0<alevel1<alevel2<alevel3<alevel4<max_pool<
             3, 3, 2, 2,
             relu<affine<con<32, 7, 7, 2, 2, input_rgb_image_sized<150>
                             //                              input_rgb_image
                             >>>>>>>>>>>>;

static anet_type net;
static bool initialized = false;

const double VideoFaceDetector::TICK_FREQUENCY = cv::getTickFrequency();

void VideoFaceDetector::add_faces(const std::string &folder) {

  for (auto file_name : directory(folder).get_files()) {
    auto pos = file_name.name().find(".jpg");
    if (pos == std::string::npos)
      continue;
    matrix<rgb_pixel> face_chip;
    dlib::load_jpeg(face_chip, file_name);
    auto &face_descriptor = net(face_chip);
    if (face_descriptor.size() == face_dim) {
      nb++;
      copy(face_descriptor.begin(), face_descriptor.end(), back_inserter(xb));
      std::cout << "data" << nb << std::endl;
    }
  }

  // reindex
  reindex();
}

VideoFaceDetector::VideoFaceDetector(const std::string cascadeFilePath,
                                     cv::VideoCapture &videoCapture,
                                     const std::string name, bool learn) {

  if (!initialized) {
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;
    initialized = true;
  }

  setFaceCascade(cascadeFilePath);
  setVideoCapture(videoCapture);

  m_name = name;

  //  m_cli = new httplib::SSLClient("localhost", 5000);

  if (learn)
    add_faces(name);
  m_learn = learn;
}

void VideoFaceDetector::setVideoCapture(cv::VideoCapture &videoCapture) {
  m_videoCapture = &videoCapture;
}

cv::VideoCapture *VideoFaceDetector::videoCapture() const {
  return m_videoCapture;
}

void VideoFaceDetector::setFaceCascade(const std::string cascadeFilePath) {
  if (m_faceCascade == nullptr) {
    m_faceCascade = new cv::CascadeClassifier(cascadeFilePath);
  } else {
    m_faceCascade->load(cascadeFilePath);
  }

  if (m_faceCascade->empty()) {
    std::cerr << "Error creating cascade classifier. Make sure the file"
              << std::endl
              << cascadeFilePath << " exists." << std::endl;
  }
}

cv::CascadeClassifier *VideoFaceDetector::faceCascade() const {
  return m_faceCascade;
}

void VideoFaceDetector::setResizedWidth(const int width) {
  m_resizedWidth = std::max(width, 1);
}

int VideoFaceDetector::resizedWidth() const { return m_resizedWidth; }

bool VideoFaceDetector::isFaceFound() const { return m_foundFace; }

cv::Rect VideoFaceDetector::face() const {
  cv::Rect faceRect = m_trackedFace;
  faceRect.x = static_cast<int>(faceRect.x / m_scale);
  faceRect.y = static_cast<int>(faceRect.y / m_scale);
  faceRect.width = static_cast<int>(faceRect.width / m_scale);
  faceRect.height = static_cast<int>(faceRect.height / m_scale);
  return faceRect;
}

std::string VideoFaceDetector::name() const { return m_name; }

cv::Point VideoFaceDetector::facePosition() const {
  cv::Point facePos;
  facePos.x = static_cast<int>(m_facePosition.x / m_scale);
  facePos.y = static_cast<int>(m_facePosition.y / m_scale);
  return facePos;
}

void VideoFaceDetector::setTemplateMatchingMaxDuration(const double s) {
  m_templateMatchingMaxDuration = s;
}

double VideoFaceDetector::templateMatchingMaxDuration() const {
  return m_templateMatchingMaxDuration;
}

VideoFaceDetector::~VideoFaceDetector() {
  if (m_faceCascade != nullptr)
    delete m_faceCascade;

  //  if (m_cli != nullptr)
  //    delete m_cli;

  delete faiss_index;
  delete quantizer;
}

cv::Rect VideoFaceDetector::doubleRectSize(const cv::Rect &inputRect,
                                           const cv::Rect &frameSize) const {
  cv::Rect outputRect;
  // Double rect size
  outputRect.width = inputRect.width * 2;
  outputRect.height = inputRect.height * 2;

  // Center rect around original center
  outputRect.x = inputRect.x - inputRect.width / 2;
  outputRect.y = inputRect.y - inputRect.height / 2;

  // Handle edge cases
  if (outputRect.x < frameSize.x) {
    outputRect.width += outputRect.x;
    outputRect.x = frameSize.x;
  }
  if (outputRect.y < frameSize.y) {
    outputRect.height += outputRect.y;
    outputRect.y = frameSize.y;
  }

  if (outputRect.x + outputRect.width > frameSize.width) {
    outputRect.width = frameSize.width - outputRect.x;
  }
  if (outputRect.y + outputRect.height > frameSize.height) {
    outputRect.height = frameSize.height - outputRect.y;
  }

  return outputRect;
}

cv::Point VideoFaceDetector::centerOfRect(const cv::Rect &rect) const {
  return cv::Point(rect.x + rect.width / 2, rect.y + rect.height / 2);
}

cv::Rect VideoFaceDetector::biggestFace(std::vector<cv::Rect> &faces) const {
  assert(!faces.empty());

  cv::Rect *biggest = &faces[0];
  for (auto &face : faces) {
    if (face.area() < biggest->area())
      biggest = &face;
  }
  return *biggest;
}

/*
 * Face template is small patch in the middle of detected face.
 */
cv::Mat VideoFaceDetector::getFaceTemplate(const cv::Mat &frame,
                                           cv::Rect face) {
  face.x += face.width / 4;
  face.y += face.height / 4;
  face.width /= 2;
  face.height /= 2;

  cv::Mat faceTemplate = frame(face).clone();
  return faceTemplate;
}

void VideoFaceDetector::limitRect(const cv::Mat &frame, cv::Rect &rect) {
  if (rect.x < 0)
    rect.x = 0;
  if (rect.y < 0)
    rect.y = 0;
  if (rect.x + rect.width > frame.cols)
    rect.width = frame.cols - rect.x;
  if (rect.y + rect.height > frame.rows)
    rect.height = frame.rows - rect.y;
}

float VideoFaceDetector::search_face_local(const std::vector<float> &xq) {

  auto *I = new long[1];
  auto *D = new float[1];

  //  std::cout << "before search" << std::endl;

  faiss_index->search(1, xq.data(), 1, D, I);

  return D[0];
}

void VideoFaceDetector::reindex() {

  if (nb < 1)
    return;
  if (faiss_index != nullptr) {
    delete faiss_index;
    delete quantizer;
  }

  quantizer = new faiss::IndexFlatL2(face_dim);
  faiss_index = new faiss::IndexIVFFlat(quantizer, face_dim, 1);
  faiss_index->nprobe = 1;

  auto data = xb.data();

  // here we specify METRIC_L2, by default it performs inner-product search
  faiss_index->train(nb, data);
  faiss_index->add(nb, data);
}

void VideoFaceDetector::searchFaceChip(const cv::Mat &chip_frame) {
  static std::vector<float> xq(face_dim);

  if (chip_frame.empty())
    return;

  //  std::cout << "chip_frame" << chip_frame.cols << std::endl;

  //  cv::imshow("test", chip_frame);

  // later will be multiple face_chips as well
  matrix<rgb_pixel> face_chip;
  assign_image(face_chip, cv_image<bgr_pixel>(chip_frame));
  auto &face_descriptor = net(face_chip);

  if (xb.empty()) {
    copy(face_descriptor.begin(), face_descriptor.end(), back_inserter(xb));
    nb = 1;
    cv::imwrite(m_name + "/1.jpg", chip_frame);
    reindex();
  } else {
    //  json data = net(face_chip);

    xq.assign(face_descriptor.begin(), face_descriptor.end());

    auto conf = search_face_local(xq);
    if (conf > 0.2f) {
      nb++;
      cv::imshow("test", chip_frame);
      std::cout << "num points: " << nb << " confidence " << conf << std::endl;
      copy(face_descriptor.begin(), face_descriptor.end(), back_inserter(xb));
      cv::imwrite(m_name + "/" + std::to_string(nb) + ".jpg", chip_frame);
      reindex();
    }
  }

  //  auto res = m_cli->Post("/search_face", data.dump(), "application/json");
  //  if (res && res->status == 200) {
  //    data = json::parse(res->body);
  //    auto first = data[0];
  //    if (first["confidence"] > 0.8f) {
  //      return first["name"];
  //    }
  //  }

  //  return "unknow";
}

void VideoFaceDetector::detectFaceAllSizes(const cv::Mat &frame) {
  // Minimum face size is 1/5th of screen height
  // Maximum face size is 2/3rds of screen height
  m_faceCascade->detectMultiScale(
      frame, m_allFaces, 1.1, 3, 0, cv::Size(frame.rows / 5, frame.rows / 5),
      cv::Size(frame.rows * 2 / 3, frame.rows * 2 / 3));

  if (m_allFaces.empty())
    return;

  m_foundFace = true;

  // Locate biggest face
  m_trackedFace = biggestFace(m_allFaces);

  // Copy face template
  m_faceTemplate = getFaceTemplate(frame, m_trackedFace);

  // Calculate roi
  m_faceRoi =
      doubleRectSize(m_trackedFace, cv::Rect(0, 0, frame.cols, frame.rows));

  // Update face position
  m_facePosition = centerOfRect(m_trackedFace);
}

void VideoFaceDetector::detectFaceAroundRoi(const cv::Mat &frame) {
  // Detect faces sized +/-20% off biggest face in previous search
  m_faceCascade->detectMultiScale(
      frame(m_faceRoi), m_allFaces, 1.1, 3, 0,
      cv::Size(m_trackedFace.width * 8 / 10, m_trackedFace.height * 8 / 10),
      cv::Size(m_trackedFace.width * 12 / 10, m_trackedFace.width * 12 / 10));

  if (m_allFaces.empty()) {
    // Activate template matching if not already started and start timer
    m_templateMatchingRunning = true;
    if (m_templateMatchingStartTime == 0)
      m_templateMatchingStartTime = cv::getTickCount();
    return;
  }

  // Turn off template matching if running and reset timer
  m_templateMatchingRunning = false;
  m_templateMatchingCurrentTime = m_templateMatchingStartTime = 0;

  // Get detected face
  m_trackedFace = biggestFace(m_allFaces);

  // Add roi offset to face
  m_trackedFace.x += m_faceRoi.x;
  m_trackedFace.y += m_faceRoi.y;

  // Get face template
  m_faceTemplate = getFaceTemplate(frame, m_trackedFace);

  // Calculate roi
  m_faceRoi =
      doubleRectSize(m_trackedFace, cv::Rect(0, 0, frame.cols, frame.rows));

  // Update face position
  m_facePosition = centerOfRect(m_trackedFace);
}

void VideoFaceDetector::detectFacesTemplateMatching(const cv::Mat &frame) {
  // Calculate duration of template matching
  m_templateMatchingCurrentTime = cv::getTickCount();
  double duration = static_cast<double>(m_templateMatchingCurrentTime -
                                        m_templateMatchingStartTime) /
                    TICK_FREQUENCY;

  // If template matching lasts for more than 2 seconds face is possibly lost
  // so disable it and redetect using cascades
  if (duration > m_templateMatchingMaxDuration) {
    m_foundFace = false;
    m_templateMatchingRunning = false;
    m_templateMatchingStartTime = m_templateMatchingCurrentTime = 0;
    m_facePosition.x = m_facePosition.y = 0;
    m_trackedFace.x = m_trackedFace.y = m_trackedFace.width =
        m_trackedFace.height = 0;
    return;
  }

  // Edge case when face exits frame while
  if (m_faceTemplate.rows * m_faceTemplate.cols == 0 ||
      m_faceTemplate.rows <= 1 || m_faceTemplate.cols <= 1) {
    m_foundFace = false;
    m_templateMatchingRunning = false;
    m_templateMatchingStartTime = m_templateMatchingCurrentTime = 0;
    m_facePosition.x = m_facePosition.y = 0;
    m_trackedFace.x = m_trackedFace.y = m_trackedFace.width =
        m_trackedFace.height = 0;
    return;
  }

  // Template matching with last known face
  // cv::matchTemplate(frame(m_faceRoi), m_faceTemplate, m_matchingResult,
  // CV_TM_CCOEFF);
  cv::matchTemplate(frame(m_faceRoi), m_faceTemplate, m_matchingResult,
                    cv::TM_SQDIFF_NORMED);
  cv::normalize(m_matchingResult, m_matchingResult, 0, 1, cv::NORM_MINMAX, -1,
                cv::Mat());
  double min, max;
  cv::Point minLoc, maxLoc;
  cv::minMaxLoc(m_matchingResult, &min, &max, &minLoc, &maxLoc);

  // Add roi offset to face position
  minLoc.x += m_faceRoi.x;
  minLoc.y += m_faceRoi.y;

  // Get detected face
  // m_trackedFace = cv::Rect(maxLoc.x, maxLoc.y, m_trackedFace.width,
  // m_trackedFace.height);
  m_trackedFace =
      cv::Rect(minLoc.x, minLoc.y, m_faceTemplate.cols, m_faceTemplate.rows);
  m_trackedFace =
      doubleRectSize(m_trackedFace, cv::Rect(0, 0, frame.cols, frame.rows));

  // Get new face template
  m_faceTemplate = getFaceTemplate(frame, m_trackedFace);

  // Calculate face roi
  m_faceRoi =
      doubleRectSize(m_trackedFace, cv::Rect(0, 0, frame.cols, frame.rows));

  // Update face position
  m_facePosition = centerOfRect(m_trackedFace);
}

cv::Point VideoFaceDetector::getFrameAndDetect(cv::Mat &frame) {
  *m_videoCapture >> frame;

  // Downscale frame to m_resizedWidth width - keep aspect ratio
  m_scale =
      static_cast<double>(std::min(m_resizedWidth, frame.cols)) / frame.cols;
  cv::Size resizedFrameSize = cv::Size(static_cast<int>(m_scale * frame.cols),
                                       static_cast<int>(m_scale * frame.rows));

  if (frame.empty() || resizedFrameSize.empty())
    return cv::Point(0, 0);

  cv::Mat resizedFrame;
  cv::resize(frame, resizedFrame, resizedFrameSize);

  if (!m_foundFace) {
    detectFaceAllSizes(resizedFrame); // Detect using cascades over whole image
  } else {
    detectFaceAroundRoi(resizedFrame); // Detect using cascades only in ROI
    if (m_templateMatchingRunning) {
      detectFacesTemplateMatching(
          resizedFrame); // Detect using template matching
    }
  }

  if (m_learn) {

    // search face chip with this face to update label
    auto trackedFace =
        cv::Rect(static_cast<int>(m_trackedFace.x / m_scale),
                 static_cast<int>(m_trackedFace.y / m_scale),
                 static_cast<int>(m_trackedFace.width / m_scale),
                 static_cast<int>(m_trackedFace.height / m_scale));
    limitRect(frame, trackedFace);
    if (trackedFace.empty())
      return cv::Point(0, 0);

    cv::Mat chip_frame;
    cv::resize(frame(trackedFace), chip_frame, cv::Size(150, 150));
    searchFaceChip(chip_frame);
  }

  return m_facePosition;
}

cv::Point VideoFaceDetector::operator>>(cv::Mat &frame) {
  return this->getFrameAndDetect(frame);
}
