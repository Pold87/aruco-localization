// g++ detect_board.cpp serialib.cpp stereoprotocol.cpp -o detect_board  `pkg-config --cflags --libs opencv`

#include <stdio.h>
#include <iostream>

#include <math.h>
#include <thread>
#include <atomic>

#include <unistd.h>

#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

#include "stereoprotocol.h"
#include "serialib.h"

#include "opencv2/aruco.hpp"
#include "opencv2/aruco/dictionary.hpp"
#include <vector>

using namespace std;
using namespace cv;

std::atomic<bool> stop(false);

/**
 */
static void help() {
    cout << "Pose estimation using a ArUco Planar Grid board" << endl;
    cout << "Parameters: " << endl;
    cout << "-w <nmarkers> # Number of markers in X direction" << endl;
    cout << "-h <nmarkers> # Number of markers in Y direction" << endl;
    cout << "-l <markerLength> # Marker side lenght (in meters)" << endl;
    cout << "-s <markerSeparation> # Separation between two consecutive"
         << "markers in the grid (in meters)" << endl;
    cout << "-d <dictionary> # DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2, "
         << "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
         << "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
         << "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16" << endl;
    cout << "-c <cameraParams> # Camera intrinsic parameters file" << endl;
    cout << "[-v <videoFile>] # Input from video file, if ommited, input comes from camera" << endl;
    cout << "[-ci <int>] # Camera id if input doesnt come from video (-v). Default is 0" << endl;
    cout << "[-dp <detectorParams>] # File of marker detector parameters" << endl;
    cout << "[-rs] # Apply refind strategy" << endl;
    cout << "[-r] # show rejected candidates too" << endl;
}



void initKalmanFilter(cv::KalmanFilter &KF, int nStates, int nMeasurements, int nInputs, double dt)
{
  KF.init(nStates, nMeasurements, nInputs, CV_64F);                 // init Kalman Filter
  cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(0.005));       // set process noise
  cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(0.1));   // set measurement noise
  cv::setIdentity(KF.errorCovPost, cv::Scalar::all(1));             // error covariance
                 /* DYNAMIC MODEL */
  //  [1 0 0 dt  0  0 dt2   0   0 0 0 0  0  0  0   0   0   0]
  //  [0 1 0  0 dt  0   0 dt2   0 0 0 0  0  0  0   0   0   0]
  //  [0 0 1  0  0 dt   0   0 dt2 0 0 0  0  0  0   0   0   0]
  //  [0 0 0  1  0  0  dt   0   0 0 0 0  0  0  0   0   0   0]
  //  [0 0 0  0  1  0   0  dt   0 0 0 0  0  0  0   0   0   0]
  //  [0 0 0  0  0  1   0   0  dt 0 0 0  0  0  0   0   0   0]
  //  [0 0 0  0  0  0   1   0   0 0 0 0  0  0  0   0   0   0]
  //  [0 0 0  0  0  0   0   1   0 0 0 0  0  0  0   0   0   0]
  //  [0 0 0  0  0  0   0   0   1 0 0 0  0  0  0   0   0   0]
  //  [0 0 0  0  0  0   0   0   0 1 0 0 dt  0  0 dt2   0   0]  
  //  [0 0 0  0  0  0   0   0   0 0 1 0  0 dt  0   0 dt2   0]
  //  [0 0 0  0  0  0   0   0   0 0 0 1  0  0 dt   0   0 dt2]
  //  [0 0 0  0  0  0   0   0   0 0 0 0  1  0  0  dt   0   0]
  //  [0 0 0  0  0  0   0   0   0 0 0 0  0  1  0   0  dt   0]
  //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  1   0   0  dt]
  //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   1   0   0]
  //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   1   0]
  //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   0   1]
  // position
  KF.transitionMatrix.at<double>(0,3) = dt;
  KF.transitionMatrix.at<double>(1,4) = dt;
  KF.transitionMatrix.at<double>(2,5) = dt;
  KF.transitionMatrix.at<double>(3,6) = dt;
  KF.transitionMatrix.at<double>(4,7) = dt;
  KF.transitionMatrix.at<double>(5,8) = dt;
  KF.transitionMatrix.at<double>(0,6) = 0.5*pow(dt,2);
  KF.transitionMatrix.at<double>(1,7) = 0.5*pow(dt,2);
  KF.transitionMatrix.at<double>(2,8) = 0.5*pow(dt,2);
  // orientation
  KF.transitionMatrix.at<double>(9,12) = dt;
  KF.transitionMatrix.at<double>(10,13) = dt;
  KF.transitionMatrix.at<double>(11,14) = dt;
  KF.transitionMatrix.at<double>(12,15) = dt;
  KF.transitionMatrix.at<double>(13,16) = dt;
  KF.transitionMatrix.at<double>(14,17) = dt;
  KF.transitionMatrix.at<double>(9,15) = 0.5*pow(dt,2);
  KF.transitionMatrix.at<double>(10,16) = 0.5*pow(dt,2);
  KF.transitionMatrix.at<double>(11,17) = 0.5*pow(dt,2);
       /* MEASUREMENT MODEL */
  //  [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
  //  [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
  //  [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
  //  [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
  //  [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
  //  [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]
  KF.measurementMatrix.at<double>(0,0) = 1;  // x
  KF.measurementMatrix.at<double>(1,1) = 1;  // y
  KF.measurementMatrix.at<double>(2,2) = 1;  // z
  KF.measurementMatrix.at<double>(3,9) = 1;  // roll
  KF.measurementMatrix.at<double>(4,10) = 1; // pitch
  KF.measurementMatrix.at<double>(5,11) = 1; // yaw
}


// Converts a given Rotation Matrix to Euler angles
cv::Mat rot2euler(const cv::Mat & rotationMatrix)
{
  cv::Mat euler(3,1,CV_64F);

  double m00 = rotationMatrix.at<double>(0,0);
  double m02 = rotationMatrix.at<double>(0,2);
  double m10 = rotationMatrix.at<double>(1,0);
  double m11 = rotationMatrix.at<double>(1,1);
  double m12 = rotationMatrix.at<double>(1,2);
  double m20 = rotationMatrix.at<double>(2,0);
  double m22 = rotationMatrix.at<double>(2,2);

  double x, y, z;

  // Assuming the angles are in radians.
  if (m10 > 0.998) { // singularity at north pole
    x = 0;
    y = CV_PI/2;
    z = atan2(m02,m22);
  }
  else if (m10 < -0.998) { // singularity at south pole
    x = 0;
    y = -CV_PI/2;
    z = atan2(m02,m22);
  }
  else
  {
    x = atan2(-m12,m11);
    y = asin(m10);
    z = atan2(-m20,m00);
  }

  euler.at<double>(0) = x;
  euler.at<double>(1) = y;
  euler.at<double>(2) = z;

  return euler;
}


// Converts a given Euler angles to Rotation Matrix
cv::Mat euler2rot(const cv::Mat & euler)
{
  cv::Mat rotationMatrix(3,3,CV_64F);

  double x = euler.at<double>(0);
  double y = euler.at<double>(1);
  double z = euler.at<double>(2);

  // Assuming the angles are in radians.
  double ch = cos(z);
  double sh = sin(z);
  double ca = cos(y);
  double sa = sin(y);
  double cb = cos(x);
  double sb = sin(x);

  double m00, m01, m02, m10, m11, m12, m20, m21, m22;

  m00 = ch * ca;
  m01 = sh*sb - ch*sa*cb;
  m02 = ch*sa*sb + sh*cb;
  m10 = sa;
  m11 = ca*cb;
  m12 = -ca*sb;
  m20 = -sh*ca;
  m21 = sh*sa*cb + ch*sb;
  m22 = -sh*sa*sb + ch*cb;

  rotationMatrix.at<double>(0,0) = m00;
  rotationMatrix.at<double>(0,1) = m01;
  rotationMatrix.at<double>(0,2) = m02;
  rotationMatrix.at<double>(1,0) = m10;
  rotationMatrix.at<double>(1,1) = m11;
  rotationMatrix.at<double>(1,2) = m12;
  rotationMatrix.at<double>(2,0) = m20;
  rotationMatrix.at<double>(2,1) = m21;
  rotationMatrix.at<double>(2,2) = m22;

  return rotationMatrix;
}



void fillMeasurements(cv::Mat &measurements,
                      const cv::Mat &translation_measured,
                      const cv::Mat &rvec)
{
    // Convert rotation matrix to euler angles
    cv::Mat measured_eulers(3, 1, CV_64F);
    cv::Mat R(3, 3, CV_64F);

    cv::Rodrigues(rvec, R);
    measured_eulers = rot2euler(R);

   //cv2.Rodrigues(rotation_measured, measured_eulers);
    
    // Set measurement to predict
    measurements.at<double>(0) = translation_measured.at<double>(0); // x
    measurements.at<double>(1) = translation_measured.at<double>(1); // y
    measurements.at<double>(2) = translation_measured.at<double>(2); // z
    measurements.at<double>(3) = measured_eulers.at<double>(0);      // roll
    measurements.at<double>(4) = measured_eulers.at<double>(1);      // pitch
    measurements.at<double>(5) = measured_eulers.at<double>(2);      // yaw
}


void predictKalmanFilter( cv::KalmanFilter &KF,
                          cv::Mat &translation_estimated,
                          cv::Mat &rotation_estimated,
                          cv::Mat &speed_estimated) {

  // First predict, to update the internal statePre variable
  // This will give us a prediction of the variables, even without
  // a detected marker
  cv::Mat estimated = KF.predict();

  // Estimated translation
  translation_estimated.at<double>(0) = estimated.at<double>(0);
  translation_estimated.at<double>(1) = estimated.at<double>(1);
  translation_estimated.at<double>(2) = estimated.at<double>(2);
  
  // Estimated euler angles
  cv::Mat eulers_estimated(3, 1, CV_64F);
  eulers_estimated.at<double>(0) = estimated.at<double>(9);
  eulers_estimated.at<double>(1) = estimated.at<double>(10);
  eulers_estimated.at<double>(2) = estimated.at<double>(11);


  // Estimated speed
  speed_estimated.at<double>(0) = estimated.at<double>(3);
  speed_estimated.at<double>(1) = estimated.at<double>(4);
  speed_estimated.at<double>(2) = estimated.at<double>(5);
  
  // Convert estimated quaternion to rotation matrix
  //rotation_estimated = euler2rot(eulers_estimated);

  cv::Mat R(3, 3, CV_64F);
  R = euler2rot(eulers_estimated);

  cv::Rodrigues(R, rotation_estimated);
  
}


void updateKalmanFilter(cv::KalmanFilter &KF,
                        cv::Mat &measurement,
                        cv::Mat &translation_estimated,
                        cv::Mat &rotation_estimated,
                        cv::Mat &speed_estimated)
{

    // The "correct" phase that is going to use the predicted value and our measurement
    cv::Mat estimated = KF.correct(measurement);

    // Estimated translation
    translation_estimated.at<double>(0) = estimated.at<double>(0);
    translation_estimated.at<double>(1) = estimated.at<double>(1);
    translation_estimated.at<double>(2) = estimated.at<double>(2);

    // Estimated euler angles
    cv::Mat eulers_estimated(3, 1, CV_64F);
    
    eulers_estimated.at<double>(0) = estimated.at<double>(9);
    eulers_estimated.at<double>(1) = estimated.at<double>(10);
    eulers_estimated.at<double>(2) = estimated.at<double>(11);


    // Estimated speed
    speed_estimated.at<double>(0) = estimated.at<double>(3);
    speed_estimated.at<double>(1) = estimated.at<double>(4);
    speed_estimated.at<double>(2) = estimated.at<double>(5);

    cv::Mat R(3, 3, CV_64F);
    R = euler2rot(eulers_estimated);

    cv::Rodrigues(R, rotation_estimated);
}


void updateTransitionMatrix(cv::KalmanFilter &KF, double &dt) {

  // position
  KF.transitionMatrix.at<double>(0,3) = dt;
  KF.transitionMatrix.at<double>(1,4) = dt;
  KF.transitionMatrix.at<double>(2,5) = dt;
  KF.transitionMatrix.at<double>(3,6) = dt;
  KF.transitionMatrix.at<double>(4,7) = dt;
  KF.transitionMatrix.at<double>(5,8) = dt;
  KF.transitionMatrix.at<double>(0,6) = 0.5*pow(dt,2);
  KF.transitionMatrix.at<double>(1,7) = 0.5*pow(dt,2);
  KF.transitionMatrix.at<double>(2,8) = 0.5*pow(dt,2);
  // orientation
  KF.transitionMatrix.at<double>(9,12) = dt;
  KF.transitionMatrix.at<double>(10,13) = dt;
  KF.transitionMatrix.at<double>(11,14) = dt;
  KF.transitionMatrix.at<double>(12,15) = dt;
  KF.transitionMatrix.at<double>(13,16) = dt;
  KF.transitionMatrix.at<double>(14,17) = dt;
  KF.transitionMatrix.at<double>(9,15) = 0.5*pow(dt,2);
  KF.transitionMatrix.at<double>(10,16) = 0.5*pow(dt,2);
  KF.transitionMatrix.at<double>(11,17) = 0.5*pow(dt,2);
  
}

int writeCoordinates(serialib * serial,
                     const double & centimeters_x,
                     const double & centimeters_y,
                     const double & centimeters_z,
                     const double & speed_x,
                     const double & speed_y,
                     const double & speed_z) {

  // How many arrays should be written?
  int32_t lengthArrayInformation = 6 * sizeof(int32_t);
  uint8_t ar[lengthArrayInformation];
  int32_t *pointer = (int32_t*) ar;
  
  pointer[0] = int32_t(centimeters_x);
  pointer[1] = int32_t(centimeters_y);
  pointer[2] = int32_t(centimeters_z);

  pointer[3] = int32_t(speed_x);
  pointer[4] = int32_t(speed_y);
  pointer[5] = int32_t(speed_z);
  
  // Send data
  SendArray(serial->fd, ar, lengthArrayInformation, 1);

  return 0;
}


/**
 */
static bool isParam(string param, int argc, char **argv) {
    for(int i = 0; i < argc; i++)
        if(string(argv[i]) == param) return true;
    return false;
}


/**
 */
static string getParam(string param, int argc, char **argv, string defvalue = "") {
    int idx = -1;
    for(int i = 0; i < argc && idx == -1; i++)
        if(string(argv[i]) == param) idx = i;
    if(idx == -1 || (idx + 1) >= argc)
        return defvalue;
    else
        return argv[idx + 1];
}


/**
 */
static bool readCameraParameters(string filename, Mat &camMatrix, Mat &distCoeffs) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    return true;
}



/**
 */
static bool readDetectorParameters(string filename, aruco::DetectorParameters &params) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["adaptiveThreshWinSizeMin"] >> params.adaptiveThreshWinSizeMin;
    fs["adaptiveThreshWinSizeMax"] >> params.adaptiveThreshWinSizeMax;
    fs["adaptiveThreshWinSizeStep"] >> params.adaptiveThreshWinSizeStep;
    fs["adaptiveThreshConstant"] >> params.adaptiveThreshConstant;
    fs["minMarkerPerimeterRate"] >> params.minMarkerPerimeterRate;
    fs["maxMarkerPerimeterRate"] >> params.maxMarkerPerimeterRate;
    fs["polygonalApproxAccuracyRate"] >> params.polygonalApproxAccuracyRate;
    fs["minCornerDistanceRate"] >> params.minCornerDistanceRate;
    fs["minDistanceToBorder"] >> params.minDistanceToBorder;
    fs["minMarkerDistanceRate"] >> params.minMarkerDistanceRate;
    fs["doCornerRefinement"] >> params.doCornerRefinement;
    fs["cornerRefinementWinSize"] >> params.cornerRefinementWinSize;
    fs["cornerRefinementMaxIterations"] >> params.cornerRefinementMaxIterations;
    fs["cornerRefinementMinAccuracy"] >> params.cornerRefinementMinAccuracy;
    fs["markerBorderBits"] >> params.markerBorderBits;
    fs["perspectiveRemovePixelPerCell"] >> params.perspectiveRemovePixelPerCell;
    fs["perspectiveRemoveIgnoredMarginPerCell"] >> params.perspectiveRemoveIgnoredMarginPerCell;
    fs["maxErroneousBitsInBorderRate"] >> params.maxErroneousBitsInBorderRate;
    fs["minOtsuStdDev"] >> params.minOtsuStdDev;
    fs["errorCorrectionRate"] >> params.errorCorrectionRate;
    return true;
}


int main_loop(){
  // Settings


  bool send_over_serial = false;  // Send data to the drone via serial connection
  bool use_kalman = true;  // Use Kalman filter or raw values
  bool estimate_obj_pose = true;  // Object or cam pose estimation?


  // Counts how often in a row the kalman filter was not updated
  // If it is too large, the filter will be resetted
  int times_nothing_detected = 0;

  //if(!isParam("-w", argc, argv) || !isParam("-h", argc, argv) || !isParam("-l", argc, argv) ||
  //   !isParam("-s", argc, argv) || !isParam("-d", argc, argv) || !isParam("-c", argc, argv)) {

  //  help();

  //  return 0;
  //}

  //int markersX = atoi(getParam("-w", argc, argv).c_str());
  int markersX = 6;
  
  //    int markersY = atoi(getParam("-h", argc, argv).c_str());
  int markersY = 6;

  // float markerLength = (float)atof(getParam("-l", argc, argv).c_str());
  float markerLength = 0.2;

  // float markerSeparation = (float)atof(getParam("-s", argc, argv).c_str());
  float markerSeparation = 0.2;

  // int dictionaryId = atoi(getParam("-d", argc, argv).c_str());
  int dictionaryId = atoi("DICT_6x6_50=8");

  aruco::Dictionary dictionary =
        aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

  bool showRejected = false;
  //if(isParam("-r", argc, argv)) showRejected = true; 

  Mat camMatrix, distCoeffs;
  bool readOk = readCameraParameters("cam_char.yml", camMatrix, distCoeffs);
    if(!readOk) {
      cerr << "Invalid camera file" << endl;
      return 0;
    }

    aruco::DetectorParameters detectorParams;
    detectorParams.doCornerRefinement = true; // do corner refinement in markers

    bool refindStrategy = false;
    //if(isParam("-rs", argc, argv)) refindStrategy = true;

    // Open webcam
    VideoCapture inputVideo;
    int waitTime;
    int camId = 1;
    inputVideo.open(camId);

    waitTime = 10;

    float axisLength = 0.5f * ((float)min(markersX, markersY) * (markerLength + markerSeparation) +
                               markerSeparation);

    // create board object
    aruco::GridBoard board =
        aruco::GridBoard::create(markersX, markersY, markerLength, markerSeparation, dictionary);

    double totalTime = 0;
    int totalIterations = 0;

    // Save coordinates in CSV file
    FILE* camera_coordinates = fopen("camera_coordinates.csv", "w");
    std::fprintf(camera_coordinates, "%s,%s,%s\n", "x", "y", "z");
    
    // Declare what you need
    cv::FileStorage file("some_name.yml", cv::FileStorage::WRITE);


    // Initialise the serial connection
    serialib serial;
    
    if(send_over_serial){
      // Open serial port
      
      int ret = serial.Open("/dev/ttyUSB0", 1000000);
      
      if (ret != 1) {
        cerr << "Error while opening serial port. Permission problem?" << endl;
        return -1;
      }
    }

    // Kalman filter

    cv::KalmanFilter KF;         // instantiate Kalman Filter
    int n_states = 18;            // the number of states
    int n_measurements = 6;       // the number of measured states

    // TODO: Maybe I can add sth here later (for action control)
    int n_inputs = 0;             // the number of action controls
    double dt = 0.05;           // time between measurements (1/FPS)
    initKalmanFilter(KF, n_states, n_measurements, n_inputs, dt);    // init function

    // Measurements contain translations and rotations 
    Mat measurements(n_measurements, 1, CV_64F);
    measurements.setTo(Scalar(0));

    int fourcc = CV_FOURCC('M', 'J', 'P', 'G');
    float fps = inputVideo.get(CV_CAP_PROP_FPS);

    int frame_width = inputVideo.get(CV_CAP_PROP_FRAME_WIDTH);
    int frame_height = inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT);
    
    VideoWriter flying_logger = VideoWriter("flying_log.avi",
                                            fourcc, fps,
                                            Size(frame_width, frame_height),
                                            true);
    
    while(inputVideo.grab() && !stop) {

      Mat image, imageCopy;
        inputVideo.retrieve(image);

        double tick = (double)getTickCount();

        vector< int > ids;
        vector< vector< Point2f > > corners, rejected;

        // Translation and rotation vectors of the marker board
        Mat rvec, tvec;

        // detect markers
        aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);

        // refind strategy to detect more markers
        if(refindStrategy)
            aruco::refineDetectedMarkers(image, board, corners, ids, rejected, camMatrix,
                                         distCoeffs);

        // estimate board pose
        int markersOfBoardDetected = 0;
        if(ids.size() > 0)
            markersOfBoardDetected =
                aruco::estimatePoseBoard(corners, ids, board, camMatrix, distCoeffs, rvec, tvec);

        double currentTime = ((double)getTickCount() - tick) / getTickFrequency();
        totalTime += currentTime;
        totalIterations++;

        updateTransitionMatrix(KF, currentTime);
        
        if(totalIterations % 30 == 0) {
            cout << "Detection Time = " << currentTime * 1000 << " ms "
                 << "(Mean = " << 1000 * totalTime / double(totalIterations) << " ms)" << endl;
        }


        // Instantiate estimated translation and rotation
        Mat translation_estimated(3, 1, CV_64F);
        Mat rotation_estimated(3, 3, CV_64F);
        Mat speed_estimated(3, 1, CV_64F);


        // If the filter has not detected anything for 60 frames (ca. 2s) reset
        // it to (0, 0, 0)
        if (times_nothing_detected > 60) {
              initKalmanFilter(KF, n_states, n_measurements, n_inputs, dt);    // init function
        }
        
        // predict the Kalman filter state
        predictKalmanFilter(KF, translation_estimated, rotation_estimated, speed_estimated);

        // Update the Kalman fiter state if at least one marker was detected
        image.copyTo(imageCopy);

        // Increate counter for Kalman filter
        ++times_nothing_detected;
        
        if (ids.size() > 0) {

          // Reset counter
          times_nothing_detected = 0;
          
          aruco::drawDetectedMarkers(imageCopy, corners, ids);

            cv::Mat R;
            cv::Rodrigues(rvec, R);  // R is 3x3

            R = R.t();  // rotation of inverse

            cv::Mat tvec_cam;
            
            tvec_cam = - R * tvec; // translation of inverse

            if (estimate_obj_pose) {
              // Object pose estimation
              fillMeasurements(measurements, tvec, rvec);
            } else {
              // Camera pose estimation
              fillMeasurements(measurements, tvec_cam, rvec);
            }
            
            // update the Kalman filter with good measurements
            updateKalmanFilter(KF, measurements,
                               translation_estimated,
                               rotation_estimated,
                               speed_estimated);
            
        }


        // Speed data
        double speed_x = speed_estimated.at<double>(0) * 100;
        double speed_y = speed_estimated.at<double>(1) * 100;
        double speed_z = speed_estimated.at<double>(2) * 100;
        
        double speed_3d;
        
        speed_3d = sqrt(pow(speed_x, 2) + pow(speed_y, 2) + pow(speed_z, 2));
             
      std::cout << "Predicted pose: " << translation_estimated << std::endl << std::endl;
      std::cout << "Predicted rotation: " << rotation_estimated << std::endl;
      std::cout << "Predicted speed: " << speed_3d << " cm/s" << std::endl;

      if(send_over_serial){
        
        // Raw data
        // double centimetersX = tvec_cam.at<double>(1, 0) * 100;
        // double centimetersY = tvec_cam.at<double>(2, 0) * 100;
        // double centimetersZ = tvec_cam.at<double>(3, 0) * 100;
        
        // Kalman data
        double centimeters_x = translation_estimated.at<double>(0) * 100;
        double centimeters_y = translation_estimated.at<double>(1) * 100;
        double centimeters_z = translation_estimated.at<double>(2) * 100;
        
        writeCoordinates(&serial,
                         centimeters_x,
                         centimeters_y,
                         centimeters_z,
                         speed_x,
                         speed_y,
                         speed_z);
      }

      // Write coordinates to file
      std::fprintf(camera_coordinates, "%f,%f,%f\n", translation_estimated.at<double>(0),
                   translation_estimated.at<double>(1),
                   translation_estimated.at<double>(2));
      
      if(showRejected && rejected.size() > 0)
        aruco::drawDetectedMarkers(imageCopy, rejected, noArray(), Scalar(100, 0, 255));
      
      if (use_kalman) {         
        aruco::drawAxis(imageCopy, camMatrix, distCoeffs, rotation_estimated, translation_estimated, axisLength);
      } else {
        if(markersOfBoardDetected > 0) {
          aruco::drawAxis(imageCopy, camMatrix, distCoeffs, rvec, tvec, axisLength);
        }
      } 

      string user_n = getlogin();

      if (user_n == "pold87") {
        imshow("out", imageCopy);        
      } else {
      // Save video to file (on Odroid)
        flying_logger.write(imageCopy);
      }
      
      char key = static_cast<char>(waitKey(waitTime));
      if (key == 27) break;
    }

    fclose(camera_coordinates);
    return 0;

}


int main(int argc, char *argv[]) {
    // ...
    thread t(main_loop); // Separate thread for loop.

    // Wait for input character (this will suspend the main thread, but the loop
    // thread will keep running).
    cin.get();

    // Set the atomic boolean to true. The loop thread will exit from 
    // loop and terminate.
    stop = true;

    t.join();

    return EXIT_SUCCESS; 
}
