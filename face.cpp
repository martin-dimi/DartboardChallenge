/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include <algorithm>

// Extra
#include <fstream>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/core/core.hpp"

// #include "opencv2/opencv.hpp"
#include <opencv2/opencv.hpp>

// #include "opencv2/highgui/highgui.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>

// #include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include <opencv2/core/utils/filesystem.hpp>

#include <iostream>
#include <stdio.h>

#include <Sobel.hpp>

using namespace std;
using namespace cv;

vector<DartboardLocation> loadGroundTruth(String path, Mat frame, String ind);
vector<DartboardLocation> detectViola( Mat frame, Mat frame_gray, vector<DartboardLocation> groundTruth);

map<int, float> calculateIOU(vector<DartboardLocation> trueFaces, vector<DartboardLocation> faces);
int getCorrectFaceCount(map<int, float> IOU, float IOUThreshold);

tuple<float, float> TPRandF1(int correctFaceCount, int groundTruthFaces, int predictedFaces);
tuple<float, float> calculatePerformance(Mat frame, Mat frame_gray, vector<DartboardLocation> groundTruth, vector<DartboardLocation> faces);

vector<DartboardLocation> calculateHoughSpace(Mat frame_gray, String name);
vector<DartboardLocation> getFacesPoints(vector<Rect> faces);
vector<DartboardLocation> calculateEstimatedPoints(vector<DartboardLocation> facePoints, vector<DartboardLocation> houghPoints);
void displayDetections(vector<DartboardLocation> locations, Mat frame, Scalar color);

/** Global variables */
String input_image_path = "images/";
String output_image_path = "output/";
String face_path = "GroundTruth_Face/";
String dart_path = "GroundTruth_Dart/";
String mag_path = "magnitudes/";
String dir_path = "directions/";
String circle_hough_path = "circleHoughs/";
String line_hough_path = "lineHoughs/";
String intersection_hough_path = "intersectionHoughs/";
String combined_hough_path = "combinedHoughs/";
String DartboardLocation_classifier = "dartcascade/cascade.xml";
String face_classifier = "frontalface.xml";
Scalar greenColor = Scalar( 0, 255, 0 );
Scalar redColor = Scalar( 0, 0, 255 );


CascadeClassifier cascade;

/** @function main */
int main( int argc, const char** argv )
{
	String name = argv[1];
	String image_path = input_image_path + name + ".jpg";

	bool isDetectingDartboard = true;
	String groundTruthPath = isDetectingDartboard ? dart_path : face_path;

	// Load the Strong Classifier in a structure called `Cascade'
	String cascadeName = isDetectingDartboard ? DartboardLocation_classifier : face_classifier;
	if(!cascade.load(cascadeName)) {
		printf("--(!)Error loading\n");
		return -1;
	} 

	// Prepare Image by turning it into Grayscale
	Mat frame = imread(image_path, IMREAD_COLOR);
	Mat frame_gray;
	cvtColor( frame, frame_gray, CV_BGR2GRAY );

	// Load ground truth and detect object with viola-jones
	vector<DartboardLocation> groundTruth = loadGroundTruth(groundTruthPath, frame, name);
	vector<DartboardLocation> facePoints = detectViola(frame, frame_gray, groundTruth);

	// Detect objects with hough spaces
	vector<DartboardLocation> houghPoints = calculateHoughSpace(frame_gray, name);
	vector<DartboardLocation> estimatedPoints = calculateEstimatedPoints(facePoints, houghPoints);

	// Display groundTruths with Red and predictions with Green
	displayDetections(estimatedPoints, frame, greenColor);
	displayDetections(groundTruth, frame, redColor);

	// Calculate perfomance
	tuple<float, float> performance = calculatePerformance(frame, frame_gray, groundTruth, estimatedPoints);
	float TPR = get<0>(performance);
	float F1  = get<1>(performance);
	cout << "Image: " << name << ", TPR: " << TPR << "%, F1: " << F1 << "%\n";

	// 4. Save Result Image
	String outputName = isDetectingDartboard ? "dart_" : "face_";
	imageWrite(frame, output_image_path + name);
	imageWrite(frame, "detected");

	return 0;
}

void displayDetections(vector<DartboardLocation> locations, Mat frame, Scalar color) {
	for(int i = 0; i < locations.size(); i++){	
		DartboardLocation loc = locations[i];
		int x = loc.x - loc.width/2;
		int y = loc.y - loc.height/2;

		cv::rectangle(frame, Point(x, y), Point(x + loc.width, y + loc.height), color, 2);
	}
}

vector<DartboardLocation> calculateHoughSpace(Mat frame_gray, String name) {
	int rows = frame_gray.rows;
    int cols = frame_gray.cols;
	int rmax = 200;
	int magThreshold = 200; 
	int houghThreshold = 160;

	Mat dxImage = calculateDx(frame_gray);
	Mat dyImage = calculateDy(frame_gray); 

	Mat gradientMag = calculateGradientMagnitude(dxImage, dyImage, magThreshold);
	Mat gradientDir = calculateGradientDirection(dxImage, dyImage);

	// imageWrite(dxImage, "dx.jpg");
	// imageWrite(dyImage, "dy.jpg");
	imageWrite(gradientMag, mag_path + name);
	imageWrite(gradientDir, dir_path + name);

	// Mat edges =  Mat(frame_gray.size(), CV_32FC1);
	// Canny(frame_gray, edges, 120, 120*3);
	// Mat e = imageWrite(edges, "Canny.jpg");

	// Create Hough Spaces
	int ***circleHough3d = calculateCircleHough(gradientMag, gradientDir, rmax);	
	Mat intersectionHough = calculateIntersectionHough(gradientMag, gradientDir, 20);
	Mat lineHough = calculateLineHough(gradientMag, gradientDir, 10);

	tuple<Mat, Mat, int**> houghOutput = combineHoughSpaces(circleHough3d, intersectionHough, rows, cols, rmax);
	int** radiusVotes = get<2>(houghOutput);
	Mat circleHough = get<1>(houghOutput);
	Mat combinedHough = get<0>(houghOutput);
	
	imageWrite(lineHough, line_hough_path + name);
	imageWrite(circleHough, circle_hough_path + name);
	imageWrite(intersectionHough, intersection_hough_path + name);
	imageWrite(combinedHough, combined_hough_path + name);

	return getCenterPoints(combinedHough, radiusVotes, houghThreshold, rmax, rmax);
}

vector<DartboardLocation> calculateEstimatedPoints(vector<DartboardLocation> facePoints, vector<DartboardLocation> houghPoints) {
	vector<DartboardLocation> points;

	for(std::size_t houghIt=0; houghIt<houghPoints.size(); ++houghIt) {
		DartboardLocation houghPoint = houghPoints[houghIt];
		double minDistance = DBL_MAX;
		DartboardLocation bestEstimate;

		for(std::size_t faceIt=0; faceIt<facePoints.size(); ++faceIt) {
			DartboardLocation facePoint = facePoints[faceIt];

			// Calculate euclidian difference
			float dist = DartboardLocation::calculateDistance(houghPoint, facePoint);
			if(dist < minDistance) {
				minDistance = dist;
				bestEstimate = DartboardLocation::getAverageLocation(houghPoint, facePoint);
			}
		}

		// cout << "Min dist: " << minDistance << "\n";
		points.insert(points.end(), bestEstimate);
	}
	return points;
}

vector<DartboardLocation> detectViola( Mat frame, Mat frame_gray, vector<DartboardLocation> groundTruth)
{
	vector<Rect> faces;

	// Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CASCADE_SCALE_IMAGE, Size(50, 50), Size(500,500) );

	return getFacesPoints(faces);
}

vector<DartboardLocation> getFacesPoints(vector<Rect> faces) {
	vector<DartboardLocation> points;

	for(std::size_t i=0; i<faces.size(); ++i) {
		Rect rect = faces[i];

		int center_X = rect.x + rect.width / 2;
		int center_Y = rect.y + rect.height / 2;

		points.insert(points.end(), DartboardLocation(center_X, center_Y, rect.width, rect.height));
	}

	return points;
}

map<int, float> calculateIOU(vector<DartboardLocation> trueFaces, vector<DartboardLocation> faces) {

	map<int, float> facesToIou;

	for(int faceNum=0; faceNum < trueFaces.size(); faceNum++) {
		DartboardLocation trueFace = trueFaces[faceNum];

		float maxIOU = -1;

	 	for(int decNum = 0; decNum < faces.size(); decNum++) {

			DartboardLocation decFace = faces[decNum];
	
			int xOverlap = max(0, min(trueFace.getRight(), decFace.getRight()) - max(trueFace.getLeft(), decFace.getLeft()));
			int yOverlap = max(0, min(trueFace.getBottom(), decFace.getBottom()) - max(trueFace.getTop(), decFace.getTop()));

			int overlapArea = xOverlap * yOverlap;

			int unionArea = trueFace.width * trueFace.height + decFace.width * decFace.height - overlapArea;

			float IOU = (float) overlapArea / (float) unionArea;

			if(IOU > maxIOU) maxIOU = IOU;

		}

		facesToIou[faceNum] = maxIOU;

	}

	return facesToIou;
}

int getCorrectFaceCount(map<int, float> IOU, float IOUThreshold){
	int counter = 0;
	for(map<int, float>::iterator it = IOU.begin(); it != IOU.end(); it++){
		if(it->second * 100 > IOUThreshold)  
			counter++;
	}

	return counter;
}

tuple<float, float> calculatePerformance(Mat frame, Mat frame_gray, vector<DartboardLocation> groundTruth, vector<DartboardLocation> faces) {
	// cout << "Calculating IOU for " << imageIndex << "\n";
	map<int, float> IOU = calculateIOU(groundTruth, faces);
	int threshold = 40;

	int correctFacesCount = getCorrectFaceCount(IOU, threshold);
	int groundTruthFaces = IOU.size();
	int predictedFaces = faces.size();

	tuple<float, float> derivations = TPRandF1(correctFacesCount, groundTruthFaces, predictedFaces);
	float TPR = get<0>(derivations) * 100.0f;
	float F1 =  get<1>(derivations) * 100.0f;

	return {TPR, F1};
}

tuple<float, float> TPRandF1(int correctFaceCount, int groundTruthFaces, int predictedFaces){
	float TP = correctFaceCount;
	float TN = 0;
	float FP = predictedFaces - correctFaceCount;
	float FN = groundTruthFaces - correctFaceCount;

	float TPR = TP + FN == 0 ? 1 : TP / (TP + FN);
	float F1 = (2.0f * TP + FP + FN) == 0 ? 1  : 2.0f * TP / (2.0f * TP + FP + FN);

	return {TPR, F1};
}

vector<DartboardLocation> loadGroundTruth(String path, Mat frame, String imageName) {

	vector<DartboardLocation> locations;
	String name = path + imageName + ".txt";
	ifstream file(name);

	if(file.is_open()) {
		String line;
		int lineNum = 0;
		while(getline(file, line)) {

			DartboardLocation location;

			stringstream stream(line);
			float input[5] =  {};

			for(int i=0; i<5; i++) {
				stream >> input[i];
			}

			location.width = input[3] * frame.cols;
			location.height = input[4] * frame.rows;
			location.x = input[1] * frame.cols;
			location.y = input[2] * frame.rows;

			locations.insert(locations.end(), location);

			lineNum++;
		}
	}
	file.close();

	return locations;
}

	