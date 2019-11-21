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

#include <iostream>
#include <stdio.h>

#include <Sobel.hpp>

using namespace std;
using namespace cv;

/** Function Headers */
vector<Rect> detectAndDisplay(Mat frame, Mat frame_gray, float locations[16][15][4], int imageIndex);

void loadGroundTruth(float locations[16][15][4], String path);

map<int, float> calculateIOU(float trueFaces[15][4], vector<Rect> faces);

int getCorrectFaceCount(map<int, float> IOU, float IOUThreshold);

tuple<float, float> TPRandF1(int correctFaceCount, int groundTruthFaces, int predictedFaces);

tuple<float, float> calculatePerformance(Mat frame, Mat frame_gray, float groundTruth[15][4], vector<Rect> faces);

void calculateHoughSpace(Mat frame_gray);

/** Global variables */
String input_image_path = "images/";
String output_image_path = "output/";
String face_path = "GroundTruth_Face/";
String dart_path = "GroundTruth_Dart/";

String dartboard_classifier = "dartcascade/cascade.xml";
String face_classifier = "frontalface.xml";

CascadeClassifier cascade;

/** @function main */
int main( int argc, const char** argv )
{
	bool isDartboard = true;

	float groundTruth[16][15][4] = {};
	String groundTruthPath = isDartboard ? dart_path : face_path;
	loadGroundTruth(groundTruth, groundTruthPath);

	// The average TPR and F1 of all images
	float overallTPR = 0;
	float overallF1  = 0;

	// Load the Strong Classifier in a structure called `Cascade'
	String cascadeName = isDartboard ? dartboard_classifier : face_classifier;
	if( !cascade.load( cascadeName ) ){ printf("--(!)Error loading\n"); return -1; };

	int ind = 13;
	for(int imageIndex = ind; imageIndex < ind + 1; imageIndex++) {
		// Prepare Image by turning it into Grayscale and normalising lighting
		String name = "dart" + to_string(imageIndex) + ".jpg";
		String image_path = input_image_path + name;
		Mat frame = imread(image_path, IMREAD_COLOR);
		Mat frame_gray;
		cvtColor( frame, frame_gray, CV_BGR2GRAY );
		equalizeHist( frame_gray, frame_gray );

		// Detect Faces and Display Result
		vector<Rect> faces = detectAndDisplay( frame, frame_gray, groundTruth, imageIndex);

		// Calculate perfomance
		tuple<float, float> performance = calculatePerformance(frame, frame_gray, groundTruth[imageIndex], faces);
		float TPR = get<0>(performance);
		float F1  = get<1>(performance);
		cout << "Image: " << imageIndex << ", TPR: " << TPR << "%, F1: " << F1 << "%\n";

		calculateHoughSpace(frame_gray);

		// 4. Save Result Image
		String outputName = isDartboard ? "dart_" : "face_";
		imwrite(output_image_path + outputName + name, frame );
	}
	// overallTPR /= 16;
	// overallF1  /= 16;

	// printf("Overall TRP: %f%, overall F1: %f%", overallTPR, overallF1);

	return 0;
}

tuple<float, float> calculatePerformance(Mat frame, Mat frame_gray, float groundTruth[15][4], vector<Rect> faces) {
	// cout << "Calculating IOU for " << imageIndex << "\n";
	map<int, float> IOU = calculateIOU(groundTruth, faces);

	int correctFacesCount = getCorrectFaceCount(IOU, 40.0f);
	int groundTruthFaces = IOU.size();
	int predictedFaces = faces.size();

	tuple<float, float> derivations = TPRandF1(correctFacesCount, groundTruthFaces, predictedFaces);
	float TPR = get<0>(derivations) * 100.0f;
	float F1 =  get<1>(derivations) * 100.0f;

	return {TPR, F1};
}

void calculateHoughSpace(Mat frame_gray) {
	// Calculating Dx and Dy
	Mat dxImage = calculateDx(frame_gray);
	Mat dyImage = calculateDy(frame_gray); 
	Mat gradientMag = calculateGradientMagnitude(dxImage, dyImage);
	Mat gradientDir = calculateGradientDirection(dxImage, dyImage);

	int rows = frame_gray.rows;
    int cols = frame_gray.cols;
	int rmax = 200;
	int magThreshold = 200; 
	int houghThreshold = 200;

	imageWrite(dxImage, "dx.jpg");
	imageWrite(dyImage, "dy.jpg");
	imageWrite(gradientMag, "gradientMag.jpg");
	imageWrite(gradientDir, "gradientDir.jpg");

	int ***hough = calculateHough(gradientMag, gradientDir, rmax, magThreshold);
	Mat houghImage = visualiseHough(hough, rows, cols, rmax);

	vector<tuple<int, int>> points = getCenterPoints(houghImage, houghThreshold, rmax, rmax);

	cout << "Number of points: " << points.size();
}

void loadGroundTruth(float locations[16][15][4], String path) {

	for(int nameNum = 0; nameNum <= 15; nameNum++) {

		String name = path + "dart" + to_string(nameNum) + ".txt";
		ifstream file(name);

		if(file.is_open()) {
			String line;
			int lineNum = 0;
			while(getline(file, line)) {

				stringstream stream(line);
				float loc[5] =  {};

				for(int i=0; i<5; i++) {
					stream >> loc[i];
				}

				for(int i=0; i<4; i++) {
					locations[nameNum][lineNum][i] = loc[i+1];
				}
				lineNum++;
			}
		}
		file.close();
	}
}

/** @function detectAndDisplay */
vector<Rect> detectAndDisplay( Mat frame, Mat frame_gray, float locations[16][15][4], int imageIndex)
{
	vector<Rect> faces;

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CASCADE_SCALE_IMAGE, Size(50, 50), Size(500,500) );

	// 3. Print number of Faces found
	// cout << faces.size() << std::endl;

	// 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}

	for(int i = 0; i < 15; i++){
		float truthX = locations[imageIndex][i][0] * frame_gray.cols;
		float truthY = locations[imageIndex][i][1] * frame_gray.rows;
		float truthW = locations[imageIndex][i][2] * frame_gray.cols;
		float truthH = locations[imageIndex][i][3] * frame_gray.rows;

		locations[imageIndex][i][0] = truthX;
		if(truthX == 0) break;
		locations[imageIndex][i][1] = truthY;
		locations[imageIndex][i][2] = truthW;
		locations[imageIndex][i][3] = truthH;

		float x = truthX - truthW / 2.0f;
		float y = truthY - truthH / 2.0f;
		
		rectangle(frame, Point(x, y), Point(x + truthW, y + truthH), Scalar( 0, 0, 255 ), 2);
	}

	return faces;

}

map<int, float> calculateIOU(float trueFaces[15][4], vector<Rect> faces) {

	map<int, float> facesToIou;

	for(int faceNum=0; faceNum < 15; faceNum++) {
		float *trueFace = trueFaces[faceNum];
		if(trueFace[0] == 0) break;

		float maxIOU = -1;

	 	for(int decNum = 0; decNum < faces.size(); decNum++) {

			Rect decFace = faces[decNum];

			int trueRight = trueFace[0] + trueFace[2] / 2.0;
			int trueLeft = trueFace[0] - trueFace[2] / 2.0;
			int trueBottom = trueFace[1] + trueFace[3] / 2.0;
			int trueTop = trueFace[1] - trueFace[3] / 2.0;
	
			int xOverlap = max(0, min(trueRight, decFace.x + decFace.width) - max(trueLeft, decFace.x));
			int yOverlap = max(0, min(trueBottom, decFace.y + decFace.height) - max(trueTop, decFace.y));

			int overlapArea = xOverlap * yOverlap;

			int unionArea = trueFace[2] * trueFace[3] + decFace.width * decFace.height - overlapArea;

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

tuple<float, float> TPRandF1(int correctFaceCount, int groundTruthFaces, int predictedFaces){
	float TP = correctFaceCount;
	float TN = 0;
	float FP = predictedFaces - correctFaceCount;
	float FN = groundTruthFaces - correctFaceCount;

	float TPR = TP + FN == 0 ? 1 : TP / (TP + FN);
	float F1 = (2.0f * TP + FP + FN) == 0 ? 1  : 2.0f * TP / (2.0f * TP + FP + FN);

	return {TPR, F1};
}

	