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

using namespace std;
using namespace cv;

/** Function Headers */
vector<Rect> detectAndDisplay(Mat frame, float locations[16][15][4], int imageIndex);

void loadGroundTruth(float locations[16][15][4]);

map<int, float> calculateIOU(float trueFaces[15][4], vector<Rect> faces);

int getCorrectFaceCount(map<int, float> IOU, float IOUThreshold);

tuple<float, float> TPRandF1(int correctFaceCount, int groundTruthFaces, int predictedFaces);

/** Global variables */
String cascade_name = "dartcascade/stage2.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv )
{

	float locations[16][15][4] = {};
	loadGroundTruth(locations);

	for(int imageIndex = 0; imageIndex <= 15; imageIndex++) {
		// 1. Read Input Image
		String name = "dart" + to_string(imageIndex) + ".jpg";
		Mat frame = imread(name, IMREAD_COLOR);

		// 2. Load the Strong Classifier in a structure called `Cascade'
		if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

		// 3. Detect Faces and Display Result
		vector<Rect> faces = detectAndDisplay( frame, locations, imageIndex);
		// cout<< faces.size();

		// cout << "Calculating IOU for " << imageIndex << "\n";
		map<int, float> IOU = calculateIOU(locations[imageIndex], faces);

		int correctFacesCount = getCorrectFaceCount(IOU, 40.0f);

		int groundTruthFaces = IOU.size();

		int predictedFaces = faces.size();

		tuple<float, float> derivations = TPRandF1(correctFacesCount, groundTruthFaces, predictedFaces);

		float TPR = get<0>(derivations) * 100.0f;
		float F1 =  get<1>(derivations) * 100.0f;

		cout << "Image: " << imageIndex << ", TPR: " << TPR << "%, F1: " << F1 << "%\n";

		// 4. Save Result Image
		imwrite( "compared_" + name, frame );
	}

	return 0;
}

void loadGroundTruth(float locations[16][15][4]) {

	for(int nameNum = 0; nameNum <= 15; nameNum++) {

		String name = "GroundTruth/dart" + to_string(nameNum) + ".txt";
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
vector<Rect> detectAndDisplay( Mat frame, float locations[16][15][4], int imageIndex)
{
	Mat frame_gray;
	vector<Rect> faces;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

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

	