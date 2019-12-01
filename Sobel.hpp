#include <stdio.h>
#include "includes.h"
#include "DartboardLocation.cpp"

#define _USE_MATH_DEFINES

#include <math.h> 

using namespace cv;
using namespace std;

int **malloc2dArray(int dim1, int dim2);
int ***malloc3dArray(int dim1, int dim2, int dim3);
Mat applyKernel(int kernel[3][3], Mat &originalImage);
uchar normaliseUcharGray(float max, float min, int x);
Mat imageWrite(Mat &image, std::string imagename);

Mat calculateDx(Mat &image);
Mat calculateDy(Mat &image);
Mat calculateGradientMagnitude(Mat &dx, Mat &dy, int threshold);
Mat calculateGradientDirection(Mat &dx, Mat &dy);

Mat calculateLineHough(Mat& magnitude, Mat& direction, float offset);
Mat calculateIntersectionHough(Mat& magnitude, Mat& direction, float offset);
int ***calculateCircleHough(Mat& magnitude, Mat& direction, int radiusMax);
tuple<Mat, Mat, int**> combineHoughSpaces(int ***hough,  Mat intersectionHough, int rows, int cols, int radiusMax);

vector<DartboardLocation> getCenterPoints(Mat houghImage, int** radiusVotes, int threshold, int deletionLengthX, int deletionLengthY);