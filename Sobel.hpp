#include <stdio.h>
#include "includes.h"

#define _USE_MATH_DEFINES

#include <math.h> 

using namespace cv;

int ***malloc3dArray(int dim1, int dim2, int dim3);

Mat calculateGradientMagnitude(Mat &dx, Mat &dy);
Mat calculateGradientDirection(Mat &dx, Mat &dy);
int ***calculateHough(Mat& magnitude, Mat& direction, int radiusMax, int threshold);
void visualiseHough(int ***hough, int rows, int cols, int radiusMax);

Mat calculateDx(Mat &image);
Mat calculateDy(Mat &image);

Mat applyKernel(int kernel[3][3], Mat &originalImage);
uchar normaliseUcharGray(float max, float min, int x);
Mat imageWrite(Mat &image, std::string imagename);