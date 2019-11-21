#include "Sobel.hpp"

using namespace cv;

int dx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

int dy[3][3] = {
    {-1, -2, -1},
    {0, 0, 0},
    {1, 2, 1}
};

Mat calculateDx(Mat &image) {
    return applyKernel(dx, image);
}

Mat calculateDy(Mat &image) {
    return applyKernel(dy, image);
}

Mat imageWrite(Mat &image, std::string imagename){
    Mat output = Mat(image.size(), image.type());
    normalize(image, output, 0, 255, NORM_MINMAX);

    imwrite(imagename, output);

    return output;
}

Mat calculateGradientMagnitude(Mat &dx, Mat &dy) {

    Mat mag = Mat(dx.size(), CV_32FC1);

     for(int x = 1; x < mag.rows - 1; x++) {	
		for(int y = 1; y < mag.cols - 1; y++) {
            float dxx = dx.at<float>(x,y);
            float dyy = dy.at<float>(x,y);

            mag.at<float>(x,y) =  sqrt(dxx*dxx + dyy*dyy);
        }
     }

     return mag;
}

Mat calculateGradientDirection(Mat &dx, Mat &dy) {

    Mat dir = Mat(dx.size(), CV_32FC1);

    for(int x = 1; x < dir.rows - 1; x++) {	
        for(int y = 1; y < dir.cols - 1; y++) {
            float result = atan2(dx.at<float>(x,y),  dy.at<float>(x,y));

            dir.at<float>(x,y) = result;
            float temp = dir.at<float>(x,y);
            // printf("Dir: %f, Temp: %f\n", result, temp);
        }
    }

    return dir;
}

Mat applyKernel(int kernel[3][3], Mat &originalImage) {

    Mat newImage = Mat(originalImage.size(), CV_32FC1);

    for(int x = 1; x < originalImage.rows - 1; x++) {	
		for(int y = 1; y < originalImage.cols - 1; y++) {
			float result = 0.0;

            for(int xOffset = -1; xOffset <= 1; xOffset++) {
                for(int yOffset = -1; yOffset <= 1; yOffset++) {
                    result += kernel[1 + xOffset][1 + yOffset] * originalImage.at<uchar>(x + xOffset, y + yOffset);
                }
            }

            newImage.at<float>(x, y) = result;
		}
	}

    return newImage;
}

uchar normaliseUcharGray(float max, float min, int x) {
    return (uchar) ((x + min) * 256.0 / (max-min) + min);
}

int ***malloc3dArray(int dim1, int dim2, int dim3)
{
    int i, j, k;
    int ***array = (int ***) malloc(dim1 * sizeof(int **));

    for (i = 0; i < dim1; i++) {
        array[i] = (int **) malloc(dim2 * sizeof(int *));
        for (j = 0; j < dim2; j++) {
            array[i][j] = (int *) malloc(dim3 * sizeof(int));
            memset(array[i][j], 0, dim3*sizeof(int));
        }   

    }
    return array;
}

int *** calculateHough(Mat& magnitude, Mat& direction, int radiusMax, int threshold) {
    int rows = magnitude.rows;
    int cols = magnitude.cols;

    int*** hough = malloc3dArray(rows, cols, radiusMax);

    for(int x = 1; x < magnitude.rows - 1; x++) {	
        for(int y = 1; y < magnitude.cols - 1; y++) {

            if(magnitude.at<float>(x,y) <= threshold)
                continue;

            float dir = direction.at<float>(x,y);

            for(int radius = 10; radius < radiusMax; radius++) {

                int x0p = x + radius * cos(dir);
                int x0m = x - radius * cos(dir);

                int y0p = y + radius * sin(dir);
                int y0m = y - radius * sin(dir);

                // X+ Y+
                if(x0p >= 0 && x0p < rows && y0p >= 0 && y0p < cols) {
                    hough[x0p][y0p][radius - 10] += 1;
                }

                // // X+ Y-
                // if(x0p >= 0 && x0p < rows && y0m >= 0 && y0m < cols) {
                //     hough[x0p][y0m][radius - 10] += 1;
                // }

                // // X- Y+
                // if(x0m >= 0 && x0m < rows && y0p >= 0 && y0p < cols) {
                //     hough[x0m][y0p][radius - 10] += 1;
                // }

                // X- Y-
                if(x0m >= 0 && x0m < rows && y0m >= 0 && y0m < cols) {
                    hough[x0m][y0m][radius - 10] += 1;
                }

            }
            
        }
    }

    return hough;
}


void visualiseHough(int ***hough, int rows, int cols, int radiusMax) {
    Mat houghImage = Mat(rows, cols, CV_32FC1, Scalar(0));

    for(int x = 1; x < rows-1; x++) {	
        for(int y = 1; y < cols-1; y++) {
            float result = 0;
            for(int r = 0; r < radiusMax; r++) {
                result += hough[x][y][r];
            }

            // Colapse the 3d Hough transform into 2d
            houghImage.at<float>(x,y) = result;
            float temp = houghImage.at<float>(x,y);
        }
    }

    houghImage = imageWrite(houghImage, "houghSpace.jpg");
    imwrite("houghSpace.jpg", houghImage);
}