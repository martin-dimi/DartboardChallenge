#include "Sobel.hpp"

using namespace cv;
using namespace std;

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

    imwrite(imagename + ".jpg", output);

    return output;
}

Mat calculateGradientMagnitude(Mat &dx, Mat &dy, int threshold) {
    Mat mag = Mat(dx.size(), CV_32FC1);

     for(int x = 1; x < mag.rows - 1; x++) {	
		for(int y = 1; y < mag.cols - 1; y++) {
            float dxx = dx.at<float>(x,y);
            float dyy = dy.at<float>(x,y);

            float magnitude = sqrt(dxx*dxx + dyy*dyy);
            if(magnitude < threshold) {
                mag.at<float>(x,y) = 0;
            } else {
                mag.at<float>(x,y) = magnitude;
            }
        }
     }

     return mag;
}

Mat calculateGradientDirection(Mat &dx, Mat &dy) {

    Mat dir = Mat(dx.size(), CV_32FC1);

    for(int x = 1; x < dir.rows - 1; x++) {	
        for(int y = 1; y < dir.cols - 1; y++) {
            dir.at<float>(x,y) = atan2(dx.at<float>(x,y),  dy.at<float>(x,y));;
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

int **malloc2dArray(int dim1, int dim2)
{
    int i, j;
    int **array = (int **) malloc(dim1 * sizeof(int *));

    for (i = 0; i < dim1; i++) {
        array[i] = (int *) malloc(dim2 * sizeof(int));
        memset(array[i], 0, dim2*sizeof(int));
    }

    return array;
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

Mat calculateLineHough(Mat& magnitude, Mat& direction, float offset) {
    int rows = magnitude.rows;
    int cols = magnitude.cols;

    int imageDiagonal = sqrt(rows * rows + cols * cols);
    int anglePool = 180;

    Mat houghImage = Mat(imageDiagonal * 2, anglePool, CV_32FC1, Scalar(0));

    for(int row = 1; row < magnitude.rows - 1; row ++) {	
        for(int col = 1; col < magnitude.cols - 1; col ++) {

            if(magnitude.at<float>(row,col) == 0) {
                continue;
            }

            float dir = direction.at<float>(row,col) * 180.0f / 3.14f;
            if(dir < 0) dir += 180;

            for(int thetha = dir - offset; thetha < dir + offset; thetha++) {

                float thethaRadian = thetha * 3.14f / 180.0f;

                int rho = col * cos(thethaRadian) + row * sin(thethaRadian);

                rho += imageDiagonal;

                if(rho > 0 && rho < imageDiagonal * 2)
                    houghImage.at<float>(rho, thetha) += 1;
            }
        }
    }

    return houghImage;
}

Mat calculateIntersectionHough(Mat& magnitude, Mat& direction, float offset) {
    int rows = magnitude.rows;
    int cols = magnitude.cols;

    Mat houghImage = Mat(rows, cols, CV_32FC1, Scalar(0));

    for(int row = 1; row < magnitude.rows - 1; row ++) {	
        for(int col = 1; col < magnitude.cols - 1; col ++) {

            if(magnitude.at<float>(row,col) <= 0) {
                continue;
            }
            const float PI = 3.14f;

            float dir = direction.at<float>(row,col);
            float dirM = PI - dir;   

            for(int delta = 0; delta < offset; delta++) {
                if(abs(dir) < PI/2) {
                    int xOffset = col + delta * cos(dirM);
                    int yOffset = row + delta * sin(dirM);

                    if(xOffset > 0 && xOffset < cols && yOffset > 0 && yOffset < rows) {
                        houghImage.at<float>(yOffset,xOffset) += 1;
                    }

                    xOffset = col - delta * cos(dirM);
                    yOffset = row - delta * sin(dirM);

                    if(xOffset > 0 && xOffset < cols && yOffset > 0 && yOffset < rows){
                        houghImage.at<float>(yOffset,xOffset) += 1;
                    }
                }
                else {
                    int xOffset = col + delta * cos(dir);
                    int yOffset = row - delta * sin(dir);

                    if(xOffset > 0 && xOffset < cols && yOffset > 0 && yOffset < rows) {
                        houghImage.at<float>(yOffset,xOffset) += 1;
                    }

                    xOffset = col - delta * cos(dir);
                    yOffset = row + delta * sin(dir);

                    if(xOffset > 0 && xOffset < cols && yOffset > 0 && yOffset < rows){
                        houghImage.at<float>(yOffset,xOffset) += 1;
                    }
                }
            }
        }
    }

    return houghImage;
}

int *** calculateCircleHough(Mat& magnitude, Mat& direction, int radiusMax) {
    int rows = magnitude.rows;
    int cols = magnitude.cols;

    int*** hough = malloc3dArray(rows, cols, radiusMax);

    for(int x = 1; x < magnitude.rows - 1; x++) {	
        for(int y = 1; y < magnitude.cols - 1; y++) {

            if(magnitude.at<float>(x,y) == 0) {
                continue;
            }

            float dir = direction.at<float>(x,y);

            for(int radius = 10; radius < radiusMax; radius+=2) {

                int x0p = x + radius * cos(dir);
                int x0m = x - radius * cos(dir);

                int y0p = y + radius * sin(dir);
                int y0m = y - radius * sin(dir);

                // X+ Y+
                if(x0p >= 0 && x0p < rows && y0p >= 0 && y0p < cols) {
                    hough[x0p][y0p][radius - 10] += 1;
                }

                // X- Y-
                if(x0m >= 0 && x0m < rows && y0m >= 0 && y0m < cols) {
                    hough[x0m][y0m][radius - 10] += 1;
                }

            }
            
        }
    }

    return hough;
}

tuple<Mat, Mat, int**> combineHoughSpaces(int ***hough,  Mat intersectionHough, int rows, int cols, int radiusMax) {
    Mat circleHough = Mat(rows, cols, CV_32FC1, Scalar(0));
    int **maxVotesRadius = malloc2dArray(rows, cols);
    int size = 1;

    for(int x = 1; x < rows-1; x++) {	
        for(int y = 1; y < cols-1; y++) {
            float result = 0;

            int maxVotes[size] = {};
            int rad[size] = {};

            for(int r = 15; r < radiusMax; r++){
                int votes = hough[x][y][r];
                result += votes;
                int i = 0;
                if(votes > maxVotes[0]){
                    i = 1;
                    while(votes > maxVotes[i] && i < size) i++;
                    for(int j = 0; j < i - 1; j++){
                        maxVotes[j] = maxVotes[j + 1];
                        rad[j] = rad[j + 1];
                    }
                    maxVotes[i - 1] = votes;
                    rad[i - 1] = r;
                }
            }

            int radiusSum = 0;
            int i = size - 1;
            while(i >= 0 && maxVotes[i] > 0){
                radiusSum += rad[i];
                i--;
            }

            maxVotesRadius[x][y] = radiusSum / (size - i - 1);

            // Colapse the 3d Hough transform into 2d
            circleHough.at<float>(x,y) = result;
        }
    }


    normalize(circleHough, circleHough, 0, 255, NORM_MINMAX);
    normalize(intersectionHough, intersectionHough, 0, 255, NORM_MINMAX);

    Mat combinedHough = 1 * circleHough + 3 * intersectionHough;

    normalize(combinedHough, combinedHough, 0, 255, NORM_MINMAX);

    return {combinedHough, circleHough, maxVotesRadius};
}

// Extracts the center points from the given Hough Images
vector<DartboardLocation> getCenterPoints(Mat houghImage, int** radiusVotes, int threshold, int deletionLengthX, int deletionLengthY) {

    int rows = houghImage.rows;
    int cols = houghImage.cols;

    vector<DartboardLocation> locations;
    bool found;

    do {
        float max = -1;
        int locX;
        int locY;
        found = false;

        for(int x = 1; x < rows-1; x++) {	
            for(int y = 1; y < cols-1; y++) {
                float value = houghImage.at<float>(x,y);

                if(value > max) {
                    max = value;
                    locX = x;
                    locY = y;
                }

            }
        }

        if(max >= threshold) {
            // Save the location coordinates
            int radius = radiusVotes[locX][locY];
            DartboardLocation loc = DartboardLocation(locY, locX, radius*2, radius*2);
            locations.insert(locations.end(), loc);
            found = true;

            // Clear the radius
            for(int x = locX - deletionLengthX/2; x < locX + deletionLengthX/2; x++) {
                // check if x is within bounds
                if(x < 1) {
                    x = 1;
                    continue;
                } else if (x > rows-1) break;
                
                for(int y = locY - deletionLengthY/2; y < locY + deletionLengthY/2; y++) {
                    // check if y is within bounds
                    if(y < 1) {
                        y = 1;
                        continue;
                    } else if (y > cols-1) break;

                    houghImage.at<float>(x,y) = 0;
                }
            }
        }
     } while (found);

    return locations;
}
