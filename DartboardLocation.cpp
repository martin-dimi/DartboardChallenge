#include <math.h>
#include <sstream>

using namespace std;

class DartboardLocation {

    public:
    int x;
    int y;
    int width;
    int height;

    DartboardLocation() {
        this->x = 0;
        this->y = 0;
        this->width = 0;
        this->height = 0;
    }    

    DartboardLocation(int x, int y):DartboardLocation() {
        this->x = x;
        this->y = y;
    }    

    DartboardLocation(int x, int y, int width, int height) {
        this->x = x;
        this->y = y;
        this->width = width;
        this->height = height;
    }    

    int getLeft() {
        return x - width/2;
    }

    int getRight() {
        return x + width/2;
    }

    int getBottom() {
        return y + height/2;
    }

    int getTop() {
        return y - height/2;
    }

    string to_string() {
        ostringstream strout;
        strout << "X: " << x << ", Y:" << y;

        return strout.str();
    }

    // Calculate the Euclidean Distance
    static float calculateDistance(DartboardLocation point1, DartboardLocation point2) {
        return sqrt(pow(point1.x - point2.x, 2) + pow(point1.y - point2.y, 2));
    }

    // Get the average point between two given
    static DartboardLocation getAverageLocation(DartboardLocation point1, DartboardLocation point2) {
        int x = point1.x;
        int y = point1.y;

        int width = (point1.width + point2.width) / 2;
        int height = (point1.height + point2.height) / 2;

        return DartboardLocation(x, y, width, height);
    }
};