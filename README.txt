# Building a Dartboard Dectector

Created using C++ with [OpenCV](http://opencv.org/).
Image detection for dartboards. Build by using techniques such as Viola-Jones and Circle, Intercention Hough spaces.
More information can be found in report.pdf

![Ouput image](output/dart2.jpg "Dart board detection in green, ground truth in red")

To compile Makefile

To run the program on an image:
`./a.out <imageName>`

e.g
`./a.out dart0`

This will output in detected.jpg

To run for all images:
`make runall`

Outputs will be found in the *output* directory.
