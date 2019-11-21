run:
	g++ -Ofast face.cpp Sobel.cpp `pkg-config --cflags --libs opencv4`
	./a.out