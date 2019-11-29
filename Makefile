run:
	g++ -Ofast face.cpp Sobel.cpp `pkg-config --cflags --libs opencv4`
	./a.out 0
	./a.out 1
	./a.out 2
	./a.out 3
	./a.out 4
	./a.out 5
	./a.out 6
	./a.out 7
	./a.out 8
	./a.out 9
	./a.out 10
	./a.out 11
	./a.out 12
	./a.out 13
	./a.out 14
	./a.out 15