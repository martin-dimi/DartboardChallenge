run:
	g++ -Ofast face.cpp Sobel.cpp `pkg-config --cflags --libs opencv4`

runall:
	g++ -Ofast face.cpp Sobel.cpp `pkg-config --cflags --libs opencv4`

	i=0; while [[ $$i -le 15 ]]; do \
		./a.out "dart$$i"; \
		((i = i + 1)); \
	done