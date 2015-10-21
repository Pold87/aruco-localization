create_marker create_marker.cpp:
	g++ -g create_marker.cpp -o create_marker `pkg-config --cflags --libs opencv` -std=c++11
