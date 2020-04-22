BIN_DIR = .
CC = g++
CFLAGS = -std=c++11 $(shell pkg-config --cflags opencv)
LIBS = $(shell pkg-config --libs opencv)



$(BIN_DIR)/ellipse: ellipse.o
	${CC} -o $(BIN_DIR)/ellipse ellipse.o $(LIBS)

ellipse.o: ellipse.cpp
	${CC} $(CLFAGS) -c ellipse.cpp

clean:
	rm -f *.o
	rm -f $(BIN_DIR)/ellipse
    
