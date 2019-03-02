CFLAGS = -Wall -Wpedantic -std=c++11
CC = g++
SRC = main.cpp loader.cpp dataset.cpp encoder.cpp activation.cpp neural_net.cpp
OBJ = ${SRC:.cpp = .o}

main: $(OBJ)
	$(CC) $(CFLAGS) -o main $(OBJ)

clean:
	rm -f core *.o main
