CC = /u/sw/toolchains/gcc-glibc/11.2.0/base/bin/mpic++
CFLAGS = -I. -Ishared-folder/try -Wall -Werror -std=c++17 -O3 -fopenmp

DOXYFILE = Doxyfile

SRCS = main.cpp 
HDRS = chrono.hpp
OBJS = $(SRCS:.cpp=.o)

all: main

main: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@

%.o: %.cpp $(HDRS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) main

doc:
	doxygen $(DOXYFILE)