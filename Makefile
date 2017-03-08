CC := g++
CFLAGS := -c -O2 -std=c++11 -fpermissive -I./3rdparty -I./src -I. -I/usr/include/ -I/usr/include/opencv/ -I/usr/include/opencv2/
LDFLAGS := -lopencv_core -lopencv_highgui -lopencv_imgproc -lgsl -lgslcblas -lm -lpthread
TARGET := TSP
SRCDIR := ./src
SRCS := $(wildcard $(SRCDIR)/*.cpp)
OBJS := $(addprefix build/,$(notdir $(SRCS:.cpp=.o)))
OFCS := $(wildcard ./3rdparty/of_celiu/*.cpp)
OFCS := $(filter-out ./3rdparty/of_celiu/Coarse2FineTwoFrames.cpp, $(OFCS))
OFOBJS := $(addprefix build/,$(notdir $(OFCS:.cpp=.o)))

all:$(TARGET)

build/%.o: src/%.cpp
	$(CC) $(CFLAGS) $< -o $@

build/%.o: 3rdparty/of_celiu/%.cpp
	$(CC) $(CFLAGS) $< -o $@

$(TARGET): $(OBJS) $(OFOBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf $(TARGET) build/*.o
