CXX=g++
CXXFLAGS=-I/usr/include/hdf5/serial -std=c++14 -O3
LDFLAGS=-L /usr/lib/x86_64-linux-gnu/hdf5/serial/ -lxxhash -lhdf5
RM=rm -rf

COMMON_HDRS=Hdf5File.h create_lsh_codes.h
TARGETS = audio glove enron mnist

all: $(TARGETS)

%.o: %.cpp 
	$(CXX) -c -o $@ $< $(CXXFLAGS)

binary-sample: BinaryFileSample.o BinaryFile.o
	$(CXX) -o $@ $^ $(CXXFLAGS)

audio: $(COMMON_HDRS) Audio.cpp
	$(CXX) $(CXXFLAGS) Audio.cpp -o $@ $(LDFLAGS)

glove: $(COMMON_HDRS) Glove.cpp
	$(CXX) $(CXXFLAGS) Glove.cpp -o $@ $(LDFLAGS)

enron: $(COMMON_HDRS) Enron.cpp
	$(CXX) $(CXXFLAGS) Enron.cpp -o $@ $(LDFLAGS)

mnist: $(COMMON_HDRS) Mnist.cpp
	$(CXX) $(CXXFLAGS) Mnist.cpp -o $@ $(LDFLAGS)

run: $(TARGETS)
	./audio 192
	./glove 128
	./enron 1344
	./mnist 768

clean:
	$(RM) *.o binary-sample audio glove enron mnist 

distclean: clean 
	$(RM) *.h5