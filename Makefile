CXX=g++
# CXXFLAGS=-I/usr/include/hdf5/serial -std=c++14 -O3
CXXFLAGS=-I/usr/include/hdf5/serial -std=c++14 -O3 -DDEBUG=1
LDFLAGS=-L /usr/lib/x86_64-linux-gnu/hdf5/serial/ -lxxhash -lhdf5
# RM=rm -rf
RM = gio trash -f

COMMON_HDRS=Hdf5File.h create_lsh_codes.h
TARGETS = audio glove enron mnist sift1m gist1m
TARGETS_FOR_LF = sift1b gist80m


all: $(TARGETS) $(TARGETS_FOR_LF)

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

sift1m: $(COMMON_HDRS) SIFT1M.cpp
	$(CXX) $(CXXFLAGS) SIFT1M.cpp -o $@ $(LDFLAGS)

gist1m: $(COMMON_HDRS) GIST1M.cpp
	$(CXX) $(CXXFLAGS) GIST1M.cpp -o $@ $(LDFLAGS)

sift1b: $(COMMON_HDRS) SIFT1B.cpp
	$(CXX) $(CXXFLAGS) SIFT1B.cpp -o $@ $(LDFLAGS)

gist80m: $(COMMON_HDRS) GIST80M.cpp
	$(CXX) $(CXXFLAGS) GIST80M.cpp -o $@ $(LDFLAGS)

gist80m-std: $(COMMON_HDRS) GIST80M-STD.cpp
	$(CXX) $(CXXFLAGS) GIST80M-STD.cpp -o $@ $(LDFLAGS)

run: $(TARGETS)
	./audio 192
	./glove 128
	./enron 1344
	./mnist 768
	./sift1m 128
	./gist1m 960


run-lf: $(TARGETS_FOR_LF)
	./sift1b 128
	./gist80m 384


clean:
	$(RM) *.o binary-sample $(TARGETS) $(TARGETS_FOR_LF)

distclean: clean 
	$(RM) *.h5