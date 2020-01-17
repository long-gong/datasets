#ifndef _FVECS_READER_
#define _FVECS_RAEDER_
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

class FvecsReaderException : public std::runtime_error {
public:
  FvecsReaderException(const std::string &rFileName, // filename
                       unsigned int nLineNumber,     // line number
                       const std::string &rMessage   // error message
                       )
      : std::runtime_error(rFileName + ":" + std::to_string(nLineNumber) +
                           ": " + rMessage) {}
};

#define BR_REQUIRED(C)                                                         \
  do {                                                                         \
    if (!(C))                                                                  \
      throw FvecsReaderException(__FILE__, __LINE__, #C " is required!");      \
  } while (false)

#define BR_REQUIRED_MSG(C, M)                                                  \
  do {                                                                         \
    if (!(C))                                                                  \
      throw FvecsReaderException(__FILE__, __LINE__,                           \
                                 std::string(#C " is required! Message: ") +   \
                                     std::string(M));                          \
  } while (false)

// class for bvecs data
class FvecsReader {
public:
  FvecsReader(const char *filename) : _filename(filename), _cur_pos(0) {
    _inf.open(filename, std::ios::in | std::ios::binary);
    BR_REQUIRED_MSG(_inf.is_open(), "Opening \"" + _filename + "\" failed!");
    _getDim();
  }
  // noncopyable
  FvecsReader(const FvecsReader &) = delete;
  FvecsReader &operator=(const FvecsReader &) = delete;

  // get  data dimension
  unsigned getDimension() const { return _dim; }

  // read <n> points starting from current position
  template <typename T = float> std::vector<T> read(size_t n) {
    size_t sz = n * (1 + _dim);
    std::vector<float> buf(sz);
    _inf.read((char *)&buf[0], sz * sizeof(float));

    auto true_n = n;
    if (!_inf.good()) {
      const size_t sz_each = sizeof(float) * (1 + _dim);
      size_t read_sz = _inf.gcount();
      BR_REQUIRED_MSG(read_sz % sz_each == 0, "Bad bvecs file!");
      buf.resize(read_sz / sizeof(float));
      true_n = read_sz / sz_each;
    }
    _cur_pos += true_n;

    std::vector<T> data(true_n * _dim);
    for (size_t i = 0, j = 0; i < data.size();) {
      j += 1; // skip dim part
      for (unsigned k = 0; k < _dim; ++k)
        data[i++] = static_cast<T>(buf[j++]);
    }

    return data;
  }

  // read from a-th point (including) until b-th point (not including)
  template <typename T = float>
  std::vector<T> read(size_t a, // first (including)
                      size_t b  // last (excluding)
  ) {
    BR_REQUIRED(b > a);
    if (a != _cur_pos) {
      const sz_each = (1 + _dim) * sizeof(float);
      size_t pos = a * sz_each;
      if (!seekTo(pos))
        return {};
    }

    return read<T>(b - a);
  }

  // read all remaining points starting from current position
  template <typename T = float> std::vector<T> read() {
    auto n = numPoints() - _cur_pos;
    return read<T>(n);
  }

  void rewind() {
    _inf.seekg(_inf.beg);
    _cur_pos = 0;
  }

  size_t size() {
    auto old_pos = _inf.tellg();
    _inf.seekg(_inf.end);
    size_t sz = _inf.tellg();
    _inf.seekg(old_pos);
    return sz;
  }

  size_t numPoints() {
    const size_t sz_each = (1 + _dim) * sizeof(float);
    return size() / sz_each;
  }

private:
  bool seekTo(size_t pos) {
    auto old_pos = _inf.tellg();
    rewind();
    _inf.seekg(pos);

    if (!_inf.good()) {
      _inf.seekg(old_pos);
      return false;
    }

    const size_t sz_each = (1 + _dim) * sizeof(float);
    _cur_pos = pos / sz_each;
    return true;
  }

  // get data dimension from file
  void _getDim() {
    _inf.read((char *)&_dim, sizeof(unsigned));
    BR_REQUIRED_MSG(_inf.good(), "Read dimension failed");
    _inf.seekg(0, _inf.beg);
  }
  std::string _filename; // filename
  size_t _cur_pos;       // current position (in points not bytes)
  std::ifstream _inf;    // file stream
  unsigned _dim;         // data dimension
};

#endif // _FVECS_READER_