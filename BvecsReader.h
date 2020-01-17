#ifndef _BVECS_READER_
#define _BVECS_RAEDER_
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

class BvecsReaderException : public std::runtime_error {
public:
  BvecsReaderException(const std::string &rFileName, // filename
                       unsigned int nLineNumber,     // line number
                       const std::string &rMessage   // error message
                       )
      : std::runtime_error(rFileName + ":" + std::to_string(nLineNumber) +
                           ": " + rMessage) {}
};

#define BR_REQUIRED(C)                                                         \
  do {                                                                         \
    if (!(C))                                                                  \
      throw BvecsReaderException(__FILE__, __LINE__, #C " is required!");      \
  } while (false)

#define BR_REQUIRED_MSG(C, M)                                                  \
  do {                                                                         \
    if (!(C))                                                                  \
      throw BvecsReaderException(__FILE__, __LINE__,                           \
                                 std::string(#C " is required! Message: ") +   \
                                     std::string(M));                          \
  } while (false)

// class for bvecs data
class BvecsReader {
public:
  BvecsReader(const char *filename) : _filename(filename), _cur_pos(0) {
    _inf.open(filename, std::ios::in | std::ios::binary);
    BR_REQUIRED_MSG(_inf.is_open(), "Opening \"" + _filename + "\" failed!");
    _getSize();
    _getDim();
    _sz_each = sizeof(int) + _dim;
    _n = _size / _sz_each;
  }
  // noncopyable
  BvecsReader(const BvecsReader &) = delete;
  BvecsReader &operator=(const BvecsReader &) = delete;

  // get  data dimension
  unsigned pointDimension() const { return _dim; }
  // total size in bytes
  size_t size() const { return _size; }
  // total number of points
  size_t numPoints() const { return _n; }

  // read <n> points starting from current position
  template <typename T = uint8_t> std::vector<T> read(size_t n) {
    size_t sz = n * _sz_each; // total size
    std::vector<char> buf(sz);
    _inf.read(&buf[0], sz);
    auto true_n = n;
    if (!_inf.good()) { // read failed
      size_t read_sz = _inf.gcount();
#ifdef DEBUG
      fprintf(stderr, "read %lu points failed, ONLY %lu was read\n", n,
              read_sz / _sz_each);
#endif
      BR_REQUIRED_MSG(read_sz % _sz_each == 0, "Bad bvecs file!");
      buf.resize(read_sz);
      true_n = read_sz / _sz_each;
    }
    _cur_pos += true_n; // update current

    std::vector<T> data(true_n * _dim);
    for (size_t i = 0, j = 0; i < data.size();) {
      j += sizeof(int); // skip dim part
      for (unsigned k = 0; k < _dim; ++k)
        data[i++] = static_cast<T>(buf[j++]);
    }

    return data;
  }

  // read from a-th point (including) until b-th point (not including)
  template <typename T = uint8_t>
  std::vector<T> read(size_t a, // first (including)
                      size_t b  // last (excluding)
  ) {
    BR_REQUIRED(b > a);
    if (a >= numPoints())
      return {};
    if (b > numPoints())
      b = numPoints();

    if (a != _cur_pos) { // starting is not _cur_pos
      size_t pos = a * _sz_each;
      if (!_seekTo(pos)) // <a> is too large
        return {};
    }

    return read<T>(b - a);
  }

  // read all remaining points starting from current position
  template <typename T = uint8_t> std::vector<T> read() {
    auto n = numPoints() - _cur_pos;
    if (n == 0)
      return {};
    return read<T>(n);
  }

  // seek to the begining of the file
  void rewind() {
    _inf.seekg(0, _inf.beg);
    _cur_pos = 0;
  }

private:
  // get data dimension from file
  void _getDim() {
    _inf.read((char *)&_dim, sizeof(unsigned));
    BR_REQUIRED_MSG(_inf.good(), "Read dimension failed");
    _inf.seekg(0, _inf.beg);
  }

  bool _seekTo(size_t pos) {
    if (pos > size() || (pos % _sz_each != 0))
      return false;
    rewind();
    _inf.seekg(pos);
    _cur_pos = pos / _sz_each;
    return true;
  }

  // size of the file in bytes
  void _getSize() {
    _inf.seekg(0, _inf.end);
    _size = _inf.tellg();
    _inf.seekg(0, _inf.beg);
  }

  std::string _filename; // filename
  size_t _cur_pos;       // current position (in points not bytes)
  std::ifstream _inf;    // file stream
  unsigned _dim;         // data dimension
  size_t _size;
  size_t _n;
  size_t _sz_each;
};

#endif // _BVECS_READER_
