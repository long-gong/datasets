#ifndef __BINARYFILE_H_
#define __BINARYFILE_H_

#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

class BinaryFile {
public:
  enum class Mode { Read, Write, Append };
  explicit BinaryFile(const std::string &filename, Mode mode = Mode::Append);

  // non-copyable
  BinaryFile(const BinaryFile &) = delete;
  BinaryFile &operator=(const BinaryFile &) = delete;

  size_t read(void *ptr, size_t size, size_t count);

  void write(void *ptr, size_t size, size_t count);

  void seek(long int offset, int origin);

  long tell() const;

  size_t size();

  ~BinaryFile();

  static bool exists(const char *filename);

private:
  FILE *fp_;
};

#endif // __BINARYFILE_H_