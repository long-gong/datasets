#include "BinaryFile.h"
#include <iostream>

BinaryFile::BinaryFile(const std::string &filename, BinaryFile::Mode mode)
    : fp_(nullptr) {
  switch (mode) {
  case Mode::Read:
    fp_ = fopen(filename.c_str(), "rb");
    break;
  case Mode::Write:
    if (exists(filename.c_str())) {
      std::cerr << "File \"" << filename
                << "\" exists. Write mode does not support overwrite "
                   "currently, please try Append mode\n";
    } else {
      fp_ = fopen(filename.c_str(), "wb");
    }
    break;
  case Mode::Append:
    fp_ = fopen(filename.c_str(), "ab");
    break;
  }

  if (fp_ == nullptr) {
    fputs("File error", stderr);
  }
}

size_t BinaryFile::read(void *ptr, size_t size, size_t count) {
  if (count == 0)
    return 0;

  if (fp_ != nullptr)
    return fread(ptr, size, count, fp_);

  return 0;
}

void BinaryFile::seek(long int offset, int origin) {

  if (fp_ != nullptr)
    fseek(fp_, offset, origin);
}

long BinaryFile::tell() const {
  if (fp_ != nullptr)
    return ftell(fp_);
  else
    return -1;
}

BinaryFile::~BinaryFile() {
  if (fp_ != nullptr) {
    fclose(fp_);
  }
}

void BinaryFile::write(void *ptr, size_t size, size_t count) {
  if (fp_ != nullptr)
    fwrite(ptr, size, count, fp_);
}

bool BinaryFile::exists(const char *filename) {
  FILE *fp = fopen(filename, "rb");
  bool ex = (fp != nullptr);
  if (ex) {
    fclose(fp);
  }
  return ex;
}

size_t BinaryFile::size() {
  auto cur_pos = tell();
  seek(0, SEEK_END);
  auto size = tell();
  seek(cur_pos, SEEK_SET);
  return size;
}