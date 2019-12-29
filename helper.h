#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

template <typename PointType>
void tofile(const std::vector<PointType> &data, const std::string &filename,
            unsigned rows, bool andprint2std = true) {
  FILE *fp = fopen(filename.c_str(), "w");
  if (!fp) {
    perror("tofile() -- fopen() failed. ");
  }
  for (unsigned i = 0; i < rows; ++i) {
    fprintf(fp, "%d", i);
    for (auto j = 0; j < data[i].size(); ++j) {
      fprintf(fp, " %.6f", data[i](j));
      if (andprint2std) {
        fprintf(stdout, " %.6f", data[i](j));
      }
    }
    fprintf(fp, "\n");
    if (andprint2std) {
      fprintf(stdout, "\n");
    }
  }
}