#include "create_lsh_codes.h"
#include <eigen3/Eigen/Dense>
#include <random>
#include <unordered_set>
#include <vector>
#include <xxhash.h>
#include <cassert>
#include "Hdf5File.h"

using namespace std;
using namespace Eigen;

const char *FOLDER_NAME = "datasets/audio";
const int NUM_QUERIES = 200;
const int SEED = 4057218;
const unsigned C_SEED = 91023221u;
XXH64_hash_t const H_SEED = 0; /* or any other value */

/** THE FOLLOWING SEVERAL FUNCTIONS WERE SHAMELESSLY COPIED
 *   FROM https://github.com/FALCONN-LIB/FALCONN **/

using Point = VectorXd;
/*
 * An auxiliary function that reads a point from a binary file that is produced
 * by a script 'prepare-dataset.sh'
 */
bool read_point(FILE *file, int d, Point *point) {
  /*
  int d;
  if (fread(&d, sizeof(int), 1, file) != 1) {
    return false;
  }
  */
  float *buf = new float[d];
  if (fread(buf, sizeof(float), d, file) != (size_t)d) {
    delete[] buf;
    return false;
  }
  point->resize(d);
  for (int i = 0; i < d; ++i) {
    (*point)[i] = buf[i];
  }
  delete[] buf;
  return true;
}

/*
 * An auxiliary function that reads a dataset from a binary file that is
 * produced by a script 'prepare-dataset.sh'
 */
void read_dataset(const string &file_name, vector<Point> *dataset) {
  FILE *file = fopen(file_name.c_str(), "rb");
  if (!file) {
    throw runtime_error("can't open the file with the dataset");
  }

  // read header first
  unsigned header[3]; /* entry size, row, col */
  assert(fread(header, sizeof(unsigned), 3, file) == 3);
#ifdef DEBUG
  fprintf(stdout, "entry size: %d, row: %d, col: %d\n", header[0], header[1],
          header[2]);

#endif
  Point p;
  dataset->clear();
  while (!feof(file) && dataset->size() < header[1] &&
         read_point(file, header[2], &p)) {
    dataset->push_back(p);
  }
  if (fclose(file)) {
    throw runtime_error("fclose() error");
  }
}

Point cal_center(const vector<Point> &dataset) {
  // find the center of mass
  Point center = dataset[0];
  for (size_t i = 1; i < dataset.size(); ++i) {
    center += dataset[i];
  }
  center /= dataset.size();

  return center;
}

void recenter(vector<Point> &dataset, const Point &center) {
  for (size_t i = 1; i < dataset.size(); ++i) {
    dataset[i] -= center;
  }
}

/*
 * Chooses a random subset of the dataset to be the queries. The queries are
 * taken out of the dataset.
 */
void gen_queries(vector<uint64_t> *dataset, vector<uint64_t> *queries,
                 int enc_dim) {
  mt19937_64 gen(SEED);
  queries->clear();
  auto n = dataset->size() / enc_dim;

  for (int i = 0; i < NUM_QUERIES; ++i) {
    uniform_int_distribution<> u(0, n - 1);
    int ind = u(gen);
    queries->insert(queries->end(), dataset->begin() + ind * enc_dim,
                    dataset->begin() + (ind + 1) * enc_dim);

    for (int j = 0; j < enc_dim; ++j)
      (*dataset)[ind + j] = (*dataset)[n - (enc_dim - j)];
    for (int j = 0; j < enc_dim; ++j)
      dataset->pop_back();
    -- n;
  }
}

// custom hash can be a standalone function object:
struct MyHash {
  std::size_t operator()(vector<uint64_t> const &s) const noexcept {
    return XXH64(&s[0], s.size() * sizeof(uint64_t), H_SEED);
  }
};

vector<uint64_t> dedup(const vector<uint64_t> &dataset, int enc_dim) {
  unordered_set<vector<uint64_t>, MyHash> myset;
  vector<uint64_t> temp;
  auto n = dataset.size() / enc_dim;

  fprintf(stdout, "Before dedup: # of points: %d\n", n);
  for (unsigned i = 0; i < n; ++i) {
    temp.clear();
    temp.insert(temp.end(), dataset.begin() + enc_dim * i,
                dataset.begin() + (i + 1) * enc_dim);
    myset.insert(temp);
  }

  fprintf(stdout, "After dedup: # of points: %d\n", myset.size());
  vector<uint64_t> unique;

  for (const auto &point : myset) {
    unique.insert(unique.end(), point.begin(), point.end());
  }

  return unique;
}

void usage(const char *progname) {
  printf(
      "Usage: %s HAMMING-DIM [DATASET-DIRECTORY]\n\n\tThis program assumes the "
      "filenames are audio.data and audio.query respectively.\n\n",
      progname);
  exit(1);
}

int main(int argc, char **argv) {
  char *progname;
  char *p;
  progname = ((p = strrchr(argv[0], '/')) ? ++p : argv[0]);
  string dirname = FOLDER_NAME;

  int m = atoi(argv[1]);
  if (argc > 3) {
    usage(progname);
  }
  if (argc == 3) {
    dirname = string(argv[2]);
  }

  vector<Point> dataset;
  read_dataset(dirname + "/audio.data", &dataset);

  auto center = cal_center(dataset);

  recenter(dataset, center);

  unsigned dim = dataset.front().size();
  auto enc_dim = m / 64;

  SimHashCodes lsh(dim, m, C_SEED);

  auto hamming_dataset = lsh.fit(dataset);

  hamming_dataset = dedup(hamming_dataset, enc_dim);

  vector<uint64_t > queries;
  gen_queries(&hamming_dataset, &queries, enc_dim);

  string h5filename = string("audio-hamming-") + to_string(m) + ".h5";

  Hdf5File  h5f(h5filename);

  h5f.write<uint64_t>(hamming_dataset, "train");
   h5f.write<uint64_t>(queries, "test");

  return 0;
}