#include "BvecsReader.h"
#include "ExceptionUtils.h"
#include "Hdf5File.h"
#include "create_lsh_codes_std.h"
#include "helper.h"
#include <cassert>
#include <cstring>
#include <random>
#include <unordered_set>
#include <vector>
#include <xxhash.h>
#include "Timer.hpp"

using namespace std;

// if disk is enough, it would be better to copy bigann_base.bvecs to
// `datasets/SIFT1B`
#ifdef CLION_DEBUG
const char *DATASET_DIR = "../datasets/SIFT1B";

const int NUM_QUERIES = 100;
const int SEED = 4057218;
const unsigned C_SEED = 91023221u;
int DIM = -1;
// please decrease this value if your PC has a small amount of DRAM
// const int N_EACH = int(1e3);
//// please increase this value if your PC has a small amount of DRAM
// const int N_FILES = 8;
// const uint64_t N_FILES_MASK = 0x7;

const int N_EACH = int(1e7);
// please increase this value if your PC has a small amount of DRAM
const int N_FILES = 1;
const uint64_t N_FILES_MASK = 0x0;

XXH64_hash_t const H_SEED = 0; /* or any other value */
#else
const char *DATASET_DIR = "/media/gtuser/LGLarge/ann-datasets/Euclidean/inria/"
                          "ANN_SIFT1B/"; //"datasets/SIFT1B";

const int NUM_QUERIES = 10000;
const int SEED = 4057218;
const unsigned C_SEED = 91023221u;
int DIM = -1;
// please decrease this value if your PC has a small amount of DRAM
const int N_EACH = int(1e7);
// please increase this value if your PC has a small amount of DRAM
const int N_FILES = 128;
const uint64_t N_FILES_MASK = 0x7f;
XXH64_hash_t const H_SEED = 0; /* or any other value */
#endif

const char *getCenterCacheFileName()
{
  static const char *CCF = "SIFT1B-CENTER-float.dat";
  return CCF;
}

const char *getCentersCacheFileName()
{
  static const std::string CsCF =
      std::string("SIFT1B-CENTERS-float-") + std::to_string(N_EACH) + ".dat";
  return CsCF.c_str();
}

// read center point from file (saved in fvecs format)
bool read_center(FILE *file,                // BINARY file handler
                 std::vector<float> &center // center coordinates (output)
)
{
  int d;
  if (fread(&d, sizeof(int), 1, file) != 1)
  {
    return false;
  }
  assert(d == DIM);

  center.resize(d);
  return fread(&center[0], sizeof(float), d, file) == (size_t)d;
}

// read centers from file
void read_centers(const string &file_name,             // BINARY filename
                  std::vector<vector<float>> &centers, // centers (output)
                  int &tn                              // total number of points
)
{
  FILE *file = fopen(file_name.c_str(), "rb");
  if (!file)
  {
    throw runtime_error("can't open the file with the dataset");
  }

  if (fread(&tn, sizeof(int), 1, file) != 1)
  {
    throw runtime_error("fread() error");
  }

  vector<float> p;
  centers.clear();
  while (read_center(file, p))
  {
    centers.push_back(p);
  }

  if (fclose(file))
  {
    throw runtime_error("fclose() error");
  }
}

// calculate sum with points storing in flat vector
std::vector<float> cal_sum(const vector<float> &dataset, // data points
                           unsigned dim                  // dimension
)
{
  std::vector<float> sum(dim, 0.0f);
  size_t n = dataset.size() / dim;
  for (size_t i = 0; i < n; ++i)
  {
    for (unsigned j = 0; j < dim; ++j)
    {
      sum[j] += dataset[i * dim + j];
    }
  }
  return sum;
}

// calculate sum with points storing in vector of vectors
std::vector<float> cal_sum(const vector<vector<float>> &dataset)
{
  unsigned dim = dataset.front().size();
  std::vector<float> sum(dim, 0.0f);
  size_t n = dataset.size();
  for (size_t i = 0; i < n; ++i)
  {
    for (unsigned j = 0; j < dim; ++j)
    {
      sum[j] += dataset[i][j];
    }
  }
  return sum;
}

// recenter
void recenter(vector<float> &dataset,          // data points
              const std::vector<float> &center // center
)
{
  unsigned dim = center.size();
  size_t n = dataset.size() / dim;
  for (size_t i = 0; i < n; ++i)
  {
    for (unsigned j = 0; j < dim; ++j)
      dataset[i * dim + j] -= center[j];
  }
}

/*
 * Chooses a random subset of the dataset to be the queries. The queries are
 * taken out of the dataset.
 */
void gen_queries(vector<uint64_t> *dataset, vector<uint64_t> *queries,
                 int enc_dim)
{
  mt19937_64 gen(SEED);
  queries->clear();
  auto n = dataset->size() / enc_dim;

  for (int i = 0; i < NUM_QUERIES; ++i)
  {
    uniform_int_distribution<> u(0, n - 1);
    int ind = u(gen);
    queries->insert(queries->end(), dataset->begin() + ind * enc_dim,
                    dataset->begin() + (ind + 1) * enc_dim);

    for (int j = 0; j < enc_dim; ++j)
      (*dataset)[ind * enc_dim + j] = (*dataset)[(n - 1) * enc_dim + j];
    for (int j = 0; j < enc_dim; ++j)
      dataset->pop_back();
    --n;
  }
}

// custom hash can be a standalone function object:
struct MyHash
{
  std::size_t operator()(vector<uint64_t> const &s) const noexcept
  {
    return XXH64(&s[0], s.size() * sizeof(uint64_t), H_SEED);
  }
};

vector<uint64_t> dedup(const vector<uint64_t> &dataset, int enc_dim)
{
  unordered_set<vector<uint64_t>, MyHash> myset;
  vector<uint64_t> temp;
  auto n = dataset.size() / enc_dim;

  fprintf(stdout, "Before dedup: # of points: %lu\n", n);

  for (unsigned i = 0; i < n; ++i)
  {
    temp.clear();
    temp.insert(temp.end(), dataset.begin() + enc_dim * i,
                dataset.begin() + (i + 1) * enc_dim);
    myset.insert(temp);
  }

  fprintf(stdout, "After dedup: # of points: %lu\n", myset.size());

  vector<uint64_t> unique;

  for (const auto &point : myset)
  {
    unique.insert(unique.end(), point.begin(), point.end());
  }

  return unique;
}

void usage(const char *progname)
{
  printf("Usage: %s HAMMING-DIM [DATASET-DIR]\n\n", progname);
  exit(1);
}

int main(int argc, char **argv)
{
  char *progname;
  char *p;
  progname = ((p = strrchr(argv[0], '/')) ? ++p : argv[0]);
  string dirname = DATASET_DIR;

  if (argc > 3 || argc < 2)
  {
    usage(progname);
  }
  if (argc == 3)
  {
    dirname = string(argv[2]);
  }

  int m = atoi(argv[1]);
  auto base_filename = dirname + "/bigann_base.bvecs";
  auto query_filename = dirname + "/bigann_query.bvecs";

  FILE *fp = fopen(base_filename.c_str(), "rb");
  if (!fp)
  {
    perror("fread() failed");
  }

  BvecsReader breader(base_filename.c_str());
  auto N = breader.numPoints();
  DIM = breader.pointDimension();

  printf("Get #points: %lu, and #dim: %d\n", N, DIM);
  fclose(fp);
  auto ng = int(ceil(N * 1.0 / N_EACH));

  // load dataset and calculate center
  vector<vector<float>> centers;

  int tn = 0;
  printf("Calculating center ...\n");

  auto cscf = getCentersCacheFileName();
  auto cscfp = fopen(cscf, "rb");
  if (cscfp != NULL)
  {
    read_centers(cscf, centers, tn);
    fclose(cscfp);
  }
  else
  {
    for (auto i = 0; i < ng; ++i)
    {
      vector<float> dataset = breader.read<float>(N_EACH);
#ifdef DEBUG
      if (i == 0)
        tofile<Point>(dataset, "sift1b-debugging-ds.txt", 10);
#endif
      tn += dataset.size() / DIM;
      centers.push_back(cal_sum(dataset, DIM));
      printf("CC: %d out of %d groups were done ...\n", i + 1, ng);
    }

    assert(tn == N);

    {
      vector<float> queries = BvecsReader{query_filename.c_str()}.read<float>();
      tn += queries.size();
      centers.push_back(cal_sum(queries, DIM));
    }

    {
      FILE *cfp = fopen(getCentersCacheFileName(), "wb");
      DS_REQUIRED(cfp != NULL);
      DS_REQUIRED(fwrite(&tn, sizeof(tn), 1, cfp) == 1);
      for (const auto &cen : centers)
      {
        int cen_dim = cen.size();
        DS_REQUIRED(fwrite(&cen_dim, sizeof(int), 1, cfp) == 1);
        DS_REQUIRED(fwrite(&cen[0], sizeof(float), DIM, cfp) == (size_t)DIM);
      }
      fclose(cfp);
    }
  }

  assert(tn > 0);
  vector<float> center = cal_sum(centers);

  for (unsigned i = 0; i < DIM; ++i)
    center[i] /= tn;
  printf("Done\n");

  FILE *cfp = fopen(getCenterCacheFileName(), "wb");

  DS_REQUIRED(fwrite(&DIM, sizeof(DIM), 1, cfp) == 1);
  DS_REQUIRED(fwrite(&center[0], sizeof(float), DIM, cfp) == (size_t)DIM);

  fclose(cfp);
  tofile<float>(center, "sift1b-center.txt", DIM);

  HighResolutionTimer timer;
  printf("Calculating LSH codes ...\n");
  auto enc_dim = m / 64;

  SimHashCodes lsh(DIM, m, C_SEED);

  vector<FILE *> temp_ofiles(N_FILES);

  for (int k = 0; k < N_FILES; ++k)
  {
    auto fn = string("temp_sift1b/") + to_string(k) + ".dat";
    temp_ofiles[k] = fopen(fn.c_str(), "wb+");
    if (!temp_ofiles[k])
    {
      perror("fopen() failed");
    }
  }

  breader.rewind();
  for (auto i = 0; i < ng; ++i)
  {
    timer.restart();
    vector<vector<uint64_t>> points_eachfile(N_FILES);
    vector<float> dataset = breader.read<float>(N_EACH);
    recenter(dataset, center);
    auto hamming_dataset = lsh.fit(dataset);
    auto n_points = hamming_dataset.size() / enc_dim;
    assert(n_points == dataset.size());
    for (int j = 0; j < n_points; ++j)
    {
      auto fid = (hamming_dataset[j * enc_dim] &
                  N_FILES_MASK); // get the last few digits
      assert(fid < N_FILES && fid >= 0);
      points_eachfile[fid].insert(points_eachfile[fid].end(),
                                  hamming_dataset.begin() + j * enc_dim,
                                  hamming_dataset.begin() + (j + 1) * enc_dim);
    }
    for (int k = 0; k < N_FILES; ++k)
    {
      fwrite(&points_eachfile[k][0], sizeof(uint64_t),
             points_eachfile[k].size(), temp_ofiles[k]);
    }
    auto e = timer.elapsed();
    printf("\tGroup %d took %.2f seconds, estimated time for the remaining groups is: %.2f seconds", i + 1, e / 1e6, (ng - i - 1) * e / 1e6);
  }

  // process queries
  {
    vector<vector<uint64_t>> points_eachfile(N_FILES);
    vector<float> dataset = BvecsReader{query_filename.c_str()}.read<float>();

    auto hamming_dataset = lsh.fit(dataset);
    for (int j = 0; j < dataset.size(); ++j)
    {
      auto fid = (hamming_dataset[j * enc_dim] &
                  N_FILES_MASK); // get the last few digits
      assert(fid < N_FILES && fid >= 0);
      points_eachfile[fid].insert(points_eachfile[fid].end(),
                                  hamming_dataset.begin() + j * enc_dim,
                                  hamming_dataset.begin() + (j + 1) * enc_dim);
    }
    for (int k = 0; k < N_FILES; ++k)
    {
      fwrite(&points_eachfile[k][0], sizeof(uint64_t),
             points_eachfile[k].size(), temp_ofiles[k]);
    }
  }

  printf("Done\n");

  vector<size_t> size_each(N_FILES, 0);

  string bfilename = string("sift1b-hamming-all-") + to_string(m) + ".dat";

  FILE *bf = fopen(bfilename.c_str(), "wb+");

  size_t n_tot = 0;
  printf("Dedup ...\n");
  for (int k = 0; k < N_FILES; ++k)
  {

    auto sz = ftell(temp_ofiles[k]) / sizeof(uint64_t);

    rewind(temp_ofiles[k]);

    vector<uint64_t> dataset(sz);

    if (fread(&dataset[0], sizeof(uint64_t), sz, temp_ofiles[k]) != sz)
    {
      perror("fread() failed");
    }

    dataset = dedup(dataset, enc_dim);

    size_each[k] = dataset.size() / enc_dim;

    n_tot += size_each[k];

    fwrite(&dataset[0], sizeof(uint64_t), dataset.size(), bf);

    fclose(temp_ofiles[k]);
    printf("Dedup: %d out of %d files were done ...\n", k + 1, N_FILES);
  }

  unordered_set<size_t> queries_ind;

  uniform_int_distribution<> u(0, n_tot - 1);
  mt19937_64 gen(SEED);

  while (queries_ind.size() < NUM_QUERIES)
  {
    queries_ind.insert(u(gen));
  }

  string tfilename = string("sift1b-hamming-train-") + to_string(m) + ".dat";
  string qfilename = string("sift1b-hamming-test-") + to_string(m) + ".dat";
  string h5filename = string("sift1b-hamming-") + to_string(m) + ".h5";

  Hdf5File h5f(h5filename);
  vector<size_t> dims{(n_tot - NUM_QUERIES) * enc_dim};
  h5f.createDataSet<uint64_t>("train", dims);

  FILE *tfp = fopen(tfilename.c_str(), "wb");

  const size_t n_each = N_EACH;
  auto nng = size_t(ceil(n_tot * 1.0 / n_each));

  size_t cumsum_o = 0, cumsum = 0;
  rewind(bf);

  vector<uint64_t> queries;
  size_t tc = 0;
  for (size_t i = 0; i < nng; ++i)
  {

    vector<uint64_t> data(n_each * enc_dim, 0ull);
    auto tsz = fread(&data[0], sizeof(uint64_t), n_each * enc_dim, bf);

    assert(tsz == data.size() || (tsz < data.size() && i == nng - 1));
    if (tsz < data.size())
      data.resize(tsz);

    cumsum += tsz / enc_dim;

    vector<uint64_t> train;

    for (int j = 0; j < tsz / enc_dim; ++j)
    {
      if (queries_ind.count(j + cumsum_o) > 0)
      {
        queries.insert(queries.end(), data.begin() + j * enc_dim,
                       data.begin() + (j + 1) * enc_dim);
      }
      else
      {
        train.insert(train.end(), data.begin() + j * enc_dim,
                     data.begin() + (j + 1) * enc_dim);
      }
    }

    fwrite(&train[0], sizeof(uint64_t), train.size(), tfp);

    h5f.write<uint64_t>(train, tc, tc + train.size(), "train");

    tc += train.size();
    cumsum_o = cumsum;
  }

  fclose(tfp);
  fclose(bf);

  assert(queries.size() == NUM_QUERIES * enc_dim);

  FILE *qfp = fopen(qfilename.c_str(), "wb");

  fwrite(&queries[0], sizeof(uint64_t), queries.size(), qfp);

  fclose(qfp);

  h5f.write<uint64_t>(queries, "test");

  return 0;
}
