#include "Hdf5File.h"
#include "create_lsh_codes.h"
#include "helper.h"
#include <cassert>
#include <eigen3/Eigen/Dense>
#include <random>
#include <unordered_set>
#include <vector>
#include <xxhash.h>

using namespace std;
using namespace Eigen;

// if disk is enough, it would be better to copy bigann_base.bvecs to
// `datasets/GIST80M`
#ifdef CLION_DEBUG
const char *DATASET_DIR = "../datasets/GIST80M";

const int NUM_QUERIES = 100;
const int SEED = 4057218;
const unsigned C_SEED = 91023221u;
const int DIM = 384;
// please decrease this value if your PC has a small amount of DRAM
const int N_EACH = int(1e3);
// please increase this value if your PC has a small amount of DRAM
const int N_FILES = 8;
const uint64_t N_FILES_MASK = 0x7;

// const int N_EACH = int(1e7);
//// please increase this value if your PC has a small amount of DRAM
// const int N_FILES = 1;
// const uint64_t N_FILES_MASK = 0x0;

XXH64_hash_t const H_SEED = 0; /* or any other value */
#else
const char *DATASET_DIR =
    "/media/gtuser/LGLarge/ann-datasets/Euclidean/tiny"; //"datasets/GIST80M";

const int NUM_QUERIES = 10000;
const int SEED = 4057218;
const unsigned C_SEED = 91023221u;
const int DIM = 384;
// please decrease this value if your PC has a small amount of DRAM
const int N_EACH = int(1e6);
// please increase this value if your PC has a small amount of DRAM
const int N_FILES = 128;
const uint64_t N_FILES_MASK = 0x7f;
XXH64_hash_t const H_SEED = 0; /* or any other value */
#endif

const char *getCenterCacheFileName()
{
  static const char *CCF = "GIST80M-CENTER-float.dat";
  return CCF;
}

const char *getCentersCacheFileName()
{
  static const std::string CsCF =
      std::string("GIST80M-CENTERS-float-") + std::to_string(N_EACH) + ".dat";
  return CsCF.c_str();
}
/** THE FOLLOWING SEVERAL FUNCTIONS WERE SHAMELESSLY COPIED
 *   FROM https://github.com/FALCONN-LIB/FALCONN **/

using Point = VectorXd;

/*
 * An auxiliary function that reads a point from a binary file that is produced
 * by a script 'prepare-dataset.sh'
 */
bool read_point(FILE *file, Point *point)
{
  //  int d;
  //  if (fread(&d, sizeof(int), 1, file) != 1) {
  //    return false;
  //  }
  //  assert(d == DIM || DIM < 0);
  auto d = DIM;
  auto *buf = new float[d];
  if (fread(buf, sizeof(float), d, file) != (size_t)d)
  {
    throw runtime_error("can't read a point");
  }
  point->resize(d);
  for (int i = 0; i < d; ++i)
  {
    (*point)[i] = buf[i];
  }
  delete[] buf;
  return true;
}

/*
 * An auxiliary function that reads a dataset from a binary file that is
 * produced by a script 'prepare-dataset.sh'
 */
void read_dataset(string file_name, vector<Point> *dataset, int dim, int start,
                  int size)
{
  FILE *file = fopen(file_name.c_str(), "rb");
  if (!file)
  {
    throw runtime_error("can't open the file with the dataset");
  }

  int vecsizeof = sizeof(float) * dim;
  // here overflow happens (so must convert to long int)
  fseek(file, (long int)start * vecsizeof, SEEK_SET);
  Point p;
  dataset->clear();
  while (dataset->size() < size && read_point(file, &p))
  {
    dataset->push_back(p);
  }
  if (fclose(file))
  {
    throw runtime_error("fclose() error");
  }
}

void read_dataset(string file_name, vector<Point> *dataset)
{
  FILE *file = fopen(file_name.c_str(), "rb");
  if (!file)
  {
    throw runtime_error("can't open the file with the dataset");
  }

  Point p;
  dataset->clear();
  while (read_point(file, &p))
  {
    dataset->push_back(p);
  }
  if (fclose(file))
  {
    throw runtime_error("fclose() error");
  }
}

bool read_center(FILE *file, Point *center)
{
  int d;
  if (fread(&d, sizeof(int), 1, file) != 1)
  {
    return false;
  }
  assert(d == DIM);
  auto *buf = new float[d];
  if (fread(buf, sizeof(float), d, file) != (size_t)d)
  {
    throw runtime_error("can't read a point");
  }
  center->resize(d);
  for (int i = 0; i < d; ++i)
  {
    (*center)[i] = buf[i];
  }
  delete[] buf;
  return true;
}

void read_centers(string file_name, vector<Point> *centers, int *tn)
{
  FILE *file = fopen(file_name.c_str(), "rb");
  if (!file)
  {
    throw runtime_error("can't open the file with the dataset");
  }

  if (fread(tn, sizeof(int), 1, file) != 1)
  {
    throw runtime_error("fread() error");
  }
  Point p;
  centers->clear();
  while (read_center(file, &p))
  {
    centers->push_back(p);
  }
  if (fclose(file))
  {
    throw runtime_error("fclose() error");
  }
}

Point cal_center(const vector<Point> &dataset)
{
  // find the center of mass
  Point center = dataset[0];
  for (size_t i = 1; i < dataset.size(); ++i)
  {
    center += dataset[i];
  }
  center /= dataset.size();

  return center;
}

Point cal_sum(const vector<Point> &dataset)
{
  Point sum = dataset[0];
  for (size_t i = 1; i < dataset.size(); ++i)
  {
    sum += dataset[i];
  }
  return sum;
}

void recenter(vector<Point> &dataset, const Point &center)
{
  for (size_t i = 0; i < dataset.size(); ++i)
  {
    dataset[i] -= center;
  }
}

size_t get_dataset_size(FILE *fp)
{
  int cur_pos = ftell(fp);
  rewind(fp);
  size_t vecsizeof = sizeof(float) * DIM;
  fseek(fp, 0, SEEK_END);
  size_t n = ftell(fp) / vecsizeof;
  fseek(fp, cur_pos, SEEK_SET);
  return n;
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
  auto base_filename = dirname + "/tinygist80million.bin";

  FILE *fp = fopen(base_filename.c_str(), "rb");
  if (!fp)
  {
    perror("fread() failed");
  }
  auto N = get_dataset_size(fp);
  printf("Get #points: %lu, and #dim: %d\n", N, DIM);
  fclose(fp);
  auto ng = int(ceil(N * 1.0 / N_EACH));

  // load dataset and calculate center

  vector<Point> centers;

  int tn = 0;
  printf("Calculating center ...\n");

  auto cscf = getCentersCacheFileName();
  auto cscfp = fopen(cscf, "rb");
  if (cscfp != NULL)
  {
    read_centers(cscf, &centers, &tn);
    fclose(cscfp);
  }
  else
  {
    for (auto i = 0; i < ng; ++i)
    {
      vector<Point> dataset;
      read_dataset(base_filename, &dataset, DIM, N_EACH * i, N_EACH);
#ifdef DEBUG
      if (i == 0)
        tofile<Point>(dataset, "gist80m-debugging-ds.txt", 10);
#endif
      tn += dataset.size();
      centers.push_back(cal_sum(dataset));
      printf("CC: %d out of %d groups were done ...\n", i + 1, ng);
    }

    assert(tn == N);

    {
      FILE *cfp = fopen(getCentersCacheFileName(), "wb");
      if (!cfp)
        perror("fopen() failed");

      if (fwrite(&tn, sizeof(tn), 1, cfp) != 1)
        perror("fwrite() failed");
      for (const auto &cen : centers)
      {
        int cen_dim = cen.size();
        if (fwrite(&cen_dim, sizeof(int), 1, cfp) != 1)
          perror("fwrite() failed");
        for (int j = 0; j < cen_dim; ++j)
        {
          float temp_cj = cen(j);
          if (fwrite(&temp_cj, sizeof(float), 1, cfp) != 1)
            perror("fwrite() failed");
        }
      }
      fclose(cfp);
    }
  }

  assert(tn > 0);
  Point center = cal_sum(centers);
  center /= tn;
  printf("Done\n");

  FILE *cfp = fopen(getCenterCacheFileName(), "wb");

  fwrite(&DIM, sizeof(DIM), 1, cfp);
  for (int i = 0; i < center.size(); ++i)
  {
    float temp = center(i);
    fwrite(&temp, sizeof(temp), 1, cfp);
  }
  fclose(cfp);
  tofile<Point>({center}, "GIST80M-center.txt", 1);

  printf("Calculating LSH codes ...\n");
  auto enc_dim = m / 64;

  SimHashCodes lsh(DIM, m, C_SEED);

  vector<FILE *> temp_ofiles(N_FILES);

  for (int k = 0; k < N_FILES; ++k)
  {
    auto fn = string("temp_gist80m/") + to_string(k) + ".dat";
    temp_ofiles[k] = fopen(fn.c_str(), "wb+");
    if (!temp_ofiles[k])
    {
      perror("fopen() failed");
    }
  }

  for (auto i = 0; i < ng; ++i)
  {
    vector<Point> dataset;
    vector<vector<uint64_t>> points_eachfile(N_FILES);
    read_dataset(base_filename, &dataset, DIM, N_EACH * i, N_EACH);
    recenter(dataset, center);
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
    printf("Encoding: %d out of %d groups were done ...\n", i + 1, ng);
  }
  printf("Done\n");

  vector<size_t> size_each(N_FILES, 0);

  string bfilename = string("gist80m-hamming-all-") + to_string(m) + ".dat";

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

  string tfilename = string("gist80m-hamming-train-") + to_string(m) + ".dat";
  string qfilename = string("gist80m-hamming-test-") + to_string(m) + ".dat";
  string h5filename = string("gist80m-hamming-") + to_string(m) + ".h5";

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