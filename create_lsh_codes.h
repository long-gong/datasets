#ifndef __CREATE_LSH_CODES_H_
#define __CREATE_LSH_CODES_H_
#include <eigen3/Eigen/Dense>
#include <bitset>
#include <chrono>
#include <random>
#include <vector>
#include <iostream>

using namespace Eigen;

const int INTERVAL = int(1e4);

class SimHashCodes {
public:
  SimHashCodes(
      unsigned dim, unsigned m,
      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count())
      : d_(dim), m_(m), seed_(seed), mat_(m, dim) {
    assert(m % 64 == 0 && "Currently we only support for multiples of 64");
    genRandomVectors();
  }

  std::vector<uint64_t> fit(const std::vector<VectorXd> &X) const {
    std::vector<uint64_t> Y;
    std::bitset<64> bits;

    int ind = 0;
    for (const auto& point : X) {
      auto raw_codes = mat_ * point;
      for (unsigned i = 0, j = 0; i < raw_codes.size(); ++i, ++j) {
        if (raw_codes(i) >= 0) {
          bits[j] = true;
        }
        if (j == 63) {
          j = 0;
          Y.push_back(bits.to_ullong());
          bits.reset();
        }
      }
      ++ ind;
      if (ind % INTERVAL == 0) {
        std::cout << ind << " out of " << X.size() << " finished ..." << std::endl;
      }
    }

    return Y;
  }

private:
  void genRandomVectors() {
    std::mt19937_64 gen(seed_);
    std::normal_distribution<double> dist_normal(0.0, 1.0);
    for (size_t i = 0; i < mat_.rows(); ++i)
      for (size_t j = 0; j < mat_.cols(); ++j)
        mat_(i, j) = dist_normal(gen);
  }
  unsigned d_;
  unsigned m_;
  unsigned seed_;
  MatrixXd mat_;
};

#endif // __CREATE_LSH_CODES_H_
