#ifndef __CREATE_LSH_CODES_H_
#define __CREATE_LSH_CODES_H_

#include <bitset>
#include <chrono>
#include <random>
#include <vector>
#include <iostream>
#include <cassert>

using MatrixXd = std::vector<float>;

const int INTERVAL = int(1e4);

class SimHashCodes
{
public:
  SimHashCodes(
      unsigned dim, unsigned m,
      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count())
      : d_(dim), m_(m), seed_(seed), mat_(m, dim)
  {
    assert(m % 64 == 0 && "Currently we only support for multiples of 64");
    genRandomVectors();
  }

  std::vector<uint64_t> fit(const std::vector<float> &X) const
  {

    std::bitset<64> bits;
    int ind = 0;
    unsigned enc_dim = m_ / 64;
    unsigned n = X.size() / d_;

    std::vector<uint64_t> Y(enc_dim * n);
    std::vector<float> raw_codes(m_);

    for (unsigned pid = 0;pid < n;++ pid)
    {
        const auto point = &X[pid * d_];
      for (unsigned rid = 0;rid < m_;++ rid) {
          raw_codes[rid] = 0;
          for (unsigned cid = 0;cid < d_;++ cid) {
              raw_codes[rid] += mat_[rid * d_ + cid] * point[cid];
          }
      }

      for (unsigned i = 0; i < enc_dim; ++i)
      {
        for (unsigned j = 0; j < 64; ++j)
        {
          if (raw_codes[i * 64 + j] >= 0)
            bits[j] = true;
          else
            bits[j] = false;
        }
        Y[enc_dim * ind + i] = bits.to_ullong();
      }

      ++ind;
    }

    return Y;
  }

private:
  void genRandomVectors()
  {
    std::mt19937_64 gen(seed_);
    std::normal_distribution<double> dist_normal(0.0, 1.0);
    for (size_t i = 0; i < m_; ++i)
      for (size_t j = 0; j < d_; ++j)
        mat_[i * d_ + j] = dist_normal(gen);
  }

  unsigned d_;     // data dimension
  unsigned m_;     // # of hashes (# of dimension for Hamming)
  unsigned seed_; // random seed
  MatrixXd mat_; // random matrix
};

#endif // __CREATE_LSH_CODES_H_
