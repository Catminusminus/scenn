#ifndef SCENN_DATASET_DATASET_HPP
#define SCENN_DATASET_DATASET_HPP

#include <scenn/matrix.hpp>
#include <scenn/util.hpp>
#include <sprout/algorithm.hpp>
#include <sprout/random.hpp>
#include <sprout/random/unique_seed.hpp>
#include <sprout/sub_array.hpp>
#include <sprout/utility.hpp>
#include <sprout/utility/pair/pair.hpp>
#include <tuple>
#include <utility>

namespace {
template <size_t M, size_t N, size_t O, class T, class U>
SCENN_CONSTEXPR auto make_dataset(scenn::Matrix<M, N, T>&& x,
                                  scenn::Matrix<M, O, U>&& y) {
  std::array<sprout::pair<scenn::Matrix<1, N, T>, scenn::Matrix<1, O, U>>, M>
      data;
  for (std::size_t i = 0; i < M; ++i) {
    data[i] = sprout::make_pair(x.get_nth_row(i), y.get_nth_row(i));
  }
  return data;
}

template <size_t M, size_t N, size_t O, class T, class U>
struct Dataset {
  std::array<sprout::pair<scenn::Matrix<1, N, T>, scenn::Matrix<1, O, U>>, M>
      data;
  SCENN_CONSTEXPR Dataset(
      std::array<sprout::pair<scenn::Matrix<1, N, T>, scenn::Matrix<1, O, U>>,
                 M>&& data)
      : data(data){};
  SCENN_CONSTEXPR Dataset(std::array<const sprout::pair<scenn::Matrix<1, N, T>,
                                                        scenn::Matrix<1, O, U>>,
                                     M>& data)
      : data(data){};
  SCENN_CONSTEXPR Dataset(scenn::Matrix<M, N, T>&& x,
                          scenn::Matrix<M, O, U>&& y)
      : data(make_dataset(std::move(x), std::move(y))){};
  SCENN_CONSTEXPR Dataset(const scenn::Matrix<M, N, T>& x,
                          const scenn::Matrix<M, O, U>& y)
      : data(make_dataset(x, y)){};
  SCENN_CONSTEXPR auto get_data() const& { return data; }
  SCENN_CONSTEXPR auto get_data() && { return std::move(data); }
  SCENN_CONSTEXPR auto shuffle(std::size_t seed) const {
    auto shuffled_data = data;
    auto engine = sprout::minstd_rand0(SPROUT_UNIQUE_SEED + seed);
    sprout::shuffle(shuffled_data.begin(), shuffled_data.end(), engine);
    return Dataset(std::move(shuffled_data));
  }
  static constexpr auto length() { return M; }
  SCENN_CONSTEXPR auto shape() const { return std::make_pair(M, N); }
  template <size_t I, size_t J>
  SCENN_CONSTEXPR auto slice() const {
    return Dataset<J - I, N, O, T, U>(
        sprout::copy<std::array<sprout::pair<scenn::Matrix<1, N, T>,
                                             scenn::Matrix<1, O, U>>,
                                J - I>,
                     decltype(data.cbegin())>(data.cbegin() + I,
                                              data.cbegin() + J));
  }
};
}  // namespace

#endif
