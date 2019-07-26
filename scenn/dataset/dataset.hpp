#ifndef SCENN_DATASET_DATASET_HPP
#define SCENN_DATASET_DATASET_HPP

#include <scenn/matrix/matrix.hpp>
#include <sprout/sub_array.hpp>

namespace {
template <size_t M, size_t N, size_t O, class T, class U>
constexpr auto make_dataset(scenn::Matrix<M, N, T>&& x, scenn::Matrix<M, O, U>&& y) {
  std::array<std::pair<scenn::Matrix<1, N, T>, scenn::Matrix<1, O, U>>, M> data;
  for (auto i = 0; i < M; ++i) {
    data[i] = std::make_pair(x[i], y[i]);
  }
  return data;
}

template <size_t M, size_t N, size_t O, class T, class U>
struct Dataset {
  std::array<std::pair<scenn::Matrix<1, N, T>, scenn::Matrix<1, O, U>>, M> data;
  constexpr Dataset(std::array<std::pair<scenn::Matrix<1, N, T>, scenn::Matrix<1, O, U>>, M>&& data) : data(
    std::forward<std::array<std::pair<scenn::Matrix<1, N, T>, scenn::Matrix<1, O, U>>, M>>(data)) {};
  constexpr Dataset(scenn::Matrix<M, N, T>&& x, scenn::Matrix<M, O, U>&& y) : data(make_dataset(
    std::forward<scenn::Matrix<M, N, T>>(x),std::forward<scenn::Matrix<M, O, U>>(y))) {};
  constexpr auto get_data() {
    return data;
  }
  constexpr auto shuffule() {
    return (*this);
  }
  template <size_t I, size_t J>
  constexpr auto slice(size_t i, size_t j) const
  {
    return Dataset(sprout::sub_array(data, i, j));
  }
};
}

#endif
