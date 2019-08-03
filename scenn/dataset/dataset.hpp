#ifndef SCENN_DATASET_DATASET_HPP
#define SCENN_DATASET_DATASET_HPP

#include <scenn/matrix.hpp>
#include <sprout/algorithm/copy.hpp>
#include <sprout/sub_array.hpp>
#include <sprout/utility.hpp>
#include <sprout/utility/pair/pair.hpp>
#include <utility>
#include <tuple>

namespace {
template <size_t M, size_t N, size_t O, class T, class U>
constexpr auto make_dataset(scenn::Matrix<M, N, T>&& x,
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
  constexpr Dataset(
      std::array<sprout::pair<scenn::Matrix<1, N, T>, scenn::Matrix<1, O, U>>,
                 M>&& data): data(data){};
  constexpr Dataset(
      std::array<const sprout::pair<scenn::Matrix<1, N, T>, scenn::Matrix<1, O, U>>,
                 M>& data): data(data){};
  constexpr Dataset(scenn::Matrix<M, N, T>&& x, scenn::Matrix<M, O, U>&& y)
      : data(make_dataset(std::move(x),std::move(y))){};
  constexpr Dataset(const scenn::Matrix<M, N, T>& x, const scenn::Matrix<M, O, U>& y)
      : data(make_dataset(x,y)){};
  constexpr auto get_data() const& { return data; }
  constexpr auto get_data() && { return std::move(data); }
  constexpr auto shuffle() const { return (*this); }
  //constexpr auto length() const { return M; }
  static constexpr auto length() {
    return M;
  }
  constexpr auto shape() const {
    return std::make_pair(M, N);
  }
  template <size_t I, size_t J>
  constexpr auto slice() const {
    // return Dataset(sprout::sub_array(data, i, j));
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
