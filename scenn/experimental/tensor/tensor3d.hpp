#ifndef SCENN_EXPERIMENTAL_TENSOR_TENSOR3D_HPP
#define SCENN_EXPERIMENTAL_TENSOR_TENSOR3D_HPP

#include <scenn/util.hpp>

namespace scenn::experimental {
template <std::size_t L, std::size_t M, std::size_t N, class T>
class Tensor3D;
template <std::size_t M, std::size_t N, class T>
using Matrix = Tensor3D<1, M, N, T>;
template <std::size_t N, class T>
using Vector = Matrix<1, N, T>;
template <std::size_t L, std::size_t M, std::size_t N, class T>
class Tensor3D {
  using value_type = T;
  using size_type = std::size_t;

  value_type data[L][M][N];
  SCENN_CONSTEXPR Tensor3D(): data{} {
    static_assert(L >= 1 && M >= 1 && N >= 1);
  }

  static SCENN_CONSTEXPR auto shape() { return std::make_pair(L, M, N); }

};
} //namespace scenn::experimental

#endif
