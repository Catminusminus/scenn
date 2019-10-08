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

  SCENN_CONSTEXPR auto operator()(size_type i, size_type j, size_type k) const {
    return data[i][j][k];
  }
  SCENN_CONSTEXPR decltype(auto) operator()(size_type i, size_type j, size_type k) {
    return data[i][j][k];
  }

  // for matrix-like tensor
  SCENN_CONSTEXPR auto operator()(size_type i, size_type j) const { static_assert(M == 1); return data[0][i][j]; }
  SCENN_CONSTEXPR decltype(auto) operator()(size_type i, size_type j) { static_assert(M == 1); return data[0][i][j]; }
  SCENN_CONSTEXPR auto operator[](size_type i, size_type j) const { static_assert(M == 1); return (*this)(i, j); }
  SCENN_CONSTEXPR decltype(auto) operator[](size_type i, size_type j) { static_assert(M == 1); return (*this)(i, j); }

  // for vector-like tensor
  SCENN_CONSTEXPR auto operator()(size_type i) const { static_assert(L == 1 && M == 1); return data[0][0][i]; }
  SCENN_CONSTEXPR decltype(auto) operator()(size_type i) { static_assert(L == 1 && M == 1); return data[0][0][i]; }
  SCENN_CONSTEXPR auto operator[](size_type i) const { static_assert(L == 1 && M == 1); return (*this)(i); }
  SCENN_CONSTEXPR decltype(auto) operator[](size_type i) { static_assert(L == 1 && M == 1); return (*this)(i); }
};
} //namespace scenn::experimental

#endif
