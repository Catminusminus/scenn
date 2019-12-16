#ifndef SCENN_EXPERIMENTAL_TENSOR_TENSOR3D_HPP
#define SCENN_EXPERIMENTAL_TENSOR_TENSOR3D_HPP

#include <scenn/util.hpp>
#include <sprout/math.hpp>
#include <tuple>
#include <utility>

namespace scenn::experimental {
template <class T, class U>
SCENN_CONSTEXPR auto is_same_value(T lhs, U rhs) {
  if (sprout::math::abs(lhs - rhs) > 2 * std::numeric_limits<T>::epsilon())
    return false;
  return true;
}

template <std::size_t L, std::size_t M, std::size_t N, class T>
struct Tensor3D;
template <std::size_t M, std::size_t N, class T>
using Matrix = Tensor3D<1, M, N, T>;
template <std::size_t N, class T>
using Vector = Matrix<1, N, T>;
template <std::size_t L, std::size_t M, std::size_t N, class T>
struct Tensor3D {
  using value_type = T;
  using size_type = std::size_t;

  value_type data[L][M][N];

  SCENN_CONSTEXPR Tensor3D() : data{} {
    static_assert(L >= 1 && M >= 1 && N >= 1);
  }
  static SCENN_CONSTEXPR auto shape() { return std::make_tuple(L, M, N); }

  SCENN_CONSTEXPR auto operator()(size_type i, size_type j, size_type k) const {
    return data[i][j][k];
  }
  SCENN_CONSTEXPR decltype(auto) operator()(size_type i, size_type j,
                                            size_type k) {
    return data[i][j][k];
  }

  // for matrix-like tensor
  SCENN_CONSTEXPR auto operator()(size_type i, size_type j) const {
    static_assert(L == 1);
    return data[0][i][j];
  }
  SCENN_CONSTEXPR decltype(auto) operator()(size_type i, size_type j) {
    static_assert(L == 1);
    return data[0][i][j];
  }

  // for vector-like tensor
  SCENN_CONSTEXPR auto operator()(size_type i) const {
    static_assert(L == 1 && M == 1);
    return data[0][0][i];
  }
  SCENN_CONSTEXPR decltype(auto) operator()(size_type i) {
    static_assert(L == 1 && M == 1);
    return data[0][0][i];
  }
  SCENN_CONSTEXPR auto operator[](size_type i) const {
    static_assert(L == 1 && M == 1);
    return (*this)(i);
  }
  SCENN_CONSTEXPR decltype(auto) operator[](size_type i) {
    static_assert(L == 1 && M == 1);
    return (*this)(i);
  }

  // for 1-dim vector
  SCENN_CONSTEXPR auto to_value() const& {
    static_assert(L == 1 && M == 1 && N == 1);
    return (*this)(0);
  }
  SCENN_CONSTEXPR auto to_value() && {
    static_assert(L == 1 && M == 1 && N == 1);
    return std::move(*this)(0);
  }

  template <std::size_t rL>
  SCENN_CONSTEXPR auto operator+(const Tensor3D<rL, M, N, T>& rhs) const {
    static_assert(rL == 1 || rL == L || L == 1);
    if constexpr (rL == 1) {
      // broadcast
      Tensor3D<L, M, N, T> ret;
      for (std::size_t i = 0; i < L; ++i)
        for (std::size_t j = 0; j < M; ++j)
          for (std::size_t k = 0; k < N; ++k)
            ret(i, j, k) = (*this)(i, j, k) + rhs(j, k);
      return ret;
    } else if constexpr (L == 1) {
      // broadcast
      Tensor3D<rL, M, N, T> ret;
      for (std::size_t i = 0; i < rL; ++i)
        for (std::size_t j = 0; j < M; ++j)
          for (std::size_t k = 0; k < N; ++k)
            ret(i, j, k) = (*this)(j, k) + rhs(i, j, k);
      return ret;
    }
    Tensor3D<L, M, N, T> ret;
    for (std::size_t i = 0; i < L; ++i)
      for (std::size_t j = 0; j < M; ++j)
        for (std::size_t k = 0; k < N; ++k)
          ret(i, j, k) = (*this)(i, j, k) + rhs(i, j, k);
    return ret;
  }
  template <std::size_t rL>
  SCENN_CONSTEXPR auto operator*(const Tensor3D<rL, M, N, T>& rhs) const {
    static_assert(rL == 1 || rL == L || L == 1);
    if constexpr (rL == 1) {
      // broadcast
      Tensor3D<L, M, N, T> ret;
      for (std::size_t i = 0; i < L; ++i)
        for (std::size_t j = 0; j < M; ++j)
          for (std::size_t k = 0; k < N; ++k)
            ret(i, j, k) = (*this)(i, j, k) * rhs(j, k);
      return ret;
    } else if constexpr (L == 1) {
      // broadcast
      Tensor3D<rL, M, N, T> ret;
      for (std::size_t i = 0; i < rL; ++i)
        for (std::size_t j = 0; j < M; ++j)
          for (std::size_t k = 0; k < N; ++k)
            ret(i, j, k) = (*this)(j, k) * rhs(i, j, k);
      return ret;
    }
    Tensor3D<L, M, N, T> ret;
    for (std::size_t i = 0; i < L; ++i)
      for (std::size_t j = 0; j < M; ++j)
        for (std::size_t k = 0; k < N; ++k)
          ret(i, j, k) = (*this)(i, j, k) * rhs(i, j, k);
    return ret;
  }
  template <std::size_t rL>
  SCENN_CONSTEXPR auto operator/(const Tensor3D<rL, M, N, T>& rhs) const {
    static_assert(rL == 1 || rL == L || L == 1);
    if constexpr (rL == 1) {
      // broadcast
      Tensor3D<L, M, N, T> ret;
      for (std::size_t i = 0; i < L; ++i)
        for (std::size_t j = 0; j < M; ++j)
          for (std::size_t k = 0; k < N; ++k)
            ret(i, j, k) = (*this)(i, j, k) / rhs(j, k);
      return ret;
    } else if constexpr (L == 1) {
      // broadcast
      Tensor3D<rL, M, N, T> ret;
      for (std::size_t i = 0; i < rL; ++i)
        for (std::size_t j = 0; j < M; ++j)
          for (std::size_t k = 0; k < N; ++k)
            ret(i, j, k) = (*this)(j, k) / rhs(i, j, k);
      return ret;
    }
    Tensor3D<L, M, N, T> ret;
    for (std::size_t i = 0; i < L; ++i)
      for (std::size_t j = 0; j < M; ++j)
        for (std::size_t k = 0; k < N; ++k)
          ret(i, j, k) = (*this)(i, j, k) / rhs(i, j, k);
    return ret;
  }

  // broadcast
  template <class R>
  SCENN_CONSTEXPR auto operator+(const R rhs) const {
    Tensor3D<L, M, N, T> ret;
    for (std::size_t i = 0; i < L; ++i)
      for (std::size_t j = 0; j < M; ++j)
        for (std::size_t k = 0; k < N; ++k)
          ret(i, j, k) = (*this)(i, j, k) + rhs;
    return ret;
  }
  template <class R>
  SCENN_CONSTEXPR auto operator*(const R rhs) const {
    Tensor3D<L, M, N, T> ret;
    for (std::size_t i = 0; i < L; ++i)
      for (std::size_t j = 0; j < M; ++j)
        for (std::size_t k = 0; k < N; ++k)
          ret(i, j, k) = (*this)(i, j, k) * rhs;
    return ret;
  }
  template <class R>
  SCENN_CONSTEXPR auto operator-(const R& rhs) const& {
    return *this + rhs * -1;
  }
  template <class R>
  SCENN_CONSTEXPR auto operator-(const R& rhs) && {
    return std::move(*this) + rhs * -1;
  }
  template <class R>
  SCENN_CONSTEXPR auto operator/(R scalar) const& {
    return *this * (1. / scalar);
  }
  template <class R>
  SCENN_CONSTEXPR auto operator/(R scalar) && {
    return std::move(*this) * (1. / scalar);
  }
  template <std::size_t I, std::size_t J, std::size_t Index>
  SCENN_CONSTEXPR auto slice() const {
    if constexpr (Index == 0) {
      static_assert(0 <= I && I < J && J <= L);
      Tensor3D<J - I, M, N, T> ret;
      for (std::size_t i = I; i < J; ++i)
        for (std::size_t j = 0; j < M; ++j)
          for (std::size_t k = 0; k < N; ++k) ret(i, j, k) = data[i][j][k];
      return ret;
    } else if constexpr (Index == 1) {
      static_assert(0 <= I && I < J && J <= M);
      Tensor3D<L, J - I, N, T> ret;
      for (std::size_t i = 0; i < L; ++i)
        for (std::size_t j = I; j < J; ++j)
          for (std::size_t k = 0; k < N; ++k) ret(i, j, k) = data[i][j][k];
      return ret;
    } else if constexpr (Index == 2) {
      static_assert(0 <= I && I < J && J <= N);
      Tensor3D<L, M, J - I, T> ret;
      for (std::size_t i = 0; i < L; ++i)
        for (std::size_t j = 0; j < M; ++j)
          for (std::size_t k = I; k < J; ++k) ret(i, j, k) = data[i][j][k];
      return ret;
    } else {
      static_assert([] { return false; }(), "Index must be 0, 1, or 2");
    }
  }
};
template <std::size_t lL, std::size_t lM, std::size_t lN, class lT,
          std::size_t rL, std::size_t rM, std::size_t rN, class rT>
SCENN_CONSTEXPR auto operator==(const Tensor3D<lL, lM, lN, lT>& lhs,
                                const Tensor3D<rL, rM, rN, rT>& rhs) {
  if (!std::is_same_v<lT, rT>) return false;
  if (lL != rL || lN != rN || lM != rM) return false;

  for (std::size_t i = 0; i < lL; ++i)
    for (std::size_t j = 0; j < lM; ++j)
      for (std::size_t k = 0; k < lN; ++k)
        if (!is_same_value(lhs.data[i][j][k], rhs.data[i][j][k])) return false;
  return true;
}

template <std::size_t lL, std::size_t lM, std::size_t lN, class lT,
          std::size_t rL, std::size_t rM, std::size_t rN, class rT>
SCENN_CONSTEXPR auto operator!=(const Tensor3D<lL, lM, lN, lT>& lhs,
                                const Tensor3D<rL, rM, rN, rT>& rhs) {
  return !(lhs == rhs);
}

template <std::size_t L, std::size_t M, std::size_t N, class T>
SCENN_CONSTEXPR void swap(Tensor3D<L, M, N, T>& lhs,
                          Tensor3D<L, M, N, T>& rhs) {
  auto tmp = std::move(lhs);
  lhs = std::move(rhs);
  rhs = std::move(tmp);
}

template <std::size_t L, std::size_t M, std::size_t N, class T>
SCENN_CONSTEXPR auto make_tensor3d_from_array(const T (&array)[L][M][N]) {
  Tensor3D<L, M, N, T> ret;
  for (std::size_t i = 0; i < L; ++i)
    for (std::size_t j = 0; j < M; ++j)
      for (std::size_t k = 0; k < N; ++k) ret(i, j, k) = array[i][j][k];
  return ret;
}

template <std::size_t M, std::size_t N, class T>
SCENN_CONSTEXPR auto make_matrix_from_array(const T (&array)[M][N]) {
  Matrix<M, N, T> ret;
  for (std::size_t j = 0; j < M; ++j)
    for (std::size_t k = 0; k < N; ++k) ret(j, k) = array[j][k];
  return ret;
}
}  // namespace scenn::experimental

#endif
