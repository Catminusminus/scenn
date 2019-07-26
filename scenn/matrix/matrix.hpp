#ifndef SCENN_MATRIX_MATRIX_HPP
#define SCENN_MATRIX_MATRIX_HPP

#include <array>
#include <sprout/math.hpp>

// This code is based on
// https://github.com/ushitora-anqou/constexpr-nn/blob/master/main.cpp

namespace scenn {
template <class T, class U>
constexpr auto is_same_value(T lhs, U rhs) {
  if (sprout::math::abs(lhs - rhs) > 2 * std::numeric_limits<T>::epsilon())
    return false;
  return true;
}

template <size_t M, size_t N, class T>
struct Matrix;
template <size_t N, class T>
using Vector = Matrix<1, N, T>;
template <size_t M, size_t N, class T>
struct Matrix {
  static constexpr int m = M;
  static constexpr int n = N;

  using value_type = T;

  T data[M][N];

  constexpr Matrix() : data{} { static_assert(N >= 1 && M >= 1); }

  constexpr auto shuffle() const { return (*this); }

  template <size_t I, size_t J>
  constexpr auto slice() const {
    static_assert(0 <= I && I < N && 0 <= J && J < M);
    Matrix<I, J, T> ret;
    for (size_t i = I; i < J; ++i)
      for (size_t j = 0; j < N; ++j) ret.data[i][j] = data[i][j];
    return ret;
  }

  constexpr auto transposed() const {
    Matrix<N, M, T> ret;
    for (size_t i = 0; i < M; ++i)
      for (size_t j = 0; j < N; ++j) ret.data[j][i] = data[i][j];
    return ret;
  }

  static constexpr auto shape() { return std::make_pair(M, N); }

  constexpr auto operator()(size_t i, size_t j) const { return data[i][j]; }
  constexpr decltype(auto) operator()(size_t i, size_t j) { return data[i][j]; }

  // for vector-like matrix
  constexpr auto operator()(size_t i) const { return data[0][i]; }
  constexpr decltype(auto) operator()(size_t i) { return data[0][i]; }
  constexpr auto operator[](size_t i) const { return (*this)(i); }
  constexpr decltype(auto) operator[](size_t i) { return (*this)(i); }

  // for 1-dim vector
  constexpr auto to_value() const& {
    static_assert(M == 1 && N == 1);
    return (*this)(0);
  }

  constexpr auto to_value() && {
    static_assert(M == 1 && N == 1);
    return std::move(*this)(0);
  }

  template <size_t rM>
  constexpr auto operator+(const Matrix<rM, N, T>& rhs) const {
    static_assert(rM == 1 || rM == M);
    if (rM == 1) {
      // broadcast
      Matrix<M, N, T> ret;
      for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j) ret(i, j) = (*this)(i, j) + rhs[j];
      return ret;
    } else {
      Matrix<M, N, T> ret;
      for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j) ret(i, j) = (*this)(i, j) + rhs(i, j);
      return ret;
    }
  }

  // broadcast
  constexpr auto operator+(T rhs) const {
    Matrix<M, N, T> ret;
    for (size_t i = 0; i < M; ++i)
      for (size_t j = 0; j < N; ++j) ret(i, j) = (*this)(i, j) + rhs;
    return ret;
  }

  template <class R>
  constexpr auto operator-(const R& rhs) const& {
    return *this + rhs * -1;
  }

  template <class R>
  constexpr auto operator-(const R& rhs) && {
    return std::move(*this) + rhs * -1;
  }

  template <class U>
  constexpr auto operator*(U scalar) const {
    Matrix<M, N, T> ret;
    for (size_t i = 0; i < M; ++i)
      for (size_t j = 0; j < N; ++j) ret(i, j) = (*this)(i, j) * scalar;
    return ret;
  }

  template <class U>
  constexpr auto operator/(U scalar) const& {
    return *this * (1. / scalar);
  }

  template <class U>
  constexpr auto operator/(U scalar) && {
    return std::move(*this) * (1. / scalar);
  }

  template <size_t L>
  constexpr auto dot(const Matrix<N, L, T>& rhs) const {
    Matrix<M, L, T> ret;
    for (size_t i = 0; i < M; ++i)
      for (size_t j = 0; j < N; ++j)
        for (size_t k = 0; k < L; ++k) ret(i, k) += (*this)(i, j) * rhs(j, k);
    return ret;
  }

  constexpr auto operator*(const Matrix<M, N, T>& rhs) const {
    Matrix<M, N, T> ret;
    for (size_t i = 0; i < M; ++i)
      for (size_t j = 0; j < N; ++j) ret(i, j) = (*this)(i, j) * rhs(i, j);
    return ret;
  }

  constexpr auto operator/(const Matrix<M, N, T>& rhs) const {
    Matrix<M, N, T> ret;
    for (size_t i = 0; i < M; ++i)
      for (size_t j = 0; j < N; ++j) ret(i, j) = (*this)(i, j) / rhs(i, j);
    return ret;
  }

  template <class Function>
  constexpr auto fmap(Function&& f) const {
    Matrix<M, N, T> ret;
    for (size_t i = 0; i < M; ++i)
      for (size_t j = 0; j < N; ++j) ret(i, j) = f((*this)(i, j));
    return ret;
  }

  constexpr auto argmax() const {
    std::size_t index = 0;
    for (size_t i = 1; i < N; ++i) {
      if ((*this)(index) < (*this)(i)) index = i;
    }
    return index;
  }
};

template <size_t lM, size_t lN, class lT, size_t rM, size_t rN, class rT>
constexpr auto operator==(const Matrix<lM, lN, lT>& lhs,
                          const Matrix<rM, rN, rT>& rhs) {
  if (!std::is_same_v<lT, rT>) return false;
  if (lN != rN || lM != rM) return false;

  for (size_t i = 0; i < lM; ++i)
    for (size_t j = 0; j < lN; ++j)
      if (!is_same_value(lhs.data[i][j], rhs.data[i][j])) return false;
  return true;
}

template <size_t lM, size_t lN, class lT, size_t rM, size_t rN, class rT>
constexpr auto operator!=(const Matrix<lM, lN, lT>& lhs,
                          const Matrix<rM, rN, rT>& rhs) {
  return !(lhs == rhs);
}

template <size_t M, size_t N, class T>
constexpr auto make_matrix_from_array(const T (&array)[M][N]) {
  Matrix<M, N, T> ret;
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      ret(i, j) = array[i][j];
    }
  }
  return ret;
}

template <size_t N, class T>
constexpr auto make_vector_from_array(const T (&array)[N]) {
  Vector<N, T> ret;
  for (size_t j = 0; j < N; ++j) {
    ret(j) = array[j];
  }
  return ret;
}

template <size_t M, size_t N, class T>
constexpr auto make_zeros_from_pair() {
  T ret[M][N] = {{0}};
  return make_matrix_from_array(ret);
}
}  // namespace scenn
#endif
