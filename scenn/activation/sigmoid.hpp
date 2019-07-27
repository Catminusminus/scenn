#ifndef SCENN_ACTIVATION_SIGMOID_HPP
#define SCENN_ACTIVATION_SIGMOID_HPP

#include <sprout/math.hpp>
#include <scenn/matrix/matrix.hpp>

namespace scenn {
template <class T>
constexpr auto sigmoid_double(T x) {
  float sigmoid_range = 34.538776394910684;

  if (x <= -sigmoid_range)
    return 1e-15;
  if (x >= sigmoid_range)
    return 1.0 - 1e-15;
  return 1.0 / (1.0 + sprout::math::exp(-x));
}

template <class T>
constexpr auto sigmoid_prime_double(T x) {
  return sigmoid_double(x) * (1 - sigmoid_double(x));
}

struct Sigmoid {
  template <size_t M, size_t N, class T>
  constexpr auto activate(const Matrix<M, N, T>& container) const {
    return container.fmap(sigmoid_double<T>);
  }
  template <size_t M, size_t N, class T>
  constexpr auto activate_prime(const Matrix<M, N, T>& container) const {
    return container.fmap(sigmoid_prime_double<T>);
  }
};
}

#endif
