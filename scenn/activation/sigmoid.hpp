#ifndef SCENN_ACTIVATION_SIGMOID_HPP
#define SCENN_ACTIVATION_SIGMOID_HPP

#include <scenn/matrix/matrix.hpp>
#include <sprout/math.hpp>

namespace scenn {
template <class T>
constexpr auto sigmoid(T x) {
  T sigmoid_range = 34.538776394910684;

  if (x <= -sigmoid_range) return 1e-15;
  if (x >= sigmoid_range) return 1.0 - 1e-15;
  return 1.0 / (1.0 + sprout::math::exp(-x));
}

template <class T>
constexpr auto sigmoid_prime(T x) {
  return sigmoid_double(x) * (1 - sigmoid_double(x));
}

struct Sigmoid {
  template <size_t M, size_t N, class T>
  constexpr auto activate(const Matrix<M, N, T>& container) const {
    return container.fmap(sigmoid<T>);
  }
  template <size_t M, size_t N, class T>
  constexpr auto activate_prime(const Matrix<M, N, T>& container) const {
    return container.fmap(sigmoid_prime<T>);
  }
};
}  // namespace scenn

#endif
