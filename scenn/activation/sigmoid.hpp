#ifndef SCENN_ACTIVATION_SIGMOID_HPP
#define SCENN_ACTIVATION_SIGMOID_HPP

#include <scenn/matrix/matrix.hpp>
#include <sprout/math.hpp>

namespace scenn {
template <class T>
constexpr auto sigmoid(T x) {
  T sigmoid_range = 34.538776394910684;

  if (x <= -sigmoid_range) return static_cast<T>(1e-15);
  if (x >= sigmoid_range) return static_cast<T>(1.0 - 1e-15);
  return static_cast<T>(1.0 / (1.0 + sprout::math::exp(-x)));
}

template <class T>
constexpr auto sigmoid_prime(T x) {
  return sigmoid(x) * (1 - sigmoid(x));
}

struct Sigmoid {
  template <size_t M, size_t N, class T>
  constexpr auto activate(const Matrix<M, N, T>& container) const {
    return container.fmap([](auto&& x) { return sigmoid<T>(x); });
  }
  template <size_t M, size_t N, class T>
  constexpr auto activate_prime(const Matrix<M, N, T>& container) const {
    return container.fmap([](auto&& x) { return sigmoid_prime<T>(x); });
  }
};
}  // namespace scenn

#endif
