#ifndef SCENN_ACTIVATION_SOFTMAX_HPP
#define SCENN_ACTIVATION_SOFTMAX_HPP

#include <scenn/matrix/matrix.hpp>
#include <scenn/util.hpp>
#include <sprout/math.hpp>

namespace scenn {
template <class T>
SCENN_CONSTEXPR auto sigmoid(T x) {
  // ref http://www.kamishima.net/mlmpyja/lr/sigmoid.html
  T sigmoid_range = 34.538776394910684;

  if (x <= -sigmoid_range) return static_cast<T>(1e-15);
  if (x >= sigmoid_range) return static_cast<T>(1.0 - 1e-15);
  return static_cast<T>(1.0 / (1.0 + sprout::math::exp(-x)));
}

template <class T>
SCENN_CONSTEXPR auto sigmoid_prime(T x) {
  return sigmoid(x) * (1 - sigmoid(x));
}

struct Sigmoid {
  template <size_t M, size_t N, class T>
  SCENN_CONSTEXPR auto activate(const Matrix<M, N, T>& container) const {
    auto mat = container.fmap([](auto&& x) { return sprout::math::exp(x - std::numeric_limits::max()); });
    auto s = mat.sum();
    return std::move(mat).fmap([](auto&& x) { return x / s});
  }
  template <size_t M, size_t N, class T>
  SCENN_CONSTEXPR auto activate_prime(const Matrix<M, N, T>& container) const {
    return container.fmap([](auto&& x) { return sigmoid_prime<T>(x); });
  }
};
}  // namespace scenn

#endif
