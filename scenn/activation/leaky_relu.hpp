#ifndef SCENN_ACTIVATION_LEAKY_RELU_HPP
#define SCENN_ACTIVATION_LEAKY_RELU_HPP

#include <scenn/matrix/matrix.hpp>
#include <scenn/util.hpp>

namespace scenn {
template <class T>
SCENN_CONSTEXPR auto leaky_relu(T x, T alpha) {
  if (x >= 0) return x;
  return alpha * x;
}

template <class T>
SCENN_CONSTEXPR auto leaky_relu_prime(T x, T alpha) {
  if (x > 0) return static_cast<T>(1);
  return alpha;
}

template <class T>
class LeakyReLU {
  T alpha;

 public:
  constexpr LeakyReLU(T alpha) : alpha(alpha){};
  template <size_t M, size_t N>
  SCENN_CONSTEXPR auto activate(const Matrix<M, N, T>& container) const {
    return container.fmap([=](auto&& x) { return leaky_relu<T>(x, alpha); });
  }
  template <size_t M, size_t N>
  SCENN_CONSTEXPR auto activate_prime(const Matrix<M, N, T>& container) const {
    return container.fmap(
        [=](auto&& x) { return leaky_relu_prime<T>(x, alpha); });
  }
  template <class V, class U>
  SCENN_CONSTEXPR auto calc_backward_pass(V&& data, U&& delta) const {
    return activate_prime(std::forward<V>(data)) * (std::forward<U>(delta));
  }
};
}  // namespace scenn

#endif
