#ifndef SCENN_ACTIVATION_THRESHOLDED_RELU_HPP
#define SCENN_ACTIVATION_THRESHOLDED_RELU_HPP

#include <scenn/matrix/matrix.hpp>
#include <scenn/util.hpp>

namespace scenn {
template <class T>
SCENN_CONSTEXPR auto thresholded_relu(T x, T theta) {
  assert_arithmetic<T>();
  if (x >= theta) return x;
  return static_cast<T>(0);
}

template <class T>
SCENN_CONSTEXPR auto thresholded_relu_prime(T x, T theta) {
  assert_arithmetic<T>();
  if (x > theta) return static_cast<T>(1);
  return static_cast<T>(0);
}

template <class T>
class ThresholdedReLU {
  T theta;

 public:
  constexpr ThresholdedReLU(T theta) : theta(theta){};
  template <std::size_t M, std::size_t N>
  SCENN_CONSTEXPR auto activate(const Matrix<M, N, T>& container) const {
    return container.fmap(
        [=](auto&& x) { return thresholded_relu<T>(x, theta); });
  }
  template <std::size_t M, std::size_t N>
  SCENN_CONSTEXPR auto activate_prime(const Matrix<M, N, T>& container) const {
    return container.fmap(
        [=](auto&& x) { return thresholded_relu_prime<T>(x, theta); });
  }
  template <class V, class U>
  SCENN_CONSTEXPR auto calc_backward_pass(V&& data, U&& delta) const {
    return activate_prime(std::forward<V>(data)) * (std::forward<U>(delta));
  }
};
}  // namespace scenn

#endif
