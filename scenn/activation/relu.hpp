#ifndef SCENN_ACTIVATION_RELU_HPP
#define SCENN_ACTIVATION_RELU_HPP

#include <scenn/matrix/matrix.hpp>
#include <scenn/util.hpp>

namespace scenn {
template <class T>
SCENN_CONSTEXPR auto relu(T x) {
  assert_arithmetic<T>();
  if (x >= 0) return x;
  return static_cast<T>(0);
}

template <class T>
SCENN_CONSTEXPR auto relu_prime(T x) {
  assert_arithmetic<T>();
  if (x > 0) return static_cast<T>(1);
  return static_cast<T>(0);
}

struct ReLU {
  template <std::size_t M, std::size_t N, class T>
  SCENN_CONSTEXPR auto activate(const Matrix<M, N, T>& container) const {
    return container.fmap([](auto&& x) { return relu<T>(x); });
  }
  template <std::size_t M, std::size_t N, class T>
  SCENN_CONSTEXPR auto activate_prime(const Matrix<M, N, T>& container) const {
    return container.fmap([](auto&& x) { return relu_prime<T>(x); });
  }
  template <class T, class U>
  SCENN_CONSTEXPR auto calc_backward_pass(T&& data, U&& delta) const {
    return activate_prime(std::forward<T>(data)) * (std::forward<U>(delta));
  }
};
}  // namespace scenn

#endif
