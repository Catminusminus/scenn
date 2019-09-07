#ifndef SCENN_ACTIVATION_HARD_SIGMOID_HPP
#define SCENN_ACTIVATION_HARD_SIGMOID_HPP

#include <scenn/matrix/matrix.hpp>
#include <scenn/util.hpp>

namespace scenn {
template <class T>
SCENN_CONSTEXPR auto hard_sigmoid(T x) {
  if (x < static_cast<T>(-2.5)) return static_cast<T>(0);
  if (x < static_cast<T>(2.5)) return static_cast<T>(0.2 * x + 0.5);
  return static_cast<T>(1);
}

template <class T>
SCENN_CONSTEXPR auto hard_sigmoid_prime(T x) {
  if (x < static_cast<T>(-2.5)) return static_cast<T>(0);
  if (x < static_cast<T>(2.5)) return static_cast<T>(0.2);
  return static_cast<T>(0);
}

struct HardSigmoid {
  template <size_t M, size_t N, class T>
  SCENN_CONSTEXPR auto activate(const Matrix<M, N, T>& container) const {
    return container.fmap([](auto&& x) { return hard_sigmoid<T>(x); });
  }
  template <size_t M, size_t N, class T>
  SCENN_CONSTEXPR auto activate_prime(const Matrix<M, N, T>& container) const {
    return container.fmap([](auto&& x) { return hard_sigmoid_prime<T>(x); });
  }
  template <class T, class U>
  SCENN_CONSTEXPR auto calc_backward_pass(T&& data, U&& delta) const {
    return activate_prime(std::forward<T>(data)) * (std::forward<U>(delta));
  }
};
}  // namespace scenn

#endif
