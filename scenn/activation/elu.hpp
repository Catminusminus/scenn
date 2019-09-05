#ifndef SCENN_ACTIVATION_ELU_HPP
#define SCENN_ACTIVATION_ELU_HPP

#include <scenn/matrix/matrix.hpp>
#include <scenn/util.hpp>
#include <sprout/math.hpp>

namespace scenn {
template <class T>
SCENN_CONSTEXPR auto elu(T x, T alpha) {
  if (x >= 0) return x;
  return alpha * (sprout::math::exp(x) - 1);
}

template <class T>
SCENN_CONSTEXPR auto elu_prime(T x, T alpha) {
  if (x > 0) return static_cast<T>(1);
  return alpha * sprout::math::exp(x);
}

template <class T>
class ELU {
  T alpha;

 public:
  constexpr ELU(T alpha) : alpha(alpha){};
  template <size_t M, size_t N>
  SCENN_CONSTEXPR auto activate(const Matrix<M, N, T>& container) const {
    return container.fmap([=](auto&& x) { return elu<T>(x, alpha); });
  }
  template <size_t M, size_t N>
  SCENN_CONSTEXPR auto activate_prime(const Matrix<M, N, T>& container) const {
    return container.fmap(
        [=](auto&& x) { return elu_prime<T>(x, alpha); });
  }
  template <class V, class U>
  SCENN_CONSTEXPR auto calc_backward_pass(V&& data, U&& delta) const {
    return activate_prime(std::forward<V>(data)) * (std::forward<U>(delta));
  }
};
}  // namespace scenn

#endif
