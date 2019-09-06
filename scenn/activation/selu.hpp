#ifndef SCENN_ACTIVATION_SELU_HPP
#define SCENN_ACTIVATION_SELU_HPP

#include <scenn/matrix/matrix.hpp>
#include <scenn/util.hpp>
#include <sprout/math.hpp>

namespace scenn {
template <class T>
SCENN_CONSTEXPR auto selu(T x, T alpha, T scale) {
  if (x >= 0) return scale * x;
  return scale * alpha * (sprout::math::exp(x) - 1);
}

template <class T>
SCENN_CONSTEXPR auto selu_prime(T x, T alpha, T scale) {
  if (x > 0) return scale;
  return scale * alpha * sprout::math::exp(x);
}

template <class T>
class SeLU {
  // ref https://www.tensorflow.org/api_docs/python/tf/keras/activations/selu
  T alpha = 1.67326324;
  T scale = 1.05070098;

 public:
  template <size_t M, size_t N>
  SCENN_CONSTEXPR auto activate(const Matrix<M, N, T>& container) const {
    return container.fmap([=](auto&& x) { return selu<T>(x, alpha, scale); });
  }
  template <size_t M, size_t N>
  SCENN_CONSTEXPR auto activate_prime(const Matrix<M, N, T>& container) const {
    return container.fmap(
        [=](auto&& x) { return selu_prime<T>(x, alpha, scale); });
  }
  template <class V, class U>
  SCENN_CONSTEXPR auto calc_backward_pass(V&& data, U&& delta) const {
    return activate_prime(std::forward<V>(data)) * (std::forward<U>(delta));
  }
};
}  // namespace scenn

#endif
