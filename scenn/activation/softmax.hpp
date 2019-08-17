#ifndef SCENN_ACTIVATION_SOFTMAX_HPP
#define SCENN_ACTIVATION_SOFTMAX_HPP

#include <scenn/matrix/matrix.hpp>
#include <scenn/util.hpp>
#include <sprout/math.hpp>

namespace scenn {
struct Softmax {
  template <size_t M, size_t N, class T>
  SCENN_CONSTEXPR auto activate(const Matrix<M, N, T>& container) const {
    auto max = container.max_value();
    auto mat = container.fmap([max](auto&& x) {
      return sprout::math::exp(x - max);
    });
    auto s = mat.sum();
    return std::move(mat).fmap([s](auto&& x) { return x / s; });
  }
  template <class T, class U>
  SCENN_CONSTEXPR auto calc_backward_pass(T&& data, U&& delta) const {
    auto y = activate(std::forward<T>(data));
    return y * (delta - (y * delta).sum());
  }
};
}  // namespace scenn

#endif
