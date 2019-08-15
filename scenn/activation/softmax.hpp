#ifndef SCENN_ACTIVATION_SOFTMAX_HPP
#define SCENN_ACTIVATION_SOFTMAX_HPP

#include <scenn/matrix/matrix.hpp>
#include <scenn/util.hpp>
#include <sprout/math.hpp>

namespace scenn {
struct Softmax {
  template <size_t M, size_t N, class T>
  SCENN_CONSTEXPR auto activate(const Matrix<M, N, T>& container) const {
    auto mat = container.fmap([](auto&& x) { return sprout::math::exp(x - std::numeric_limits<decltype(x)>::max()); });
    auto s = mat.sum();
    return std::move(mat).fmap([](auto&& x) { return x / s; });
  }
  template <class T, class U>
  SCENN_CONSTEXPR auto calc_backward_pass(T&& data, U&& delta) const {
    auto y = activate(std::forward<T>(data)); 
    return y * (delta - (y * delta).to_value());
  }
};
}  // namespace scenn

#endif
