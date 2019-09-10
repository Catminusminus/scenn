#ifndef SCENN_LOSS_MSE_HPP
#define SCENN_LOSS_MSE_HPP

#include <scenn/util.hpp>
#include <sprout/math.hpp>

namespace scenn {
struct MAE {
  template <class T, class U>
  static SCENN_CONSTEXPR auto loss_function(const T& predictions,
                                            const U& labels) {
    auto diff = predictions - labels;
    return std::move(diff)
        .fmap([](auto&& x) {
          return sprout::math::abs(std::forward<decltype(x)>(x));
        })
        .sum();
  }
  template <class T, class U>
  static SCENN_CONSTEXPR auto loss_derivative(const T& predictions,
                                              const U& labels) {
    auto diff = predictions - labels;
    return std::move(diff).fmap([](auto&& x) { return x <= 0 ? 1 : -1; });
  }
};
}  // namespace scenn
#endif
