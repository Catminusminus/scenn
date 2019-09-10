#ifndef SCENN_LOSS_HINGE_HPP
#define SCENN_LOSS_HINGE_HPP

#include <scenn/util.hpp>

namespace scenn {
struct Hinge {
  template <class T, class U>
  static SCENN_CONSTEXPR auto loss_function(const T& predictions,
                                            const U& labels) {
    return ((predictions * labels) * (-1) + 1)
        .fmap([](auto&& x) { return x > 0 ? x : 0; })
        .sum();
  }
  template <class T, class U>
  static SCENN_CONSTEXPR auto loss_derivative(const T& predictions,
                                              const U& labels) {
    return ((predictions * labels) * (-1) + 1)
        .fmap_with_index(
            [&labels](auto&& x, auto&& i) { return x > 0 ? -labels[i] : 0; });
  }
};
}  // namespace scenn
#endif
