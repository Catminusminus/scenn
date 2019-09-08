#ifndef SCENN_LOSS_HINGE_HPP
#define SCENN_LOSS_HINGE_HPP

#include <scenn/util.hpp>

namespace scenn {
struct Hinge {
  template <class T, class U>
  static SCENN_CONSTEXPR auto loss_function(const T& predictions,
                                            const U& labels) {
    return (1 - (predictions * labels))
        .fmap([](auto&& x) { return x > 0 ? x : 0; })
        .sum();
  }
  template <class T, class U>
  static SCENN_CONSTEXPR auto loss_derivative(const T& predictions,
                                              const U& labels) {
    return (1 - (predictions * labels)).fmap_with_index([](auto&& x, auto&& i) {
      return x > 0 ? -labels[i] : 0;
    });
  }
};
}  // namespace scenn
#endif
