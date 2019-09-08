#ifndef SCENN_LOSS_MSLE_HPP
#define SCENN_LOSS_MSLE_HPP

#include <scenn/util.hpp>
#include <sprout/math.hpp>

namespace scenn {
struct MSLE {
  template <class T, class U>
  static SCENN_CONSTEXPR auto loss_function(const T& predictions,
                                            const U& labels) {
    auto diff =
        (predictions + 1).fmap(sprout::math::log<predictions::value_type>) -
        (labels + 1).fmap(sprout::math::log<labels::value_type>);
    return (diff.dot(diff.transposed()) * 0.5).to_value();
  }
  template <class T, class U>
  static SCENN_CONSTEXPR auto loss_derivative(const T& predictions,
                                              const U& labels) {
    return ((predictions + 1).fmap(sprout::math::log<predictions::value_type>) -
            (labels + 1).fmap(sprout::math::log<labels::value_type>)) /
           (predictions + 1);
  }
};
}  // namespace scenn
#endif
