#ifndef SCENN_UTIL_ASSERT_FLOATING_POINT_HPP
#define SCENN_UTIL_ASSERT_FLOATING_POINT_HPP

#include <scenn/util/config.hpp>
#include <type_traits>

template <class T>
SCENN_CONSTEXPR auto assert_floating_point() {
  if constexpr (!std::is_floating_point_v<T>)
    static_assert([] { return false; }());
}

#endif
