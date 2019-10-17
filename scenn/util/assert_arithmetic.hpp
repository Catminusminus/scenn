#ifndef SCENN_UTIL_ASSERT_ARITHMETIC_HPP
#define SCENN_UTIL_ASSERT_ARITHMETIC_HPP

#include <scenn/util/config.hpp>
#include <type_traits>

template <class T>
SCENN_CONSTEXPR auto assert_arithmetic() {
  static_assert(std::is_arithmetic_v<T>);
}

#endif