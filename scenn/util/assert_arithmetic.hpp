#ifndef SCENN_UTIL_ASSERT_ARITHMETIC_HPP
#define SCENN_UTIL_ASSERT_ARITHMETIC_HPP

#include <scenn/util/config.hpp>
#include <type_traits>

template <class T>
SCENN_CONSTEXPR auto assert_arithmetic() {
  if constexpr (!std::is_arithmetic_v<T>) static_assert([] { return false; }());
}

#endif