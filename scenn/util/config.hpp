#ifndef SCENN_UTIL_CONFIG_HPP
#define SCENN_UTIL_CONFIG_HPP
// #define SCENN_DISABLE_CONSTEXPR
#ifndef SCENN_DISABLE_CONSTEXPR
#define SCENN_CONSTEXPR constexpr
#define SCENN_ASSERT static_assert
#define SCENN_STATIC
#else
#define SCENN_CONSTEXPR
#define SCENN_ASSERT assert
#define SCENN_STATIC static
#endif

#endif
