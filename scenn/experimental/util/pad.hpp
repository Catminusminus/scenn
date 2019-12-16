#ifndef SCENN_EXPERIMENTAL_UTIL_PAD_HPP
#define SCENN_EXPERIMENTAL_UTIL_PAD_HPP

#include <scenn/experimental/tensor/tensor3d.hpp>
#include <scenn/util.hpp>

namespace scenn::experimental {
template <std::size_t Pad, std::size_t L, std::size_t M, std::size_t N, class T>
SCENN_CONSTEXPR auto pad(const Tensor3D<L, M, N, T>& tensor) {
  /**
  Tensor3D<L, Pad + M + Pad, N, T> ret;
  for (std::size_t i = 0; i < L; ++i)
    for (std::size_t j = 0; j < Pad + M + Pad; ++j)
      for (std::size_t k = 0; k < N; ++k)
        ret(i, j, k) = Pad <= j && j < Pad + M ? tensor(i, j - Pad, k) : 0;
  return ret;
  */
  Tensor3D<L, M, Pad + N + Pad, T> ret;
  for (std::size_t i = 0; i < L; ++i)
    for (std::size_t j = 0; j < M; ++j)
      for (std::size_t k = 0; k < Pad + N + Pad; ++k)
        ret(i, j, k) = Pad <= k && k < Pad + N ? tensor(i, j, k - Pad) : 0;
  return ret;
}
}  // namespace scenn::experimental

#endif
