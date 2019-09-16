#ifndef SCENN_EXPERIMENTAL_OPTIMIZER_ADAM_HPP
#define SCENN_EXPERIMENTAL_OPTIMIZER_ADAM_HPP

namespace scenn::experimental {
template <class T>
struct Adam {
  T lr, beta1, beta2, epsilon;
  constexpr Adam()
};
}

#endif
