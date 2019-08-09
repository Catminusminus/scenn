#include <scenn/activation.hpp>
#include <scenn/layer/activation_layer.hpp>

int main() {
  using namespace scenn;
  constexpr auto sigmoid_activation_layer = ActivationLayer<1, float>(Sigmoid());
  [[maybe_unused]] constexpr auto deltas_cleared_updated =
      sigmoid_activation_layer.make_by_input_data(make_zeros_from_pair<1, 1, float>())
          .make_by_input_delta(make_zeros_from_pair<1, 1, float>())
          .clear_deltas()
          .update_params(1.0F);
}
