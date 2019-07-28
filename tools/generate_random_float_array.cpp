// This code is based on https://cpprefjp.github.io/reference/random.html
#include <fstream>
#include <random>

int main() {
  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());

  std::normal_distribution<float> dist(0.0, 1.0);

  std::ofstream file("../scenn/matrix/csv/normal_distribution_float.csv");
  for (std::size_t n = 0; n < 1000 * 1000; ++n) {
    double result = dist(engine);
    file << result << ",\n";
  }
}
