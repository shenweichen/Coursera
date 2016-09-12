#include <iostream>

long long lcm_naive(int a, int b) {
  for (long l = 1; l <= (long long)a * b; ++l)
    if (l % a == 0 && l % b == 0)
      return l;

  return (long long)a * b;
}

int gcd_fast(int a, int b) {
  if (b == 0)
    return a;
  return gcd_fast(b, a % b);
}

long long lcm_fast(int a, int b) { return a / gcd_fast(a, b) * b; }

int main() {
  int a, b;
  std::cin >> a >> b;
  std::cout << lcm_fast(a, b) << std::endl;
  return 0;
}
