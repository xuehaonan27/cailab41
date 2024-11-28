#ifndef AVX_HH
#define AVX_HH

#include <cstdint>

void solve_avx512(
    const uint8_t *y_data,
    const uint8_t *u_data,
    const uint8_t *v_data,
    uint8_t **y_result,
    uint8_t **u_result,
    uint8_t **v_result);

void test();

#endif // AVX_HH