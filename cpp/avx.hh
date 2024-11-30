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

void solve_avx512_loop_unfold(
    const uint8_t *y_data,
    const uint8_t *u_data,
    const uint8_t *v_data,
    uint8_t **y_result,
    uint8_t **u_result,
    uint8_t **v_result);

void test();

void solve_avx512_part3(
    const uint8_t *p1_y_data,
    const uint8_t *p1_u_data,
    const uint8_t *p1_v_data,
    const uint8_t *p2_y_data,
    const uint8_t *p2_u_data,
    const uint8_t *p2_v_data,
    uint8_t **y_result,
    uint8_t **u_result,
    uint8_t **v_result);

#endif // AVX_HH