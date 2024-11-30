#ifndef SSE2_HH
#define SSE2_HH

#include <cstdint>

void solve_sse2_part2(
    const uint8_t *y_data,
    const uint8_t *u_data,
    const uint8_t *v_data,
    uint8_t **y_result,
    uint8_t **u_result,
    uint8_t **v_result);

#endif // SSE2_HH