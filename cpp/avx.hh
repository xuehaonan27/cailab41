#ifndef AVX_HH
#define AVX_HH

void solve_avx512(
    const unsigned char *y_data,
    const unsigned char *u_data,
    const unsigned char *v_data,
    unsigned char **y_result,
    unsigned char **u_result,
    unsigned char **v_result);

#endif // AVX_HH