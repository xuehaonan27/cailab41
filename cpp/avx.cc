#include "avx.hh"

#include "misc.hh"
#include <cstdint>
#include <immintrin.h> // immintrin is for AVX

void solve_avx512(
    const uint8_t *y_data,
    const uint8_t *u_data,
    const uint8_t *v_data,
    uint8_t **y_result,
    uint8_t **u_result,
    uint8_t **v_result)
{
#define VECTOR_SIZE 32 // An AVX512 vector register could hold 32 uint16_t

    const __m512i u16_512_16 = _mm512_set1_epi16(16);
    const __m512i u16_512_298 = _mm512_set1_epi16(298);

    for (int image_idx = 0; image_idx < 84; image_idx++)
    {
        const uint8_t alpha = 1 + image_idx * 3;
        for (int j = 0; j < HEIGHT; j += 2) // read 2 Y data lines a time
        {
            for (int i = 0; i < WIDTH; i += VECTOR_SIZE)
            {
                size_t y_index = j * WIDTH + i;
                size_t uv_index = size_t(j / 2) * size_t(WIDTH / 2) + size_t(i / 2);

                // Load data
                __m512i y_vec_1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *)(y_data + y_index)));
                __m512i y_vec_2 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *)(y_data + WIDTH + y_index)));
                __m128i u_vec_8 = _mm_loadu_si128((__m128i *)(u_data + uv_index));
                __m128i v_vec_8 = _mm_loadu_si128((__m128i *)(v_data + uv_index));
                __m128i u_vec_8_copy = u_vec_8;
                __m128i v_vec_8_copy = v_vec_8;

                __m256i u_vec_8_low = _mm256_castsi128_si256(u_vec_8);
                __m256i u_vec_8_high = _mm256_castsi128_si256(u_vec_8_copy);
                __m256i v_vec_8_low = _mm256_castsi128_si256(v_vec_8);
                __m256i v_vec_8_high = _mm256_castsi128_si256(v_vec_8_copy);

                __m256i u_vec_repeated = _mm256_unpacklo_epi8(u_vec_8_low, u_vec_8_high);
                __m256i v_vec_repeated = _mm256_unpacklo_epi8(v_vec_8_low, v_vec_8_high);

                __m512i u_vec = _mm512_cvtepu8_epi16(u_vec_repeated);
                __m512i v_vec = _mm512_cvtepu8_epi16(v_vec_repeated);

                // YUV -> ARGB
                
            }
        }
    }
}
