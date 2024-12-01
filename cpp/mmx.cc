#include "mmx.hh"

#include "misc.hh"
#include <xmmintrin.h> // mmintrin is for MMX
#include <cstdint>

#define VECTOR_SIZE 4

static __m64 u16_512_16 = _mm_set1_pi16(16);
static __m64 u16_512_298 = _mm_set1_pi16(298);
static __m64 u16_512_409 = _mm_set1_pi16(409);
static __m64 u16_512_128 = _mm_set1_pi16(128);
static __m64 u16_512_8 = _mm_set1_pi16(8);
static __m64 u16_512_100 = _mm_set1_pi16(100);
static __m64 u16_512_208 = _mm_set1_pi16(208);
static __m64 u16_512_516 = _mm_set1_pi16(516);
static __m64 u16_512_0xff = _mm_set1_pi16(0xff);
static __m64 u16_512_255 = _mm_set1_pi16(255);
static __m64 u16_512_0 = _mm_set1_pi16(0);
static __m64 u16_512_66 = _mm_set1_pi16(66);
static __m64 u16_512_129 = _mm_set1_pi16(129);
static __m64 u16_512_25 = _mm_set1_pi16(25);
static __m64 u16_512_minus38 = _mm_set1_pi16(-38);
static __m64 u16_512_74 = _mm_set1_pi16(74);
static __m64 u16_512_112 = _mm_set1_pi16(112);
static __m64 u16_512_94 = _mm_set1_pi16(94);
static __m64 u16_512_18 = _mm_set1_pi16(18);
static __m64 u16_512_256 = _mm_set1_pi16(256);
static __m64 u16_512_42 = _mm_set1_pi16(42);
static __m64 u16_512_153 = _mm_set1_pi16(153);
static __m64 u16_512_4 = _mm_set1_pi16(4);
static __m64 u16_512_2 = _mm_set1_pi16(2);
static __m64 u16_512_38 = _mm_set1_pi16(38);
static __m64 u16_512_24224 = _mm_set1_pi16(24224);
static __m64 u16_512_2016 = _mm_set1_pi16(2016);
static __m64 u16_512_5152 = _mm_set1_pi16(5152);
static __m64 u16_512_8192 = _mm_set1_pi16(8192);
static __m64 u16_512_16384 = _mm_set1_pi16(16384);
static __m64 u16_512_1920 = _mm_set1_pi16(1920);
static __m64 u16_512_120 = _mm_set1_pi16(120);
static __m64 u16_512_47 = _mm_set1_pi16(47);
const int imm8 = 0xD8;

void solve_mmx_part2(
    const uint8_t *y_data,
    const uint8_t *u_data,
    const uint8_t *v_data,
    uint8_t **y_result,
    uint8_t **u_result,
    uint8_t **v_result)
{
    for (int image_idx = 0; image_idx < 84; image_idx++)
    {
        const uint8_t alpha = 1 + image_idx * 3;
        __m64 alpha_vec = _mm_set1_pi16(alpha);

        for (int j = 0; j < HEIGHT; j += 2) // read 2 Y data lines a time
        {
            for (int i = 0; i < WIDTH; i += VECTOR_SIZE)
            {
                size_t y_index_1 = j * WIDTH + i;
                size_t y_index_2 = (j + 1) * WIDTH + i;
                size_t uv_index = size_t(j / 2) * size_t(WIDTH / 2) + size_t(i / 2);

                // Load data
                __m64 y_vec_1 = _mm_unpacklo_pi8(_mm_cvtsi32_si64(*(int *)(y_data + y_index_1)), _mm_setzero_si64());
                __m64 y_vec_2 = _mm_unpacklo_pi8(_mm_cvtsi32_si64(*(int *)(y_data + y_index_2)), _mm_setzero_si64());
                uint16_t u1 = (uint16_t)(*((uint8_t *)u_data + uv_index));
                uint16_t u2 = (uint16_t)(*((uint8_t *)u_data + uv_index + 1));
                uint16_t v1 = (uint16_t)(*((uint8_t *)v_data + uv_index));
                uint16_t v2 = (uint16_t)(*((uint8_t *)v_data + uv_index + 1));
                __m64 u_vec = _mm_set_pi16(u2, u2, u1, u1);
                __m64 v_vec = _mm_set_pi16(v2, v2, v1, v1);

                // YUV -> ARGB
                __m64 y_vec_1_mul_42 = _mm_mullo_pi16(u16_512_42, y_vec_1); // 42 * (y1)
                __m64 y_vec_2_mul_42 = _mm_mullo_pi16(u16_512_42, y_vec_2); // 42 * (y2)
                __m64 v_vec_mul_153 = _mm_mullo_pi16(u16_512_153, v_vec);   // 153 * (v)
                __m64 u_vec_mul_100 = _mm_mullo_pi16(u16_512_100, u_vec);   // 100 * (u)
                __m64 v_vec_mul_47 = _mm_mullo_pi16(u16_512_47, v_vec);     // 47 * (v)
                __m64 u_vec_mul_4 = _mm_mullo_pi16(u16_512_4, u_vec);       // 4 * (u)

                __m64 r_yvec1_unclamped =
                    _mm_add_pi16(
                        _mm_srai_pi16(
                            _mm_add_pi16(
                                _mm_sub_pi16(y_vec_1_mul_42, u16_512_24224), v_vec_mul_153),
                            8),
                        _mm_add_pi16(_mm_sub_pi16(y_vec_1, u16_512_128), v_vec));

                __m64 r_yvec2_unclamped =
                    _mm_add_pi16(
                        _mm_srai_pi16(
                            _mm_add_pi16(
                                _mm_sub_pi16(y_vec_2_mul_42, u16_512_24224), v_vec_mul_153),
                            8),
                        _mm_add_pi16(_mm_sub_pi16(y_vec_2, u16_512_128), v_vec));

                __m64 g_yvec1_unclamped =
                    _mm_add_pi16(
                        _mm_srai_pi16(
                            _mm_add_pi16(
                                _mm_sub_pi16(y_vec_1_mul_42, u_vec_mul_100),
                                _mm_add_pi16(u16_512_2016, v_vec_mul_47)),
                            8),
                        _mm_sub_pi16(_mm_add_pi16(y_vec_1, u16_512_128), v_vec));

                __m64 g_yvec2_unclamped =
                    _mm_add_pi16(
                        _mm_srai_pi16(
                            _mm_add_pi16(
                                _mm_sub_pi16(y_vec_2_mul_42, u_vec_mul_100),
                                _mm_add_pi16(u16_512_2016, v_vec_mul_47)),
                            8),
                        _mm_sub_pi16(_mm_add_pi16(y_vec_2, u16_512_128), v_vec));

                __m64 b_yvec1_unclamped =
                    _mm_add_pi16(
                        _mm_srai_pi16(
                            _mm_sub_pi16(
                                _mm_add_pi16(
                                    y_vec_1_mul_42,
                                    u_vec_mul_4),
                                u16_512_5152),
                            8),
                        _mm_add_pi16(
                            _mm_sub_pi16(y_vec_1, u16_512_256),
                            _mm_mullo_pi16(u16_512_2, u_vec)));

                __m64 b_yvec2_unclamped =
                    _mm_add_pi16(
                        _mm_srai_pi16(
                            _mm_sub_pi16(
                                _mm_add_pi16(
                                    y_vec_2_mul_42,
                                    u_vec_mul_4),
                                u16_512_5152),
                            8),
                        _mm_add_pi16(
                            _mm_sub_pi16(y_vec_2, u16_512_256),
                            _mm_mullo_pi16(u16_512_2, u_vec)));

                __m64 r_yvec1_clamped = _mm_max_pi16(_mm_min_pi16(r_yvec1_unclamped, u16_512_255), u16_512_0);
                __m64 r_yvec2_clamped = _mm_max_pi16(_mm_min_pi16(r_yvec2_unclamped, u16_512_255), u16_512_0);
                __m64 g_yvec1_clamped = _mm_max_pi16(_mm_min_pi16(g_yvec1_unclamped, u16_512_255), u16_512_0);
                __m64 g_yvec2_clamped = _mm_max_pi16(_mm_min_pi16(g_yvec2_unclamped, u16_512_255), u16_512_0);
                __m64 b_yvec1_clamped = _mm_max_pi16(_mm_min_pi16(b_yvec1_unclamped, u16_512_255), u16_512_0);
                __m64 b_yvec2_clamped = _mm_max_pi16(_mm_min_pi16(b_yvec2_unclamped, u16_512_255), u16_512_0);
                __m64 r_yvec1 = _mm_and_si64(r_yvec1_clamped, u16_512_0xff);
                __m64 r_yvec2 = _mm_and_si64(r_yvec2_clamped, u16_512_0xff);
                __m64 g_yvec1 = _mm_and_si64(g_yvec1_clamped, u16_512_0xff);
                __m64 g_yvec2 = _mm_and_si64(g_yvec2_clamped, u16_512_0xff);
                __m64 b_yvec1 = _mm_and_si64(b_yvec1_clamped, u16_512_0xff);
                __m64 b_yvec2 = _mm_and_si64(b_yvec2_clamped, u16_512_0xff);

                // Alpha mix
                __m64 r_prime_yvec1 = _mm_srli_pi16(_mm_mullo_pi16(alpha_vec, r_yvec1), 8);
                __m64 g_prime_yvec1 = _mm_srli_pi16(_mm_mullo_pi16(alpha_vec, g_yvec1), 8);
                __m64 b_prime_yvec1 = _mm_srli_pi16(_mm_mullo_pi16(alpha_vec, b_yvec1), 8);
                __m64 r_prime_yvec2 = _mm_srli_pi16(_mm_mullo_pi16(alpha_vec, r_yvec2), 8);
                __m64 g_prime_yvec2 = _mm_srli_pi16(_mm_mullo_pi16(alpha_vec, g_yvec2), 8);
                __m64 b_prime_yvec2 = _mm_srli_pi16(_mm_mullo_pi16(alpha_vec, b_yvec2), 8);

                // ARGB -> YUV
                __m64 y_prime_yvec1 =
                    _mm_add_pi16(
                        _mm_srai_pi16(
                            _mm_add_pi16(
                                _mm_add_pi16(
                                    _mm_sub_pi16(_mm_mullo_pi16(u16_512_66, r_prime_yvec1), u16_512_8192),
                                    _mm_sub_pi16(_mm_mullo_pi16(u16_512_129, g_prime_yvec1), u16_512_16384)),
                                _mm_mullo_pi16(u16_512_25, b_prime_yvec1)),
                            8),
                        u16_512_112);

                __m64 y_prime_yvec2 =
                    _mm_add_pi16(
                        _mm_srai_pi16(
                            _mm_add_pi16(
                                _mm_add_pi16(
                                    _mm_sub_pi16(_mm_mullo_pi16(u16_512_66, r_prime_yvec2), u16_512_8192),
                                    _mm_sub_pi16(_mm_mullo_pi16(u16_512_129, g_prime_yvec2), u16_512_16384)),
                                _mm_mullo_pi16(u16_512_25, b_prime_yvec2)),
                            8),
                        u16_512_112);

                __m64 u_prime_yvec1 =
                    _mm_add_pi16(
                        _mm_srai_pi16(
                            _mm_add_pi16(
                                _mm_sub_pi16(
                                    _mm_mullo_pi16(u16_512_112, b_prime_yvec1),
                                    _mm_mullo_pi16(u16_512_38, r_prime_yvec1)),
                                _mm_sub_pi16(
                                    u16_512_128,
                                    _mm_mullo_pi16(u16_512_74, g_prime_yvec1))),
                            8),
                        u16_512_128);

                __m64 v_prime_yvec1 =
                    _mm_add_pi16(
                        _mm_srai_pi16(
                            _mm_add_pi16(
                                _mm_sub_pi16(
                                    _mm_mullo_pi16(u16_512_112, r_prime_yvec1),
                                    _mm_mullo_pi16(u16_512_94, g_prime_yvec1)),
                                _mm_sub_pi16(
                                    u16_512_128,
                                    _mm_mullo_pi16(u16_512_18, b_prime_yvec1))),
                            8),
                        u16_512_128);

                int y_prime_yvec1_packed = _mm_cvtsi64_si32(_mm_packs_pu16(y_prime_yvec1, _mm_setzero_si64()));
                int y_prime_yvec2_packed = _mm_cvtsi64_si32(_mm_packs_pu16(y_prime_yvec2, _mm_setzero_si64()));
                int u_prime_yvec1_packed = _mm_cvtsi64_si32(_mm_packs_pu16(u_prime_yvec1, _mm_setzero_si64()));
                int v_prime_yvec1_packed = _mm_cvtsi64_si32(_mm_packs_pu16(v_prime_yvec1, _mm_setzero_si64()));
                uint16_t u_prime_yvec1_extracted = (uint16_t)((u_prime_yvec1_packed >> 8) & 0xffff);
                uint16_t v_prime_yvec1_extracted = (uint16_t)((v_prime_yvec1_packed >> 8) & 0xffff);

                // Store value
                *(int *)((uint8_t *)y_result + image_idx * Y_SIZE + y_index_1) = y_prime_yvec1_packed;
                *(int *)((uint8_t *)y_result + image_idx * Y_SIZE + y_index_2) = y_prime_yvec2_packed;
                *(uint16_t *)((uint8_t *)u_result + image_idx * U_SIZE + uv_index) = u_prime_yvec1_extracted;
                *(uint16_t *)((uint8_t *)v_result + image_idx * V_SIZE + uv_index) = v_prime_yvec1_extracted;
            }
        }
    }
}

void solve_mmx_part3(
    const uint8_t *p1_y_data,
    const uint8_t *p1_u_data,
    const uint8_t *p1_v_data,
    const uint8_t *p2_y_data,
    const uint8_t *p2_u_data,
    const uint8_t *p2_v_data,
    uint8_t **y_result,
    uint8_t **u_result,
    uint8_t **v_result)
{

    for (int image_idx = 0; image_idx < 84; image_idx++)
    {
        const uint8_t alpha = 1 + image_idx * 3;
        __m64 alpha_vec = _mm_set1_pi16(alpha);
        const __m64 alpha_256_minus_vec = _mm_sub_pi16(u16_512_256, alpha_vec);

        for (int j = 0; j < HEIGHT; j += 2) // read 2 Y data lines a time
        {
            for (int i = 0; i < WIDTH; i += VECTOR_SIZE)
            {

                size_t y_index_1 = j * WIDTH + i;
                size_t y_index_2 = (j + 1) * WIDTH + i;
                size_t uv_index = size_t(j / 2) * size_t(WIDTH / 2) + size_t(i / 2);

                // Load data
                __m64 p1_y_vec_1 = _mm_unpacklo_pi8(_mm_cvtsi32_si64(*(int *)(p1_y_data + y_index_1)), _mm_setzero_si64());
                __m64 p1_y_vec_2 = _mm_unpacklo_pi8(_mm_cvtsi32_si64(*(int *)(p1_y_data + y_index_2)), _mm_setzero_si64());
                __m64 p2_y_vec_1 = _mm_unpacklo_pi8(_mm_cvtsi32_si64(*(int *)(p2_y_data + y_index_1)), _mm_setzero_si64());
                __m64 p2_y_vec_2 = _mm_unpacklo_pi8(_mm_cvtsi32_si64(*(int *)(p2_y_data + y_index_2)), _mm_setzero_si64());
                uint16_t p1_u1 = (uint16_t)(*((uint8_t *)p1_u_data + uv_index));
                uint16_t p1_u2 = (uint16_t)(*((uint8_t *)p1_u_data + uv_index + 1));
                uint16_t p1_v1 = (uint16_t)(*((uint8_t *)p1_v_data + uv_index));
                uint16_t p1_v2 = (uint16_t)(*((uint8_t *)p1_v_data + uv_index + 1));
                uint16_t p2_u1 = (uint16_t)(*((uint8_t *)p2_u_data + uv_index));
                uint16_t p2_u2 = (uint16_t)(*((uint8_t *)p2_u_data + uv_index + 1));
                uint16_t p2_v1 = (uint16_t)(*((uint8_t *)p2_v_data + uv_index));
                uint16_t p2_v2 = (uint16_t)(*((uint8_t *)p2_v_data + uv_index + 1));
                __m64 p1_u_vec = _mm_set_pi16(p1_u2, p1_u2, p1_u1, p1_u1);
                __m64 p1_v_vec = _mm_set_pi16(p1_v2, p1_v2, p1_v1, p1_v1);
                __m64 p2_u_vec = _mm_set_pi16(p2_u2, p2_u2, p2_u1, p2_u1);
                __m64 p2_v_vec = _mm_set_pi16(p2_v2, p2_v2, p2_v1, p2_v1);

                // YUV -> ARGB
                __m64 p1_y_vec_1_mul_42 = _mm_mullo_pi16(u16_512_42, p1_y_vec_1); // 42 * (y1)
                __m64 p1_y_vec_2_mul_42 = _mm_mullo_pi16(u16_512_42, p1_y_vec_2); // 42 * (y2)
                __m64 p1_v_vec_mul_153 = _mm_mullo_pi16(u16_512_153, p1_v_vec);   // 153 * (v)
                __m64 p1_u_vec_mul_100 = _mm_mullo_pi16(u16_512_100, p1_u_vec);   // 100 * (u)
                __m64 p1_v_vec_mul_47 = _mm_mullo_pi16(u16_512_47, p1_v_vec);     // 47 * (v)
                __m64 p1_u_vec_mul_4 = _mm_mullo_pi16(u16_512_4, p1_u_vec);       // 4 * (u)

                __m64 p1_r_yvec1_unclamped =
                    _mm_add_pi16(
                        _mm_srai_pi16(
                            _mm_add_pi16(
                                _mm_sub_pi16(p1_y_vec_1_mul_42, u16_512_24224), p1_v_vec_mul_153),
                            8),
                        _mm_add_pi16(_mm_sub_pi16(p1_y_vec_1, u16_512_128), p1_v_vec));

                __m64 p1_r_yvec2_unclamped =
                    _mm_add_pi16(
                        _mm_srai_pi16(
                            _mm_add_pi16(
                                _mm_sub_pi16(p1_y_vec_2_mul_42, u16_512_24224), p1_v_vec_mul_153),
                            8),
                        _mm_add_pi16(_mm_sub_pi16(p1_y_vec_2, u16_512_128), p1_v_vec));

                __m64 p1_g_yvec1_unclamped =
                    _mm_add_pi16(
                        _mm_srai_pi16(
                            _mm_add_pi16(
                                _mm_sub_pi16(p1_y_vec_1_mul_42, p1_u_vec_mul_100),
                                _mm_add_pi16(u16_512_2016, p1_v_vec_mul_47)),
                            8),
                        _mm_sub_pi16(_mm_add_pi16(p1_y_vec_1, u16_512_128), p1_v_vec));

                __m64 p1_g_yvec2_unclamped =
                    _mm_add_pi16(
                        _mm_srai_pi16(
                            _mm_add_pi16(
                                _mm_sub_pi16(p1_y_vec_2_mul_42, p1_u_vec_mul_100),
                                _mm_add_pi16(u16_512_2016, p1_v_vec_mul_47)),
                            8),
                        _mm_sub_pi16(_mm_add_pi16(p1_y_vec_2, u16_512_128), p1_v_vec));

                __m64 p1_b_yvec1_unclamped =
                    _mm_add_pi16(
                        _mm_srai_pi16(
                            _mm_sub_pi16(
                                _mm_add_pi16(
                                    p1_y_vec_1_mul_42,
                                    p1_u_vec_mul_4),
                                u16_512_5152),
                            8),
                        _mm_add_pi16(
                            _mm_sub_pi16(p1_y_vec_1, u16_512_256),
                            _mm_mullo_pi16(u16_512_2, p1_u_vec)));

                __m64 p1_b_yvec2_unclamped =
                    _mm_add_pi16(
                        _mm_srai_pi16(
                            _mm_sub_pi16(
                                _mm_add_pi16(
                                    p1_y_vec_2_mul_42,
                                    p1_u_vec_mul_4),
                                u16_512_5152),
                            8),
                        _mm_add_pi16(
                            _mm_sub_pi16(p1_y_vec_2, u16_512_256),
                            _mm_mullo_pi16(u16_512_2, p1_u_vec)));

                __m64 p1_r_yvec1_clamped = _mm_max_pi16(_mm_min_pi16(p1_r_yvec1_unclamped, u16_512_255), u16_512_0);
                __m64 p1_r_yvec2_clamped = _mm_max_pi16(_mm_min_pi16(p1_r_yvec2_unclamped, u16_512_255), u16_512_0);
                __m64 p1_g_yvec1_clamped = _mm_max_pi16(_mm_min_pi16(p1_g_yvec1_unclamped, u16_512_255), u16_512_0);
                __m64 p1_g_yvec2_clamped = _mm_max_pi16(_mm_min_pi16(p1_g_yvec2_unclamped, u16_512_255), u16_512_0);
                __m64 p1_b_yvec1_clamped = _mm_max_pi16(_mm_min_pi16(p1_b_yvec1_unclamped, u16_512_255), u16_512_0);
                __m64 p1_b_yvec2_clamped = _mm_max_pi16(_mm_min_pi16(p1_b_yvec2_unclamped, u16_512_255), u16_512_0);
                __m64 p1_r_yvec1 = _mm_and_si64(p1_r_yvec1_clamped, u16_512_0xff);
                __m64 p1_r_yvec2 = _mm_and_si64(p1_r_yvec2_clamped, u16_512_0xff);
                __m64 p1_g_yvec1 = _mm_and_si64(p1_g_yvec1_clamped, u16_512_0xff);
                __m64 p1_g_yvec2 = _mm_and_si64(p1_g_yvec2_clamped, u16_512_0xff);
                __m64 p1_b_yvec1 = _mm_and_si64(p1_b_yvec1_clamped, u16_512_0xff);
                __m64 p1_b_yvec2 = _mm_and_si64(p1_b_yvec2_clamped, u16_512_0xff);

                // YUV -> ARGB
                __m64 p2_y_vec_1_mul_42 = _mm_mullo_pi16(u16_512_42, p2_y_vec_1); // 42 * (y1)
                __m64 p2_y_vec_2_mul_42 = _mm_mullo_pi16(u16_512_42, p2_y_vec_2); // 42 * (y2)
                __m64 p2_v_vec_mul_153 = _mm_mullo_pi16(u16_512_153, p2_v_vec);   // 153 * (v)
                __m64 p2_u_vec_mul_100 = _mm_mullo_pi16(u16_512_100, p2_u_vec);   // 100 * (u)
                __m64 p2_v_vec_mul_47 = _mm_mullo_pi16(u16_512_47, p2_v_vec);     // 47 * (v)
                __m64 p2_u_vec_mul_4 = _mm_mullo_pi16(u16_512_4, p2_u_vec);       // 4 * (u)

                __m64 p2_r_yvec1_unclamped =
                    _mm_add_pi16(
                        _mm_srai_pi16(
                            _mm_add_pi16(
                                _mm_sub_pi16(p2_y_vec_1_mul_42, u16_512_24224), p2_v_vec_mul_153),
                            8),
                        _mm_add_pi16(_mm_sub_pi16(p2_y_vec_1, u16_512_128), p2_v_vec));

                __m64 p2_r_yvec2_unclamped =
                    _mm_add_pi16(
                        _mm_srai_pi16(
                            _mm_add_pi16(
                                _mm_sub_pi16(p2_y_vec_2_mul_42, u16_512_24224), p2_v_vec_mul_153),
                            8),
                        _mm_add_pi16(_mm_sub_pi16(p2_y_vec_2, u16_512_128), p2_v_vec));

                __m64 p2_g_yvec1_unclamped =
                    _mm_add_pi16(
                        _mm_srai_pi16(
                            _mm_add_pi16(
                                _mm_sub_pi16(p2_y_vec_1_mul_42, p2_u_vec_mul_100),
                                _mm_add_pi16(u16_512_2016, p2_v_vec_mul_47)),
                            8),
                        _mm_sub_pi16(_mm_add_pi16(p2_y_vec_1, u16_512_128), p2_v_vec));

                __m64 p2_g_yvec2_unclamped =
                    _mm_add_pi16(
                        _mm_srai_pi16(
                            _mm_add_pi16(
                                _mm_sub_pi16(p2_y_vec_2_mul_42, p2_u_vec_mul_100),
                                _mm_add_pi16(u16_512_2016, p2_v_vec_mul_47)),
                            8),
                        _mm_sub_pi16(_mm_add_pi16(p2_y_vec_2, u16_512_128), p2_v_vec));

                __m64 p2_b_yvec1_unclamped =
                    _mm_add_pi16(
                        _mm_srai_pi16(
                            _mm_sub_pi16(
                                _mm_add_pi16(
                                    p2_y_vec_1_mul_42,
                                    p2_u_vec_mul_4),
                                u16_512_5152),
                            8),
                        _mm_add_pi16(
                            _mm_sub_pi16(p2_y_vec_1, u16_512_256),
                            _mm_mullo_pi16(u16_512_2, p2_u_vec)));

                __m64 p2_b_yvec2_unclamped =
                    _mm_add_pi16(
                        _mm_srai_pi16(
                            _mm_sub_pi16(
                                _mm_add_pi16(
                                    p2_y_vec_2_mul_42,
                                    p2_u_vec_mul_4),
                                u16_512_5152),
                            8),
                        _mm_add_pi16(
                            _mm_sub_pi16(p2_y_vec_2, u16_512_256),
                            _mm_mullo_pi16(u16_512_2, p2_u_vec)));

                __m64 p2_r_yvec1_clamped = _mm_max_pi16(_mm_min_pi16(p2_r_yvec1_unclamped, u16_512_255), u16_512_0);
                __m64 p2_r_yvec2_clamped = _mm_max_pi16(_mm_min_pi16(p2_r_yvec2_unclamped, u16_512_255), u16_512_0);
                __m64 p2_g_yvec1_clamped = _mm_max_pi16(_mm_min_pi16(p2_g_yvec1_unclamped, u16_512_255), u16_512_0);
                __m64 p2_g_yvec2_clamped = _mm_max_pi16(_mm_min_pi16(p2_g_yvec2_unclamped, u16_512_255), u16_512_0);
                __m64 p2_b_yvec1_clamped = _mm_max_pi16(_mm_min_pi16(p2_b_yvec1_unclamped, u16_512_255), u16_512_0);
                __m64 p2_b_yvec2_clamped = _mm_max_pi16(_mm_min_pi16(p2_b_yvec2_unclamped, u16_512_255), u16_512_0);
                __m64 p2_r_yvec1 = _mm_and_si64(p2_r_yvec1_clamped, u16_512_0xff);
                __m64 p2_r_yvec2 = _mm_and_si64(p2_r_yvec2_clamped, u16_512_0xff);
                __m64 p2_g_yvec1 = _mm_and_si64(p2_g_yvec1_clamped, u16_512_0xff);
                __m64 p2_g_yvec2 = _mm_and_si64(p2_g_yvec2_clamped, u16_512_0xff);
                __m64 p2_b_yvec1 = _mm_and_si64(p2_b_yvec1_clamped, u16_512_0xff);
                __m64 p2_b_yvec2 = _mm_and_si64(p2_b_yvec2_clamped, u16_512_0xff);

                // #define MIX(a, x1, x2) ((((a) * ((x1) - (x2))) >> 8) + x2)
                // image overlap
                // #define MIX(a, x1, x2) (((((a) * (x1)) + ((256 - (a)) * (x2))) + 128) >> 8)
                // Max = 256 * 255 + 128 = 65408 < 65535
                __m64 r_prime_yvec1 =
                    _mm_srli_pi16(
                        _mm_add_pi16(
                            _mm_add_pi16(
                                _mm_mullo_pi16(alpha_vec, p1_r_yvec1),
                                _mm_mullo_pi16(
                                    alpha_256_minus_vec,
                                    p2_r_yvec1)),
                            u16_512_128),
                        8);
                __m64 g_prime_yvec1 =
                    _mm_srli_pi16(
                        _mm_add_pi16(
                            _mm_add_pi16(
                                _mm_mullo_pi16(alpha_vec, p1_g_yvec1),
                                _mm_mullo_pi16(
                                    alpha_256_minus_vec,
                                    p2_g_yvec1)),
                            u16_512_128),
                        8);
                __m64 b_prime_yvec1 =
                    _mm_srli_pi16(
                        _mm_add_pi16(
                            _mm_add_pi16(
                                _mm_mullo_pi16(alpha_vec, p1_b_yvec1),
                                _mm_mullo_pi16(
                                    alpha_256_minus_vec,
                                    p2_b_yvec1)),
                            u16_512_128),
                        8);

                __m64 r_prime_yvec2 =
                    _mm_srli_pi16(
                        _mm_add_pi16(
                            _mm_add_pi16(
                                _mm_mullo_pi16(alpha_vec, p1_r_yvec2),
                                _mm_mullo_pi16(
                                    alpha_256_minus_vec,
                                    p2_r_yvec2)),
                            u16_512_128),
                        8);
                __m64 g_prime_yvec2 =
                    _mm_srli_pi16(
                        _mm_add_pi16(
                            _mm_add_pi16(
                                _mm_mullo_pi16(alpha_vec, p1_g_yvec2),
                                _mm_mullo_pi16(
                                    alpha_256_minus_vec,
                                    p2_g_yvec2)),
                            u16_512_128),
                        8);
                __m64 b_prime_yvec2 =
                    _mm_srli_pi16(
                        _mm_add_pi16(
                            _mm_add_pi16(
                                _mm_mullo_pi16(alpha_vec, p1_b_yvec2),
                                _mm_mullo_pi16(
                                    alpha_256_minus_vec,
                                    p2_b_yvec2)),
                            u16_512_128),
                        8);

                // ARGB -> YUV
                __m64 y_prime_yvec1 =
                    _mm_add_pi16(
                        _mm_srai_pi16(
                            _mm_add_pi16(
                                _mm_add_pi16(
                                    _mm_sub_pi16(_mm_mullo_pi16(u16_512_66, r_prime_yvec1), u16_512_8192),
                                    _mm_sub_pi16(_mm_mullo_pi16(u16_512_129, g_prime_yvec1), u16_512_16384)),
                                _mm_mullo_pi16(u16_512_25, b_prime_yvec1)),
                            8),
                        u16_512_112);

                __m64 y_prime_yvec2 =
                    _mm_add_pi16(
                        _mm_srai_pi16(
                            _mm_add_pi16(
                                _mm_add_pi16(
                                    _mm_sub_pi16(_mm_mullo_pi16(u16_512_66, r_prime_yvec2), u16_512_8192),
                                    _mm_sub_pi16(_mm_mullo_pi16(u16_512_129, g_prime_yvec2), u16_512_16384)),
                                _mm_mullo_pi16(u16_512_25, b_prime_yvec2)),
                            8),
                        u16_512_112);

                __m64 u_prime_yvec1 =
                    _mm_add_pi16(
                        _mm_srai_pi16(
                            _mm_add_pi16(
                                _mm_sub_pi16(
                                    _mm_mullo_pi16(u16_512_112, b_prime_yvec1),
                                    _mm_mullo_pi16(u16_512_38, r_prime_yvec1)),
                                _mm_sub_pi16(
                                    u16_512_128,
                                    _mm_mullo_pi16(u16_512_74, g_prime_yvec1))),
                            8),
                        u16_512_128);

                __m64 v_prime_yvec1 =
                    _mm_add_pi16(
                        _mm_srai_pi16(
                            _mm_add_pi16(
                                _mm_sub_pi16(
                                    _mm_mullo_pi16(u16_512_112, r_prime_yvec1),
                                    _mm_mullo_pi16(u16_512_94, g_prime_yvec1)),
                                _mm_sub_pi16(
                                    u16_512_128,
                                    _mm_mullo_pi16(u16_512_18, b_prime_yvec1))),
                            8),
                        u16_512_128);

                int y_prime_yvec1_packed = _mm_cvtsi64_si32(_mm_packs_pu16(y_prime_yvec1, _mm_setzero_si64()));
                int y_prime_yvec2_packed = _mm_cvtsi64_si32(_mm_packs_pu16(y_prime_yvec2, _mm_setzero_si64()));
                int u_prime_yvec1_packed = _mm_cvtsi64_si32(_mm_packs_pu16(u_prime_yvec1, _mm_setzero_si64()));
                int v_prime_yvec1_packed = _mm_cvtsi64_si32(_mm_packs_pu16(v_prime_yvec1, _mm_setzero_si64()));
                uint16_t u_prime_yvec1_extracted = (uint16_t)((u_prime_yvec1_packed >> 8) & 0xffff);
                uint16_t v_prime_yvec1_extracted = (uint16_t)((v_prime_yvec1_packed >> 8) & 0xffff);

                // Store value
                *(int *)((uint8_t *)y_result + image_idx * Y_SIZE + y_index_1) = y_prime_yvec1_packed;
                *(int *)((uint8_t *)y_result + image_idx * Y_SIZE + y_index_2) = y_prime_yvec2_packed;
                *(uint16_t *)((uint8_t *)u_result + image_idx * U_SIZE + uv_index) = u_prime_yvec1_extracted;
                *(uint16_t *)((uint8_t *)v_result + image_idx * V_SIZE + uv_index) = v_prime_yvec1_extracted;
            }
        }
    }
}

#undef VECTOR_SIZE