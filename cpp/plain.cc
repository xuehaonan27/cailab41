#include <cmath>
#include <cstdio>
#include <cstdint>

#include "misc.hh"

void solve_plain(
    const unsigned char *y_data,
    const unsigned char *u_data,
    const unsigned char *v_data,
    unsigned char **y_result,
    unsigned char **u_result,
    unsigned char **v_result)
{
    for (int image_idx = 0; image_idx < 84; image_idx++)
    {
        unsigned char alpha = 1 + image_idx * 3;
        float alpha_percentage = float(alpha) / 256;
        for (int j = 0; j < HEIGHT; j++)
        {
            for (int i = 0; i < WIDTH; i++)
            {
                size_t y_index = j * WIDTH + i;
                size_t uv_index = size_t(j / 2) * size_t(WIDTH / 2) + size_t(i / 2); // Cannot be changed

                unsigned char y = y_data[y_index];
                unsigned char u = u_data[uv_index];
                unsigned char v = v_data[uv_index];

                // YUV to ARGB
                float r = y + 1.140 * v;
                float g = y - 0.394 * u - 0.581 * v;
                float b = y + 2.032 * u;

                // Alpha mix
                float r2 = r * alpha_percentage;
                float g2 = g * alpha_percentage;
                float b2 = b * alpha_percentage;

                // RGB to YUV
                float y2 = 0.299 * r2 + 0.587 * g2 + 0.114 * b2;
                float u2 = -0.147 * r2 - 0.289 * g2 + 0.436 * b2;
                float v2 = 0.615 * r2 - 0.515 * g2 - 0.100 * b2;

                *(unsigned char *)((char *)y_result + image_idx * Y_SIZE + y_index) = (unsigned char)y2;
                *(unsigned char *)((char *)u_result + image_idx * U_SIZE + uv_index) = (unsigned char)u2;
                *(unsigned char *)((char *)v_result + image_idx * V_SIZE + uv_index) = (unsigned char)v2;
            }
        }
    }
}

// #define YUV2R_OPT(y, u, v) clamp((298 * (y) + 409 * (v) - 56992) >> 8)
// #define YUV2G_OPT(y, u, v) clamp((298 * (y) - 100 * (u) - 208 * (v) + 34784) >> 8)
// #define YUV2B_OPT(y, u, v) clamp((298 * (y) + 516 * (u) - 70688) >> 8)

static __inline__ int16_t
clamp(int16_t x)
{
    if (x > 255)
        return 255;
    if (x < 0)
        return 0;
    return x;
}

void solve_plain_int(
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
        for (int j = 0; j < HEIGHT; j++)
        {
            for (int i = 0; i < WIDTH; i++)
            {
                size_t y_index = j * WIDTH + i;
                size_t uv_index = size_t(j / 2) * size_t(WIDTH / 2) + size_t(i / 2);

                const uint8_t y = y_data[y_index];
                const uint8_t u = u_data[uv_index];
                const uint8_t v = v_data[uv_index];

                // YUV -> RGB
                // #define YUV2R(y, u, v) clamp((298 * ((y) - 16) + 409 * ((v) - 128) + 128) >> 8)
                // #define YUV2G(y, u, v) clamp((298 * ((y) - 16) - 100 * ((u) - 128) - 208 * ((v) - 128) + 128) >> 8)
                // #define YUV2B(y, u, v) clamp((298 * ((y) - 16) + 516 * ((u) - 128) + 128) >> 8)
                // #define YUV2R(y, u, v) clamp(((42 * ((y) - 16) + 153 * ((v) - 128) + 128) >> 8) + ((y) - 16) + ((v) - 128))
                // #define YUV2G(y, u, v) clamp((((42 * ((y) - 16) - 100 * ((u) - 128)) + (128 - 208 * ((v) - 128))) >> 8) + ((y) - 16))
                // #define YUV2B(y, u, v) clamp(((42 * ((y) - 16) + 4 * ((u) - 128) + 128) >> 8) + ((y) - 16) + 2 * ((u) - 128))
                // #define YUV2R(y, u, v) clamp((298 * (y) + 409 * (v) - 56992) >> 8)
                // #define YUV2G(y, u, v) clamp((298 * (y) - 100 * (u) - 208 * (v) + 34784) >> 8)
                // #define YUV2B(y, u, v) clamp((298 * (y) + 516 * (u) - 70688) >> 8)

#define YUV2R(y, u, v) clamp(((42 * (y) - 24224 + 153 * (v)) >> 8) + ((y) - 128) + (v))
#define YUV2G(y, u, v) clamp((((42 * (y) - 100 * (u)) + (2016 - 208 * (v))) >> 8) + ((y) + 128))
#define YUV2B(y, u, v) clamp(((42 * (y) + 4 * (u) - 5152) >> 8) + ((y) - 256) + 2 * (u))

                const uint8_t r = YUV2R(y, u, v) & 0xff;
                const uint8_t g = YUV2G(y, u, v) & 0xff;
                const uint8_t b = YUV2B(y, u, v) & 0xff;

                // Alpha mix
                const uint16_t r2 = alpha * r >> 8;
                const uint16_t g2 = alpha * g >> 8;
                const uint16_t b2 = alpha * b >> 8;

// RGB -> YUV
// #define RGB2Y(r, g, b) (uint8_t)(((66 * (r) + 129 * (g) + 25 * (b) + 128) >> 8) + 16)
// #define RGB2U(r, g, b) (uint8_t)(((-38 * (r) - 74 * (g) + 112 * (b) + 128) >> 8) + 128)
// #define RGB2V(r, g, b) (uint8_t)(((112 * (r) - 94 * (g) - 18 * (b) + 128) >> 8) + 128)
#define RGB2Y(r, g, b) (uint8_t)((((66 * (r) - 8192) + (129 * (g) - 16384) + (25 * (b) - 1920)) >> 8) + 120)
#define RGB2U(r, g, b) (uint8_t)((((112 * (b) - 38 * (r)) + (128 - 74 * (g))) >> 8) + 128)
#define RGB2V(r, g, b) (uint8_t)((((112 * (r) - 94 * (g)) + (128 - 18 * (b))) >> 8) + 128)
                uint8_t y2 = RGB2Y(r2, g2, b2);
                uint8_t u2 = RGB2U(r2, g2, b2);
                uint8_t v2 = RGB2V(r2, g2, b2);

                *(uint8_t *)((uint8_t *)y_result + image_idx * Y_SIZE + y_index) = y2;
                *(uint8_t *)((uint8_t *)u_result + image_idx * U_SIZE + uv_index) = u2;
                *(uint8_t *)((uint8_t *)v_result + image_idx * V_SIZE + uv_index) = v2;
            }
        }
    }
}

void solve_plain_part3(
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
        for (int j = 0; j < HEIGHT; j++)
        {
            for (int i = 0; i < WIDTH; i++)
            {
                size_t y_index = j * WIDTH + i;
                size_t uv_index = size_t(j / 2) * size_t(WIDTH / 2) + size_t(i / 2);

                const uint8_t p1_y = p1_y_data[y_index];
                const uint8_t p1_u = p1_u_data[uv_index];
                const uint8_t p1_v = p1_v_data[uv_index];

                const uint8_t p2_y = p2_y_data[y_index];
                const uint8_t p2_u = p2_u_data[uv_index];
                const uint8_t p2_v = p2_v_data[uv_index];

                // #define YUV2R(y, u, v) clamp((298 * ((y) - 16) + 409 * ((v) - 128) + 128) >> 8)
                // #define YUV2G(y, u, v) clamp((298 * ((y) - 16) - 100 * ((u) - 128) - 208 * ((v) - 128) + 128) >> 8)
                // #define YUV2B(y, u, v) clamp((298 * ((y) - 16) + 516 * ((u) - 128) + 128) >> 8)
                const uint8_t p1_r = YUV2R(p1_y, p1_u, p1_v) & 0xff;
                const uint8_t p1_g = YUV2G(p1_y, p1_u, p1_v) & 0xff;
                const uint8_t p1_b = YUV2B(p1_y, p1_u, p1_v) & 0xff;

                const uint8_t p2_r = YUV2R(p2_y, p2_u, p2_v) & 0xff;
                const uint8_t p2_g = YUV2G(p2_y, p2_u, p2_v) & 0xff;
                const uint8_t p2_b = YUV2B(p2_y, p2_u, p2_v) & 0xff;

                // image overlap
// #define MIX(a, x1, x2) ((((a) * (x1)) + (256 - (a)) * (x2)) >> 8)
// #define MIX(a, x1, x2) ((((a) * ((x1) - (x2))) >> 8) + x2)
#define MIX(a, x1, x2) (((((a) * (x1)) + ((256 - (a)) * (x2))) + 128) >> 8)
                const uint16_t r_prime = MIX(alpha, p1_r, p2_r);
                const uint16_t g_prime = MIX(alpha, p1_g, p2_g);
                const uint16_t b_prime = MIX(alpha, p1_b, p2_b);

                // RGB -> YUV
                // #define RGB2Y(r, g, b) (uint8_t)(((66 * (r) + 129 * (g) + 25 * (b) + 128) >> 8) + 16)
                // #define RGB2U(r, g, b) (uint8_t)(((-38 * (r) - 74 * (g) + 112 * (b) + 128) >> 8) + 128)
                // #define RGB2V(r, g, b) (uint8_t)(((112 * (r) - 94 * (g) - 18 * (b) + 128) >> 8) + 128)
                uint8_t y_prime = RGB2Y(r_prime, g_prime, b_prime);
                uint8_t u_prime = RGB2U(r_prime, g_prime, b_prime);
                uint8_t v_prime = RGB2V(r_prime, g_prime, b_prime);

                *(uint8_t *)((uint8_t *)y_result + image_idx * Y_SIZE + y_index) = y_prime;
                *(uint8_t *)((uint8_t *)u_result + image_idx * U_SIZE + uv_index) = u_prime;
                *(uint8_t *)((uint8_t *)v_result + image_idx * V_SIZE + uv_index) = v_prime;
            }
        }
    }
}