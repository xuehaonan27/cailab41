#include <iostream>
#include <filesystem>
#include <fstream>
#include <ctime>
#include <malloc.h>
#include <cstring>
#include <cstdint>
#include <cstdio>

#include "plain.hh"
#include "mmx.hh"
#include "sse2.hh"
#include "avx.hh"

#include "misc.hh"
namespace fs = std::filesystem;

enum Option
{
    Plain,
    Mmx,
    Sse2,
    Avx2,
    Avx512,
};

int solve_part2(Option opt, fs::path &yuv_path, fs::path &save_dir)
{
    std::cout << "YUV file: " << yuv_path << std::endl;
    std::cout << "Save to directory: " << save_dir << std::endl;

    if (!fs::exists(save_dir))
    {
        fs::create_directories(save_dir);
    }

    std::ifstream yuv_file(yuv_path, std::ios::binary);
    if (!yuv_file)
    {
        std::cerr << "Cannot open " << yuv_path << std::endl;
        return EXIT_FAILURE;
    }

    /* Picture read */

    uint8_t *y_data = (uint8_t *)malloc(Y_SIZE);
    uint8_t *u_data = (uint8_t *)malloc(U_SIZE);
    uint8_t *v_data = (uint8_t *)malloc(V_SIZE);

    // Read fields
    yuv_file.read(reinterpret_cast<char *>(y_data), Y_SIZE);
    yuv_file.read(reinterpret_cast<char *>(u_data), U_SIZE);
    yuv_file.read(reinterpret_cast<char *>(v_data), V_SIZE);
    yuv_file.close();

    uint8_t **y_result = (uint8_t **)malloc(84 * Y_SIZE);
    uint8_t **u_result = (uint8_t **)malloc(84 * U_SIZE);
    uint8_t **v_result = (uint8_t **)malloc(84 * V_SIZE);

    clock_t start_time, end_time;

    switch (opt)
    {
    case Option::Plain:
        start_time = clock();
        solve_plain_int(y_data, u_data, v_data, y_result, u_result, v_result);
        end_time = clock();
        break;
    case Option::Mmx:
        start_time = clock();
        solve_mmx_part2(y_data, u_data, v_data, y_result, u_result, v_result);
        end_time = clock();
        break;
    case Option::Sse2:
        start_time = clock();
        solve_sse2_part2(y_data, u_data, v_data, y_result, u_result, v_result);
        end_time = clock();
        break;
    case Option::Avx2:
        break;
    case Option::Avx512:
        start_time = clock();
        solve_avx512(y_data, u_data, v_data, y_result, u_result, v_result);
        // solve_avx512_loop_unfold(y_data, u_data, v_data, y_result, u_result, v_result);
        end_time = clock();
        break;
    default:
        break;
    }

    double elapsed_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "Elapsed Time: " << elapsed_time << " seconds" << std::endl;

    for (int i = 0; i < 84; i++)
    {
        fs::path output_path = save_dir / (std::to_string(i) + ".yuv");
        std::ofstream output_file(output_path, std::ios::binary);
        output_file.write(reinterpret_cast<char *>((char *)y_result + i * Y_SIZE), Y_SIZE);
        output_file.write(reinterpret_cast<char *>((char *)u_result + i * U_SIZE), U_SIZE);
        output_file.write(reinterpret_cast<char *>((char *)v_result + i * V_SIZE), V_SIZE);
        output_file.close();
    }

    std::cout << "Files saved" << std::endl;

    free(y_data);
    free(u_data);
    free(v_data);
    free(y_result);
    free(u_result);
    free(v_result);

    return EXIT_SUCCESS;
}

int solve_part3(Option opt, fs::path &yuv_path_1, fs::path &yuv_path_2, fs::path &save_dir)
{
    std::cout << "YUV file: " << yuv_path_1 << ' ' << yuv_path_2 << std::endl;
    std::cout << "Save to directory" << save_dir << std::endl;

    if (!fs::exists(save_dir))
    {
        fs::create_directories(save_dir);
    }

    std::ifstream yuv_file_1(yuv_path_1, std::ios::binary);
    if (!yuv_file_1)
    {
        std::cerr << "Cannot open " << yuv_path_1 << std::endl;
        return EXIT_FAILURE;
    }

    std::ifstream yuv_file_2(yuv_path_2, std::ios::binary);
    if (!yuv_file_2)
    {
        std::cerr << "Cannot open " << yuv_path_2 << std::endl;
        return EXIT_FAILURE;
    }

    /* Picture read */
    uint8_t *p1_y_data = (uint8_t *)malloc(Y_SIZE);
    uint8_t *p1_u_data = (uint8_t *)malloc(U_SIZE);
    uint8_t *p1_v_data = (uint8_t *)malloc(V_SIZE);

    uint8_t *p2_y_data = (uint8_t *)malloc(Y_SIZE);
    uint8_t *p2_u_data = (uint8_t *)malloc(U_SIZE);
    uint8_t *p2_v_data = (uint8_t *)malloc(V_SIZE);

    // Read fields
    yuv_file_1.read(reinterpret_cast<char *>(p1_y_data), Y_SIZE);
    yuv_file_1.read(reinterpret_cast<char *>(p1_u_data), U_SIZE);
    yuv_file_1.read(reinterpret_cast<char *>(p1_v_data), V_SIZE);
    yuv_file_1.close();
    yuv_file_2.read(reinterpret_cast<char *>(p2_y_data), Y_SIZE);
    yuv_file_2.read(reinterpret_cast<char *>(p2_u_data), U_SIZE);
    yuv_file_2.read(reinterpret_cast<char *>(p2_v_data), V_SIZE);
    yuv_file_2.close();

    uint8_t **y_result = (uint8_t **)malloc(84 * Y_SIZE);
    uint8_t **u_result = (uint8_t **)malloc(84 * U_SIZE);
    uint8_t **v_result = (uint8_t **)malloc(84 * V_SIZE);

    clock_t start_time, end_time;

    switch (opt)
    {
    case Option::Plain:
        start_time = clock();
        solve_plain_part3(
            p1_y_data, p1_u_data, p1_v_data,
            p2_y_data, p2_u_data, p2_v_data,
            y_result, u_result, v_result);
        end_time = clock();
        break;
    case Option::Mmx:
        break;
    case Option::Sse2:
        start_time = clock();
        solve_sse2_part3(
            p1_y_data, p1_u_data, p1_v_data,
            p2_y_data, p2_u_data, p2_v_data,
            y_result, u_result, v_result);
        end_time = clock();
        break;
    case Option::Avx2:
        break;
    case Option::Avx512:
        start_time = clock();
        solve_avx512_part3(
            p1_y_data, p1_u_data, p1_v_data,
            p2_y_data, p2_u_data, p2_v_data,
            y_result, u_result, v_result);
        end_time = clock();
        break;
    default:
        break;
    }

    double elapsed_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "Elapsed Time: " << elapsed_time << " seconds" << std::endl;

    for (int i = 0; i < 84; i++)
    {
        fs::path output_path = save_dir / (std::to_string(i) + ".yuv");
        std::ofstream output_file(output_path, std::ios::binary);
        output_file.write(reinterpret_cast<char *>((char *)y_result + i * Y_SIZE), Y_SIZE);
        output_file.write(reinterpret_cast<char *>((char *)u_result + i * U_SIZE), U_SIZE);
        output_file.write(reinterpret_cast<char *>((char *)v_result + i * V_SIZE), V_SIZE);
        output_file.close();
    }

    std::cout << "Files saved" << std::endl;

    free(p1_y_data);
    free(p1_u_data);
    free(p1_v_data);
    free(p2_y_data);
    free(p2_u_data);
    free(p2_v_data);
    free(y_result);
    free(u_result);
    free(v_result);
    return EXIT_SUCCESS;
}

int solve_test(fs::path &yuv_path_1, fs::path &yuv_path_2, fs::path &save_path)
{
    std::cout << "YUV file: " << yuv_path_1 << ' ' << yuv_path_2 << std::endl;
    std::cout << "Test result to " << save_path << std::endl;

    FILE *output_file = fopen(save_path.c_str(), "w");
    if (!output_file)
    {
        std::cerr << "Fail to open the output file" << std::endl;
    }

    std::ifstream yuv_file_1(yuv_path_1, std::ios::binary);
    if (!yuv_file_1)
    {
        std::cerr << "Cannot open " << yuv_path_1 << std::endl;
        return EXIT_FAILURE;
    }

    std::ifstream yuv_file_2(yuv_path_2, std::ios::binary);
    if (!yuv_file_2)
    {
        std::cerr << "Cannot open " << yuv_path_2 << std::endl;
        return EXIT_FAILURE;
    }

    /* Picture read */
    uint8_t *p1_y_data = (uint8_t *)malloc(Y_SIZE);
    uint8_t *p1_u_data = (uint8_t *)malloc(U_SIZE);
    uint8_t *p1_v_data = (uint8_t *)malloc(V_SIZE);

    uint8_t *p2_y_data = (uint8_t *)malloc(Y_SIZE);
    uint8_t *p2_u_data = (uint8_t *)malloc(U_SIZE);
    uint8_t *p2_v_data = (uint8_t *)malloc(V_SIZE);

    // Read fields
    yuv_file_1.read(reinterpret_cast<char *>(p1_y_data), Y_SIZE);
    yuv_file_1.read(reinterpret_cast<char *>(p1_u_data), U_SIZE);
    yuv_file_1.read(reinterpret_cast<char *>(p1_v_data), V_SIZE);
    yuv_file_1.close();
    yuv_file_2.read(reinterpret_cast<char *>(p2_y_data), Y_SIZE);
    yuv_file_2.read(reinterpret_cast<char *>(p2_u_data), U_SIZE);
    yuv_file_2.read(reinterpret_cast<char *>(p2_v_data), V_SIZE);
    yuv_file_2.close();

    uint8_t **y_result_plain = (uint8_t **)malloc(84 * Y_SIZE);
    uint8_t **u_result_plain = (uint8_t **)malloc(84 * U_SIZE);
    uint8_t **v_result_plain = (uint8_t **)malloc(84 * V_SIZE);

    uint8_t **y_result_avx512 = (uint8_t **)malloc(84 * Y_SIZE);
    uint8_t **u_result_avx512 = (uint8_t **)malloc(84 * U_SIZE);
    uint8_t **v_result_avx512 = (uint8_t **)malloc(84 * V_SIZE);

    solve_plain_part3(
        p1_y_data, p1_u_data, p1_v_data,
        p2_y_data, p2_u_data, p2_v_data,
        y_result_plain, u_result_plain, v_result_plain);

    solve_avx512_part3(
        p1_y_data, p1_u_data, p1_v_data,
        p2_y_data, p2_u_data, p2_v_data,
        y_result_avx512, u_result_avx512, v_result_avx512);

    // Compare all these bytes

    // Y result
    for (int i = 0; i < 84 * Y_SIZE; i++)
    {
        uint8_t plain_byte = *((uint8_t *)y_result_plain + i);
        uint8_t avx512_byte = *((uint8_t *)y_result_avx512 + i);
        if (plain_byte != avx512_byte)
        {
            fprintf(output_file, "Y[%d] plain=%02x avx512=%02x\n", i, plain_byte, avx512_byte);
        }
    }

    // U result
    for (int i = 0; i < 84 * U_SIZE; i++)
    {
        uint8_t plain_byte = *((uint8_t *)u_result_plain + i);
        uint8_t avx512_byte = *((uint8_t *)u_result_avx512 + i);
        if (plain_byte != avx512_byte)
        {
            fprintf(output_file, "U[%d] plain=%02x avx512=%02x\n", i, plain_byte, avx512_byte);
        }
    }

    // V result
    for (int i = 0; i < 84 * V_SIZE; i++)
    {
        uint8_t plain_byte = *((uint8_t *)v_result_plain + i);
        uint8_t avx512_byte = *((uint8_t *)v_result_avx512 + i);
        if (plain_byte != avx512_byte)
        {
            fprintf(output_file, "V[%d] plain=%02x avx512=%02x\n", i, plain_byte, avx512_byte);
        }
    }

    fclose(output_file);

    free(p1_y_data);
    free(p1_u_data);
    free(p1_v_data);
    free(p2_y_data);
    free(p2_u_data);
    free(p2_v_data);
    free(y_result_plain);
    free(u_result_plain);
    free(v_result_plain);
    free(y_result_avx512);
    free(u_result_avx512);
    free(v_result_avx512);
    return EXIT_SUCCESS;
}

int main(int argc, char *argv[])
{
    // test();
    // return 0;
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <yuv_path> <save_dir>" << std::endl;
        return EXIT_FAILURE;
    }
    std::string part = std::string(argv[1]);
    std::string option = std::string(argv[2]);
    Option opt;
    if (option == "plain")
    {
        opt = Option::Plain;
    }
    else if (option == "mmx")
    {
        opt = Option::Mmx;
    }
    else if (option == "sse2")
    {
        opt = Option::Sse2;
    }
    else if (option == "avx2")
    {
        opt = Option::Avx2;
    }
    else if (option == "avx512")
    {
        opt = Option::Avx512;
    }
    else
    {
        std::cerr << "Invalid option: " << option << std::endl;
        return EXIT_FAILURE;
    }

    if (part == "part2")
    {
        fs::path yuv_path = argv[3];
        fs::path save_dir = argv[4];
        return solve_part2(opt, yuv_path, save_dir);
    }
    else if (part == "part3")
    {
        fs::path yuv_path_1 = argv[3];
        fs::path yuv_path_2 = argv[4];
        fs::path save_dir = argv[5];
        return solve_part3(opt, yuv_path_1, yuv_path_2, save_dir);
    }
    else if (part == "test")
    {
        fs::path yuv_path_1 = argv[3];
        fs::path yuv_path_2 = argv[4];
        fs::path save_path = argv[5];
        return solve_test(yuv_path_1, yuv_path_2, save_path);
    }
    else
    {
        std::cerr << "Invalid part: " << part << ", available: part2 part3" << std::endl;
        return EXIT_FAILURE;
    }
}
