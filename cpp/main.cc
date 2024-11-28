#include <iostream>
#include <filesystem>
#include <fstream>
#include <ctime>
#include <malloc.h>
#include <cstring>
#include <cstdint>

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

int main(int argc, char *argv[])
{
    // test();
    // return 0;
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <yuv_path> <save_dir>" << std::endl;
        return EXIT_FAILURE;
    }
    std::string option = std::string(argv[1]);
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

    fs::path yuv_path = argv[2];
    fs::path save_dir = argv[3];

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
        break;
    case Option::Sse2:
        break;
    case Option::Avx2:
        break;
    case Option::Avx512:
        start_time = clock();
        solve_avx512(y_data, u_data, v_data, y_result, u_result, v_result);
        end_time = clock();
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
