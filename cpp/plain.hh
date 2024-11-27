#ifndef PLAIN_HH
#define PLAIN_HH

void solve_plain(
    const unsigned char *y_data,
    const unsigned char *u_data,
    const unsigned char *v_data,
    unsigned char **y_result,
    unsigned char **u_result,
    unsigned char **v_result);

void solve_plain_int(
    const unsigned char *y_data,
    const unsigned char *u_data,
    const unsigned char *v_data,
    unsigned char **y_result,
    unsigned char **u_result,
    unsigned char **v_result);

#endif // PLAIN_HH