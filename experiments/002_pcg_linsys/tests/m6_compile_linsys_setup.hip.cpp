#include "pcg/linsys_setup.hip.hpp"

template <typename T>
void instantiate_form_schur_system()
{
    const uint32_t state_size = 2;
    const uint32_t control_size = 1;
    const uint32_t knot_points = 3;

    T* d_G_dense = nullptr;
    T* d_C_dense = nullptr;
    T* d_g = nullptr;
    T* d_c = nullptr;
    T* d_S = nullptr;
    T* d_Pinv = nullptr;
    T* d_gamma = nullptr;

    T rho = static_cast<T>(1.0);

    form_schur_system<T>(
        state_size,
        control_size,
        knot_points,
        d_G_dense,
        d_C_dense,
        d_g,
        d_c,
        d_S,
        d_Pinv,
        d_gamma,
        rho
    );
}

int main()
{
    // Compile-only test.
    // Do not run this binary yet, because the pointers are intentionally null.
    return 0;
}

// Force template instantiation during compilation.
template void instantiate_form_schur_system<float>();
template void instantiate_form_schur_system<double>();
