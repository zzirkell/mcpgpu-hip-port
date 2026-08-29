#pragma once
#include <hip/hip_cooperative_groups.h>
namespace glass{
    /*      L1      */
    #include "./src/L1/axpy.hip.hpp"
    #include "./src/L1/copy.hip.hpp"
    #include "./src/L1/dot.hip.hpp"
    #include "./src/L1/ident.hip.hpp"
    #include "./src/L1/scal.hip.hpp"
    #include "./src/L1/swap.hip.hpp"

    /*      L2      */
    #include "./src/L2/gemv.hip.hpp"

    /*      L3      */
    #include "./src/L3/gemm.hip.hpp"
    #include "./src/L3/inv.hip.hpp"
}