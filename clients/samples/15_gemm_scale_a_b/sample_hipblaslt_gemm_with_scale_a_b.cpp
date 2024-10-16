/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "helper.h"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/library_types.h>
#include <hipblaslt/hipblaslt.h>
#include <iostream>
#include <vector>

template <typename T, typename To>
class SMART_DESTROYER
{
    typedef To (*destroy_func)(T*);
public:
    SMART_DESTROYER(T* ptr, destroy_func func)
    {
        _ptr  = ptr;
        _func = func;
    }
    ~SMART_DESTROYER()
    {
        if(_ptr != nullptr)
            _func(_ptr);
    }
    T*           _ptr = nullptr;
    destroy_func _func;
};
template <typename T, typename To>
class SMART_DESTROYER_NON_PTR
{
    typedef To (*destroy_func)(T);
public:
    SMART_DESTROYER_NON_PTR(T* ptr, destroy_func func)
    {
        _ptr  = ptr;
        _func = func;
    }
    ~SMART_DESTROYER_NON_PTR()
    {
        if(_ptr != nullptr)
            _func(*_ptr);
    }
    T*           _ptr = nullptr;
    destroy_func _func;
};

hipError_t hipFreeWrapper(float* ptr)
{
    return hipFree(static_cast<void*>(ptr));
}

void simpleGemmScaleAB(hipblasLtHandle_t  handle,
                       hipblasOperation_t trans_a,
                       hipblasOperation_t trans_b,
                       int64_t            m,
                       int64_t            n,
                       int64_t            k,
                       int64_t            batch_count,
                       float&             alpha,
                       float&             beta,
                       void*              d_a,
                       void*              d_b,
                       void*              d_c,
                       void*              d_d,
                       void*              d_workspace,
                       int64_t            max_workspace_size,
                       hipStream_t        stream,
                       float              h_scale_a,
                       float              h_scale_b);

int main()
{
    Runner<hipblaslt_f8_fnuz, hipblaslt_f8_fnuz, hipblasLtHalf, float, float> runner(
        128, 128, 128, 1, 1.f, 0.f, 32 * 1024 * 1024);

    float scale_a = 0.5f; // scale A setting
    float scale_b = 2.0f; // scale B setting
    std::cout << "Running with Scale A = " << scale_a << " and Scale B = " << scale_b << std::endl;
    runner.run([&runner, scale_a, scale_b] {
        simpleGemmScaleAB(runner.handle,
                          HIPBLAS_OP_N,
                          HIPBLAS_OP_N,
                          runner.m,
                          runner.n,
                          runner.k,
                          runner.batch_count,
                          runner.alpha,
                          runner.beta,
                          runner.d_a,
                          runner.d_b,
                          runner.d_c,
                          runner.d_d,
                          runner.d_workspace,
                          runner.max_workspace_size,
                          runner.stream,
                          scale_a,
                          scale_b);
    });

    return 0;
}

void simpleGemmScaleAB(hipblasLtHandle_t  handle,
                       hipblasOperation_t trans_a,
                       hipblasOperation_t trans_b,
                       int64_t            m,
                       int64_t            n,
                       int64_t            k,
                       int64_t            batch_count,
                       float&             alpha,
                       float&             beta,
                       void*              d_a,
                       void*              d_b,
                       void*              d_c,
                       void*              d_d,
                       void*              d_workspace,
                       int64_t            max_workspace_size,
                       hipStream_t        stream,
                       float              h_scale_a,
                       float              h_scale_b)
{
    float* d_scale_a;
    float* d_scale_b;
    CHECK_HIP_ERROR(hipMalloc(&d_scale_a, sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_scale_b, sizeof(float)));

    SMART_DESTROYER<float, hipError_t> scaleA_destroyer(d_scale_a, hipFreeWrapper);
    SMART_DESTROYER<float, hipError_t> scaleB_destroyer(d_scale_b, hipFreeWrapper);

    CHECK_HIP_ERROR(
        hipMemcpyAsync(d_scale_a, &h_scale_a, sizeof(float), hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(
        hipMemcpyAsync(d_scale_b, &h_scale_b, sizeof(float), hipMemcpyHostToDevice, stream));

    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, HIP_R_8F_E4M3_FNUZ, m, k, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, HIP_R_8F_E4M3_FNUZ, k, n, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, HIP_R_16F, m, n, m));

    SMART_DESTROYER_NON_PTR<hipblasLtMatrixLayout_t, hipblasStatus_t> matA_destroyer(&matA, hipblasLtMatrixLayoutDestroy);
    SMART_DESTROYER_NON_PTR<hipblasLtMatrixLayout_t, hipblasStatus_t> matB_destroyer(&matB, hipblasLtMatrixLayoutDestroy);
    SMART_DESTROYER_NON_PTR<hipblasLtMatrixLayout_t, hipblasStatus_t> matC_destroyer(&matC, hipblasLtMatrixLayoutDestroy);
    SMART_DESTROYER_NON_PTR<hipblasLtMatrixLayout_t, hipblasStatus_t> matD_destroyer(&matD, hipblasLtMatrixLayoutDestroy);

    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));
    SMART_DESTROYER_NON_PTR<hipblasLtMatmulDesc_t, hipblasStatus_t> matmul_destroyer(&matmul, hipblasLtMatmulDescDestroy);

    // Set A and B matrix scale factors
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale_a, sizeof(float*)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scale_b, sizeof(float*)));

    hipblasLtMatmulPreference_t pref;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
    SMART_DESTROYER_NON_PTR<hipblasLtMatmulPreference_t, hipblasStatus_t> pref_destroyer(&pref, hipblasLtMatmulPreferenceDestroy);
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatmulPreferenceSetAttribute(pref,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &max_workspace_size,
                                              sizeof(max_workspace_size)));

    const int                        request_solutions = 5;
    hipblasLtMatmulHeuristicResult_t heuristicResult[request_solutions];
    int                              returnedAlgoCount = 0;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                          matmul,
                                                          matA,
                                                          matB,
                                                          matC,
                                                          matD,
                                                          pref,
                                                          request_solutions,
                                                          heuristicResult,
                                                          &returnedAlgoCount));

    if(returnedAlgoCount == 0)
    {
        std::cerr << "No valid solution found!" << std::endl;
        return;
    }

    uint64_t workspace_size = max_workspace_size;
    for(int i = 0; i < returnedAlgoCount; i++)
        workspace_size = std::max(workspace_size, heuristicResult[i].workspaceSize);

    // Perform matrix multiplication
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle,
                                          matmul,
                                          &alpha,
                                          d_a,
                                          matA,
                                          d_b,
                                          matB,
                                          &beta,
                                          d_c,
                                          matC,
                                          d_d,
                                          matD,
                                          &heuristicResult[0].algo,
                                          d_workspace,
                                          workspace_size,
                                          stream));


    std::cout << "Matrix multiplication completed successfully." << std::endl;
}
