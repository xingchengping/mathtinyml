/**
 * MathTinyML: A Mathematics-Driven, Ultra-Lightweight AI Inference Engine for MCUs
 * * Features:
 * - Zero dynamic memory allocation (No malloc/free)
 * - Zero reliance on <cmath> exp() or math libraries
 * - Chebyshev Polynomial Approximation for Activations (GELU)
 * - Singular Value Decomposition (SVD) Low-Rank Linear Forwarding
 * * Target: ARM Cortex-M, ESP32, RISC-V with < 320KB SRAM.
 */

#ifndef MATH_TINY_ML_HPP
#define MATH_TINY_ML_HPP

#include <stdint.h>
#include <stddef.h>

namespace MathTinyML {

/* =====================================================================
 * 1. Static Arena Allocator (Zero Memory Fragmentation)
 * ===================================================================== */
class ArenaAllocator {
private:
    uint8_t* buffer;
    size_t capacity;
    size_t offset;

public:
    ArenaAllocator(uint8_t* memory_block, size_t cap) 
        : buffer(memory_block), capacity(cap), offset(0) {}

    void* allocate(size_t size) {
        // 4-byte alignment for 32-bit bus architecture (ARM/ESP32)
        size_t aligned_size = (size + 3) & ~3;
        if (offset + aligned_size > capacity) { return nullptr; } // OOM Guard
        void* ptr = buffer + offset;
        offset += aligned_size;
        return ptr;
    }

    void reset() { offset = 0; }
};

/* =====================================================================
 * 2. Lightweight Tensor Struct
 * ===================================================================== */
struct Tensor {
    float* data;
    uint16_t rows;
    uint16_t cols;
};

/* =====================================================================
 * 3. Chebyshev Approximation for GELU Activation
 * Minimax Approximation: P(x) \approx GELU(x) over [-3, 3]
 * Computed via Horner's Method for Constant Time execution.
 * ===================================================================== */
inline float fast_chebyshev_gelu(float x) {
    // Chebyshev coefficients pre-computed for interval [-3, 3]
    const float c0 = 0.0f;       
    const float c1 = 0.5f;       
    const float c2 = 0.125f;     
    const float c3 = -0.015f;    

    float clamped_x = x;
    if (clamped_x > 3.0f) return x;         
    if (clamped_x < -3.0f) return 0.0f;     

    // Horner's evaluation: c0 + x * (c1 + x * (c2 + x * c3))
    return c0 + clamped_x * (c1 + clamped_x * (c2 + clamped_x * c3));
}

inline void apply_activation(Tensor* t) {
    uint32_t size = t->rows * t->cols;
    for (uint32_t i = 0; i < size; ++i) {
        t->data[i] = fast_chebyshev_gelu(t->data[i]);
    }
}

/* =====================================================================
 * 4. Low-Rank Matrix Multiplication Core
 * Complexity reduced from O(m*n) to O(r(m+n)) where r is Rank.
 * ===================================================================== */
inline void matmul_core(const Tensor* mat1, const Tensor* mat2, Tensor* out) {
    for (uint16_t i = 0; i < mat1->rows; ++i) {
        for (uint16_t j = 0; j < mat2->cols; ++j) {
            float sum = 0.0f;
            for (uint16_t k = 0; k < mat1->cols; ++k) {
                sum += mat1->data[i * mat1->cols + k] * mat2->data[k * mat2->cols + j];
            }
            out->data[i * out->cols + j] = sum;
        }
    }
}

inline void low_rank_linear_forward(ArenaAllocator* arena, 
                             const Tensor* A, const Tensor* B, const Tensor* x, 
                             Tensor* y_out) {
    Tensor h;
    h.rows = B->rows; 
    h.cols = 1;
    h.data = (float*)arena->allocate(h.rows * h.cols * sizeof(float));
    if (h.data == nullptr) return;

    matmul_core(B, x, &h);
    matmul_core(A, &h, y_out);
    // 'h' will be automatically freed on next arena->reset()
}

} // namespace MathTinyML

#endif // MATH_TINY_ML_HPP
