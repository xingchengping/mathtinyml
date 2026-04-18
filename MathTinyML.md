MathTinyML

Running neural nets on potato-tier MCUs using pure math. No bloated frameworks, no <cmath>, no malloc().

I got sick of TensorFlow Lite for Microcontrollers eating up all my SRAM and clock cycles on ESP32/STM32 boards. FPU operations like exp() are way too slow for extreme-edge devices. So I threw out the standard approach and built an inference engine driven entirely by polynomial approximations and matrix factorization.

I'm 14, so I don't have time to write a massive codebase. This is a single-header library. Just drop it in and compile.

How it works (The Math)

1. Killing exp() with Chebyshev Approximation

Modern activations (GELU, Softmax) rely on $e^x$. Calling standard math libraries on an MCU for every parameter is a death sentence for performance. Taylor expansion is garbage the moment you move away from $x=0$.

Instead, I mapped the activation domain to $[-3, 3]$ and used first-kind Chebyshev Polynomials $T_n(x)$ to get a Minimax approximation.

$$P_k(x) = \sum_{i=0}^k c_i x^i \approx \text{GELU}(x)$$

Why this matters: The Chebyshev truncation error has a $2^n$ damping factor in the denominator.


$$E_n(x) \le \frac{1}{2^n (n+1)!} \max_{-1 \le \xi \le 1} |f^{(n+1)}(\xi)|$$

It guarantees a strictly bounded worst-case error. Evaluated using Horner's method, computing GELU now takes exactly 3 multiplications and 3 additions per value. Constant time $O(1)$. No branching. Pipeline stays happy.

2. Shattering the SRAM wall with SVD

A standard $768 \times 768$ weight matrix is ~2.3MB. It simply doesn't fit on an MCU.
Since neural nets are intrinsically low-rank, I apply Singular Value Decomposition (SVD) and aggressively truncate the singular values, keeping only the top $r$.

$$W \approx \hat{W} = U_r \Sigma_r V_r^T = A_{m \times r} B_{r \times n}$$

Forward pass goes from $O(m \times n)$ MACs to $O(r(m+n))$. For $r=32$, that's a 91.6% reduction in both memory and compute. By the Eckart-Young-Mirsky Theorem, the perturbation error is strictly bounded by the largest discarded singular value:


$$||y - \hat{y}||_2 \le \sigma_{r+1} ||x||_2$$

3. Zero-Allocation Memory

malloc() leads to fragmentation, which leads to your MCU crashing 3 days later.
This engine uses a static arena allocator. You give it a chunk of SRAM at boot, and all tensors are allocated via pointer bumping with strict 4-byte alignment. Garbage collection costs zero CPU cycles (just reset the offset to 0 after the forward pass).

Usage

Header-only. Include MathTinyML.hpp and you're good.

#include "MathTinyML.hpp"

// Allocate arena (e.g., 64KB)
uint8_t memory_pool[64 * 1024];
MathTinyML::ArenaAllocator arena(memory_pool, sizeof(memory_pool));

// Load factored weights from Flash
MathTinyML::Tensor A = {flash_ptr_A, 768, 32};
MathTinyML::Tensor B = {flash_ptr_B, 32, 768};
MathTinyML::Tensor x = {sensor_data, 768, 1};

// Setup output
MathTinyML::Tensor y = {nullptr, 768, 1};
y.data = (float*)arena.allocate(768 * sizeof(float));

// Forward + Activation
MathTinyML::low_rank_linear_forward(&arena, &A, &B, &x, &y);
MathTinyML::apply_activation(&y);

// Zero-cost GC
arena.reset();


Contributing

Feel free to open a PR if you can optimize the math further. But I have middle school homework, so PR reviews might take a while.

License: MIT.