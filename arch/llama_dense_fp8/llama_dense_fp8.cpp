/* llama_dense_fp8 -- arch/llama_fp8 over a bf16 checkpoint quantised by rad-convert.
 *
 * For MiniCPM5-2B, which OpenBMB publishes only in bf16. The recipe
 * data/recipes/minicpm5-2b-iq8r.recipe stores every linear as rotated int8 with a scale per
 * (row, 256 of K); data/recipes/minicpm5-2b-bf16-dspark.recipe keeps the model's own linears bf16
 * and quantises only the merged drafter.
 *
 * A separate .so because the selection key is the (architecture, quantisation) pair (spec §2.4),
 * and a dense checkpoint's descriptor is empty. Everything else is arch/llama_fp8/llama_fp8.cpp;
 * `#if LLAMA_DENSE_SRC` there changes only the descriptor check and the graph dump text.
 */
#define LLAMA_DENSE_SRC 1
#define LLAMA_NS        llama_dense_fp8
#define LLAMA_QUANT     ""

#include "../llama_fp8/llama_fp8.cpp"
