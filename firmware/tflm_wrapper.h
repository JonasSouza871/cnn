#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Retorna 0 se OK, !=0 se erro
int  tflm_init(void);

// Ponteiro para o buffer de entrada (INT8) e quantidade de bytes
int8_t*  tflm_input_ptr(int* nbytes);

// Ponteiro para o buffer de saída (INT8) e quantidade de bytes
int8_t*  tflm_output_ptr(int* nbytes);

// Quantização (scale e zero_point) de input/output
float tflm_input_scale(void);
int   tflm_input_zero_point(void);
float tflm_output_scale(void);
int   tflm_output_zero_point(void);

// Executa inferência: 0 OK, !=0 erro
int  tflm_invoke(void);

// Diagnóstico
int  tflm_arena_used_bytes(void);

#ifdef __cplusplus
}
#endif
