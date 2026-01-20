#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int tflm_init(void);  // Inicializa TFLM e carrega modelo, retorna 0 se OK

int8_t* tflm_input_ptr(int* nbytes); // Ponteiro pro buffer de entrada int8[784]
int8_t* tflm_output_ptr(int* nbytes);  // Ponteiro pro buffer de saída int8[10]

float tflm_input_scale(void); // Scale do tensor de entrada
int tflm_input_zero_point(void); // Zero point do tensor de entrada
float tflm_output_scale(void);// Scale do tensor de saída
int tflm_output_zero_point(void); // Zero point do tensor de saída

int tflm_invoke(void); // Executa inferência, retorna 0 se OK
int tflm_arena_used_bytes(void);  // Retorna bytes usados da arena (debug)

#ifdef __cplusplus
}
#endif
