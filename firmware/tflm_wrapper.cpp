#include "tflm_wrapper.h"
#include "mnist_cnn_int8_model_v1.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Arena de 120KB pra tensores intermediários da CNN
static constexpr int kTensorArenaSize = 120 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];  // alinhado em 16 bytes pra performance

static const tflite::Model* model_ptr = nullptr;
static tflite::MicroInterpreter* interpreter_ptr = nullptr;
static TfLiteTensor* input_ptr = nullptr;   // tensor de entrada [1, 28, 28, 1] int8
static TfLiteTensor* output_ptr = nullptr;  // tensor de saída [1, 10] int8

// Inicializa TFLM e carrega modelo da flash
extern "C" int tflm_init(void) {
    model_ptr = tflite::GetModel(mnist_cnn_int8_model);  // carrega modelo embarcado
    if (!model_ptr) return 1;
    if (model_ptr->version() != TFLITE_SCHEMA_VERSION) return 2;  // verifica compatibilidade
    
    // Registra apenas as operações usadas pelo modelo (economiza memória)
    static tflite::MicroMutableOpResolver<8> resolver;
    resolver.AddConv2D();           // camadas convolucionais
    resolver.AddMean();             // GlobalAveragePooling2D é implementado como MEAN
    resolver.AddFullyConnected();   // camada densa
    resolver.AddSoftmax();          // ativação final
    resolver.AddReshape();          // reshape entre camadas
    resolver.AddQuantize();         // operações de quantização
    resolver.AddDequantize();
    
    // Cria interpretador estático (evita alocação dinâmica)
    static tflite::MicroInterpreter static_interpreter(
        model_ptr, resolver, tensor_arena, kTensorArenaSize
    );
    interpreter_ptr = &static_interpreter;
    
    if (interpreter_ptr->AllocateTensors() != kTfLiteOk) return 3;  // aloca memória pros tensores
    
    input_ptr  = interpreter_ptr->input(0);   // pega referência do tensor de entrada
    output_ptr = interpreter_ptr->output(0);  // pega referência do tensor de saída
    if (!input_ptr || !output_ptr) return 4;
    
    // Valida que o modelo é realmente int8
    if (input_ptr->type != kTfLiteInt8)  return 5;
    if (output_ptr->type != kTfLiteInt8) return 6;
    
    return 0;  // sucesso
}

// Retorna ponteiro pro buffer de entrada int8[784]
extern "C" int8_t* tflm_input_ptr(int* nbytes) {
    if (!input_ptr) return nullptr;
    if (nbytes) *nbytes = input_ptr->bytes;  // 784 bytes
    return input_ptr->data.int8;
}

// Retorna ponteiro pro buffer de saída int8[10]
extern "C" int8_t* tflm_output_ptr(int* nbytes) {
    if (!output_ptr) return nullptr;
    if (nbytes) *nbytes = output_ptr->bytes;  // 10 bytes
    return output_ptr->data.int8;
}

// Retorna scale do tensor de entrada (usado na quantização)
extern "C" float tflm_input_scale(void) {
    return input_ptr ? input_ptr->params.scale : 0.0f;
}

// Retorna zero_point do tensor de entrada (usado na quantização)
extern "C" int tflm_input_zero_point(void) {
    return input_ptr ? input_ptr->params.zero_point : 0;
}

// Retorna scale do tensor de saída (usado na dequantização)
extern "C" float tflm_output_scale(void) {
    return output_ptr ? output_ptr->params.scale : 0.0f;
}

// Retorna zero_point do tensor de saída (usado na dequantização)
extern "C" int tflm_output_zero_point(void) {
    return output_ptr ? output_ptr->params.zero_point : 0;
}

// Executa inferência: processa input_ptr e gera resultado em output_ptr
extern "C" int tflm_invoke(void) {
    if (!interpreter_ptr) return 1;
    return (interpreter_ptr->Invoke() == kTfLiteOk) ? 0 : 2;
}

// Retorna quantos bytes da arena estão sendo usados (útil pra debug)
extern "C" int tflm_arena_used_bytes(void) {
    if (!interpreter_ptr) return -1;
    return (int)interpreter_ptr->arena_used_bytes();
}
