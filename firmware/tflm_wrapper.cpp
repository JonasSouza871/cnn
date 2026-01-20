#include "tflm_wrapper.h"

#include "mnist_cnn_int8_model_v1.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
//#include "tensorflow/lite/version.h"

// Arena (ajuste se precisar)
static constexpr int kTensorArenaSize = 120 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

static const tflite::Model* model_ptr = nullptr;
static tflite::MicroInterpreter* interpreter_ptr = nullptr;
static TfLiteTensor* input_ptr = nullptr;
static TfLiteTensor* output_ptr = nullptr;

// Inicialização do TFLM
extern "C" int tflm_init(void) {
    model_ptr = tflite::GetModel(mnist_cnn_int8_model);
    if (!model_ptr) return 1;
    if (model_ptr->version() != TFLITE_SCHEMA_VERSION) return 2;

    // Resolver mínimo (para a CNN proposta)
    static tflite::MicroMutableOpResolver<8> resolver;
    resolver.AddConv2D();
    resolver.AddMean();            // GlobalAveragePooling2D -> MEAN
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddQuantize();
    resolver.AddDequantize();

    // Interpreter estático (evita new em alguns cenários)
    static tflite::MicroInterpreter static_interpreter(
        model_ptr, resolver, tensor_arena, kTensorArenaSize
    );
    interpreter_ptr = &static_interpreter;

    if (interpreter_ptr->AllocateTensors() != kTfLiteOk) return 3;

    input_ptr  = interpreter_ptr->input(0);
    output_ptr = interpreter_ptr->output(0);
    if (!input_ptr || !output_ptr) return 4;

    // Esperado: int8
    if (input_ptr->type != kTfLiteInt8)  return 5;
    if (output_ptr->type != kTfLiteInt8) return 6;

    return 0;
}

extern "C" int8_t* tflm_input_ptr(int* nbytes) {
    if (!input_ptr) return nullptr;
    if (nbytes) *nbytes = input_ptr->bytes;
    return input_ptr->data.int8;
}

extern "C" int8_t* tflm_output_ptr(int* nbytes) {
    if (!output_ptr) return nullptr;
    if (nbytes) *nbytes = output_ptr->bytes;
    return output_ptr->data.int8;
}

extern "C" float tflm_input_scale(void) {
    return input_ptr ? input_ptr->params.scale : 0.0f;
}
extern "C" int tflm_input_zero_point(void) {
    return input_ptr ? input_ptr->params.zero_point : 0;
}
extern "C" float tflm_output_scale(void) {
    return output_ptr ? output_ptr->params.scale : 0.0f;
}
extern "C" int tflm_output_zero_point(void) {
    return output_ptr ? output_ptr->params.zero_point : 0;
}

extern "C" int tflm_invoke(void) {
    if (!interpreter_ptr) return 1;
    return (interpreter_ptr->Invoke() == kTfLiteOk) ? 0 : 2;
}

extern "C" int tflm_arena_used_bytes(void) {
    if (!interpreter_ptr) return -1;
    return (int)interpreter_ptr->arena_used_bytes();
}