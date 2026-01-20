/** 
 * @file cnn_mnist_probs.c
 * @brief Inferência CNN MNIST INT8 com PROBABILIDADES em % no display e serial
 */

#include "pico/stdlib.h"
#include "hardware/i2c.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "tflm_wrapper.h"
#include "ssd1306.h"
#include "font.h"

// ============================================================================
// CONFIGURAÇÕES
// ============================================================================
#define MNIST_SIZE 784
#define CSV_BUFFER_SIZE 8192

// ============================================================================
// GLOBAIS
// ============================================================================
ssd1306_t display;
static char csv_buffer[CSV_BUFFER_SIZE];
static int csv_pos = 0;
static absolute_time_t last_byte_time;

// ============================================================================
// FUNÇÕES AUXILIARES
// ============================================================================

static int argmax_i8(const int8_t* v, int n) {
    int best = 0;
    int8_t bestv = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > bestv) {
            bestv = v[i];
            best = i;
        }
    }
    return best;
}

static int8_t quantize_f32_to_i8(float x, float scale, int zp) {
    float q = (x / scale) + (float)zp;
    if (q > 127.0f) q = 127.0f;
    if (q < -128.0f) q = -128.0f;
    return (int8_t)q;
}

// ============================================================================
// CONVERSÃO INT8 -> PROBABILIDADE
// ============================================================================
void softmax_i8_to_probs(const int8_t* logits, float scale, int zero_point, float* probs, int n) {
    // Dequantizar: x = scale * (q - zero_point)
    float dequant[10];
    float max_val = -1e9f;
    
    for (int i = 0; i < n; i++) {
        dequant[i] = scale * ((float)logits[i] - (float)zero_point);
        if (dequant[i] > max_val) max_val = dequant[i];
    }
    
    // Softmax estável: exp(x - max) / sum(exp(x - max))
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        dequant[i] = expf(dequant[i] - max_val);
        sum += dequant[i];
    }
    
    for (int i = 0; i < n; i++) {
        probs[i] = (dequant[i] / sum) * 100.0f; // Em porcentagem
    }
}

// ============================================================================
// PARSER CSV
// ============================================================================
static int parse_csv_line(const char* line, uint8_t* label, uint8_t* pixels) {
    const char* ptr = line;
    char num_buf[8];
    int num_pos = 0;
    int field_count = 0;
    
    // Ignorar linhas vazias ou comentários
    while (*ptr == ' ' || *ptr == '\t') ptr++;
    if (*ptr == '\0' || *ptr == '\n' || *ptr == '#') return -1;
    
    while (*ptr != '\0' && *ptr != '\n' && field_count <= MNIST_SIZE) {
        // Ler número
        num_pos = 0;
        while (*ptr >= '0' && *ptr <= '9' && num_pos < sizeof(num_buf)-1) {
            num_buf[num_pos++] = *ptr++;
        }
        num_buf[num_pos] = '\0';
        
        if (num_pos > 0) {
            int value = atoi(num_buf);
            
            if (field_count == 0) {
                *label = (uint8_t)(value & 0xFF);
            } else if (field_count <= MNIST_SIZE) {
                if (value < 0) value = 0;
                if (value > 255) value = 255;
                pixels[field_count - 1] = (uint8_t)value;
            }
            
            field_count++;
        }
        
        while (*ptr == ',' || *ptr == ' ' || *ptr == '\t') ptr++;
    }
    
    return (field_count == 785) ? 0 : -1;
}

// ============================================================================
// DISPLAY COM PROBABILIDADES
// ============================================================================
void show_results(const float* probs, uint8_t true_label) {
    typedef struct {
        int digit;
        float prob;
    } prediction_t;
    
    prediction_t preds[10];
    for (int i = 0; i < 10; i++) {
        preds[i].digit = i;
        preds[i].prob = probs[i];
    }
    
    // Sort descending
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9 - i; j++) {
            if (preds[j].prob < preds[j+1].prob) {
                prediction_t temp = preds[j];
                preds[j] = preds[j+1];
                preds[j+1] = temp;
            }
        }
    }
    
    ssd1306_fill(&display, false);
    
    char line[24];
    snprintf(line, sizeof(line), "REAL: %d", true_label);
    ssd1306_draw_string(&display, line, 0, 0, false);
    
    snprintf(line, sizeof(line), "1:%d %.1f%%", preds[0].digit, preds[0].prob);
    ssd1306_draw_string(&display, line, 0, 14, false);
    
    snprintf(line, sizeof(line), "2:%d %.1f%%", preds[1].digit, preds[1].prob);
    ssd1306_draw_string(&display, line, 0, 26, false);
    
    snprintf(line, sizeof(line), "3:%d %.1f%%", preds[2].digit, preds[2].prob);
    ssd1306_draw_string(&display, line, 0, 38, false);
    
    bool correct = (preds[0].digit == true_label);
    snprintf(line, sizeof(line), "PRED:%d %s", preds[0].digit, correct ? "OK!" : "ERR");
    ssd1306_draw_string(&display, line, 0, 52, false);
    
    ssd1306_send_data(&display);
}

// ============================================================================
// INFERÊNCIA
// ============================================================================
static void run_inference(uint8_t label, const uint8_t* pixels) {
    static int8_t *in = NULL, *out = NULL;
    static float in_scale, out_scale;
    static int in_zp, out_zp;
    static bool initialized = false;
    
    if (!initialized) {
        int in_bytes, out_bytes;
        in = tflm_input_ptr(&in_bytes);
        out = tflm_output_ptr(&out_bytes);
        in_scale = tflm_input_scale();
        in_zp = tflm_input_zero_point();
        out_scale = tflm_output_scale();
        out_zp = tflm_output_zero_point();
        
        printf("\n=== TFLM CONFIG ===\n");
        printf("Input: scale=%.6f, zp=%d\n", in_scale, in_zp);
        printf("Output: scale=%.6f, zp=%d\n", out_scale, out_zp);
        initialized = true;
    }
    
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║         NOVA INFERÊNCIA               ║\n");
    printf("╚═══════════════════════════════════════╝\n");
    printf("Label real: %d\n", label);
    printf("Primeiros 5 pixels: %d,%d,%d,%d,%d\n", 
           pixels[0], pixels[1], pixels[2], pixels[3], pixels[4]);
    
    // Quantizar entrada
    for (int i = 0; i < MNIST_SIZE; i++) {
        float normalized = (float)pixels[i] / 255.0f;
        in[i] = quantize_f32_to_i8(normalized, in_scale, in_zp);
    }
    
    // Invocar modelo
    int rc = tflm_invoke();
    if (rc != 0) {
        printf("✗ ERRO tflm_invoke: %d\n", rc);
        return;
    }
    
    printf("✓ Invoke OK\n\n");
    
    // Converter para probabilidades
    float probs[10];
    softmax_i8_to_probs(out, out_scale, out_zp, probs, 10);
    
    // Mostrar probabilidades
    printf("PROBABILIDADES:\n");
    printf("┌─────┬──────────┐\n");
    printf("│ Dig │  Prob    │\n");
    printf("├─────┼──────────┤\n");
    for (int i = 0; i < 10; i++) {
        printf("│  %d  │ %6.2f%% │", i, probs[i]);
        if (i == label) printf(" ← REAL");
        printf("\n");
    }
    printf("└─────┴──────────┘\n\n");
    
    // Predição
    int pred = argmax_i8(out, 10);
    bool correct = (pred == label);
    
    printf("╔═══════════════════════════════════════╗\n");
    if (correct) {
        printf("║  ✓ ACERTOU! Pred=%d Real=%d          ║\n", pred, label);
    } else {
        printf("║  ✗ ERROU! Pred=%d Real=%d            ║\n", pred, label);
    }
    printf("║  Confiança: %.1f%%                     ║\n", probs[pred]);
    printf("╚═══════════════════════════════════════╝\n\n");
    
    // Atualizar display
    show_results(probs, label);
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    stdio_init_all();
    sleep_ms(2000);
    
    printf("\n");
    printf("╔════════════════════════════════════════════════╗\n");
    printf("║  MNIST CNN INT8 - Probabilidades em %%         ║\n");
    printf("║  Raspberry Pi Pico W + TFLite Micro           ║\n");
    printf("╚════════════════════════════════════════════════╝\n");
    printf("\n");
    
    // Init I2C + display
    i2c_init(i2c1, 400 * 1000);
    gpio_set_function(14, GPIO_FUNC_I2C);
    gpio_set_function(15, GPIO_FUNC_I2C);
    gpio_pull_up(14);
    gpio_pull_up(15);
    
    ssd1306_init(&display, 128, 64, false, 0x3C, i2c1);
    ssd1306_config(&display);
    ssd1306_fill(&display, false);
    ssd1306_draw_string(&display, "MNIST CNN", 0, 0, false);
    ssd1306_draw_string(&display, "Modo: Probs %", 0, 16, false);
    ssd1306_draw_string(&display, "Aguarde...", 0, 28, false);
    ssd1306_send_data(&display);
    
    // Init TFLM
    printf("Inicializando TensorFlow Lite Micro...\n");
    int rc = tflm_init();
    if (rc != 0) {
        printf("✗ ERRO tflm_init: %d\n", rc);
        ssd1306_fill(&display, false);
        ssd1306_draw_string(&display, "ERROR!", 0, 0, false);
        ssd1306_draw_string(&display, "TFLM Init Failed", 0, 20, false);
        ssd1306_send_data(&display);
        while (1) tight_loop_contents();
    }
    
    printf("✓ TFLM OK - Arena: %d bytes\n", tflm_arena_used_bytes());
    
    ssd1306_fill(&display, false);
    ssd1306_draw_string(&display, "PRONTO!", 0, 0, false);
    ssd1306_draw_string(&display, "Envie linha CSV:", 0, 16, false);
    ssd1306_draw_string(&display, "label,p1,...,p784", 0, 28, false);
    ssd1306_send_data(&display);
    
    printf("\n╔════════════════════════════════════════════════╗\n");
    printf("║  FORMATO CSV:                                  ║\n");
    printf("║  label,pixel1,pixel2,...,pixel784              ║\n");
    printf("║                                                 ║\n");
    printf("║  Cole uma linha do arquivo de teste e          ║\n");
    printf("║  pressione ENTER                               ║\n");
    printf("╚════════════════════════════════════════════════╝\n\n");
    printf("Aguardando dados...\n\n");
    
    csv_pos = 0;
    last_byte_time = get_absolute_time();
    
    while (1) {
        int ch = getchar_timeout_us(100);
        
        if (ch >= 0) {
            last_byte_time = get_absolute_time();
            
            if (ch == '\n' || ch == '\r') {
                if (csv_pos > 0) {
                    csv_buffer[csv_pos] = '\0';
                    
                    printf(">>> Recebido %d caracteres\n", csv_pos);
                    
                    uint8_t label;
                    uint8_t pixels[MNIST_SIZE];
                    
                    if (parse_csv_line(csv_buffer, &label, pixels) == 0) {
                        printf("✓ Parse OK\n");
                        run_inference(label, pixels);
                    } else {
                        printf("✗ Parse FALHOU\n");
                        printf("Formato esperado: label,p1,p2,...,p784\n\n");
                    }
                    
                    csv_pos = 0;
                }
            } 
            else if (csv_pos < CSV_BUFFER_SIZE - 1) {
                if (ch >= 32 && ch <= 126) {
                    csv_buffer[csv_pos++] = (char)ch;
                } else if (ch == '\t') {
                    csv_buffer[csv_pos++] = ' ';
                }
                
                if (csv_pos % 500 == 0) {
                    printf("Recebendo: %d chars...\n", csv_pos);
                }
            } else {
                printf("✗ Buffer cheio! Resetando\n");
                csv_pos = 0;
            }
        } else {
            int64_t elapsed = absolute_time_diff_us(last_byte_time, get_absolute_time());
            if (csv_pos > 0 && elapsed > 3000000) {
                printf("Timeout - resetando (%d chars)\n", csv_pos);
                csv_pos = 0;
            }
        }
        
        tight_loop_contents();
    }
    
    return 0;
}