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

#define MNIST_SIZE 784           // 28x28 pixels
#define CSV_BUFFER_SIZE 8192     // buffer pra receber linha CSV via serial

ssd1306_t display;
static char csv_buffer[CSV_BUFFER_SIZE];
static int csv_pos = 0;
static absolute_time_t last_byte_time;  // usado pra detectar timeout


// Retorna o índice do maior valor no array (classe predita)
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


// Converte float [0-1] pra int8 quantizado usando scale e zero_point
static int8_t quantize_f32_to_i8(float x, float scale, int zp) {
    float q = (x / scale) + (float)zp;  // formula: q = x/scale + zero_point
    if (q > 127.0f) q = 127.0f;         // clamp no range int8
    if (q < -128.0f) q = -128.0f;
    return (int8_t)q;
}


// Converte logits int8 pra probabilidades usando softmax
// probs[] vai ter os valores em porcentagem (0-100)
void softmax_i8_to_probs(const int8_t* logits, float scale, int zero_point, float* probs, int n) {
    float dequant[10];
    float max_val = -1e9f;
    
    // Primeiro dequantiza os logits int8 -> float
    for (int i = 0; i < n; i++) {
        dequant[i] = scale * ((float)logits[i] - (float)zero_point);  // x = scale * (q - zp)
        if (dequant[i] > max_val) max_val = dequant[i];  // acha o max pra estabilidade numérica
    }
    
    // Softmax: exp(x - max) / sum(exp(x - max))
    // Subtrair max evita overflow no exp()
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        dequant[i] = expf(dequant[i] - max_val);
        sum += dequant[i];
    }
    
    // Normaliza e converte pra porcentagem
    for (int i = 0; i < n; i++) {
        probs[i] = (dequant[i] / sum) * 100.0f;
    }
}


// Faz parse de uma linha CSV no formato: label,pixel1,pixel2,...,pixel784
// Retorna 0 se OK, -1 se erro
static int parse_csv_line(const char* line, uint8_t* label, uint8_t* pixels) {
    const char* ptr = line;
    char num_buf[8];
    int num_pos = 0;
    int field_count = 0;
    
    // Pula espaços e ignora linhas vazias/comentários
    while (*ptr == ' ' || *ptr == '\t') ptr++;
    if (*ptr == '\0' || *ptr == '\n' || *ptr == '#') return -1;
    
    // Lê cada campo separado por vírgula
    while (*ptr != '\0' && *ptr != '\n' && field_count <= MNIST_SIZE) {
        // Lê os dígitos do número atual
        num_pos = 0;
        while (*ptr >= '0' && *ptr <= '9' && num_pos < sizeof(num_buf)-1) {
            num_buf[num_pos++] = *ptr++;
        }
        num_buf[num_pos] = '\0';
        
        if (num_pos > 0) {
            int value = atoi(num_buf);
            
            if (field_count == 0) {
                *label = (uint8_t)(value & 0xFF);  // primeiro campo é o label
            } else if (field_count <= MNIST_SIZE) {
                // Campos 1-784 são os pixels, clamp em [0-255]
                if (value < 0) value = 0;
                if (value > 255) value = 255;
                pixels[field_count - 1] = (uint8_t)value;
            }
            
            field_count++;
        }
        
        // Pula vírgulas e espaços
        while (*ptr == ',' || *ptr == ' ' || *ptr == '\t') ptr++;
    }
    
    return (field_count == 785) ? 0 : -1;  // precisa ter exatamente 1 label + 784 pixels
}


// Mostra no display OLED o top 3 de predições com probabilidades
void show_results(const float* probs, uint8_t true_label) {
    typedef struct {
        int digit;
        float prob;
    } prediction_t;
    
    // Copia probabilidades pra struct que vai ser ordenada
    prediction_t preds[10];
    for (int i = 0; i < 10; i++) {
        preds[i].digit = i;
        preds[i].prob = probs[i];
    }
    
    // Ordena decrescente por probabilidade (bubble sort)
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9 - i; j++) {
            if (preds[j].prob < preds[j+1].prob) {
                prediction_t temp = preds[j];
                preds[j] = preds[j+1];
                preds[j+1] = temp;
            }
        }
    }
    
    // Monta tela do display
    ssd1306_fill(&display, false);
    
    char line[24];
    
    // Linha 0: label verdadeiro
    snprintf(line, sizeof(line), "REAL: %d", true_label);
    ssd1306_draw_string(&display, line, 0, 0, false);
    
    // Top 3 predições com probabilidades
    snprintf(line, sizeof(line), "1:%d %.1f%%", preds[0].digit, preds[0].prob);
    ssd1306_draw_string(&display, line, 0, 14, false);
    
    snprintf(line, sizeof(line), "2:%d %.1f%%", preds[1].digit, preds[1].prob);
    ssd1306_draw_string(&display, line, 0, 26, false);
    
    snprintf(line, sizeof(line), "3:%d %.1f%%", preds[2].digit, preds[2].prob);
    ssd1306_draw_string(&display, line, 0, 38, false);
    
    // Última linha: mostra se acertou ou errou
    bool correct = (preds[0].digit == true_label);
    snprintf(line, sizeof(line), "PRED:%d %s", preds[0].digit, correct ? "OK!" : "ERR");
    ssd1306_draw_string(&display, line, 0, 52, false);
    
    ssd1306_send_data(&display);  // envia buffer pro display
}


// Executa a inferência completa: quantiza input, roda modelo, calcula probs e exibe
static void run_inference(uint8_t label, const uint8_t* pixels) {
    // Variáveis static pra não precisar buscar a cada inferência
    static int8_t *in = NULL, *out = NULL;
    static float in_scale, out_scale;
    static int in_zp, out_zp;
    static bool initialized = false;
    
    // Na primeira execução, pega os ponteiros e parâmetros de quantização do modelo
    if (!initialized) {
        int in_bytes, out_bytes;
        in = tflm_input_ptr(&in_bytes);    // ponteiro pro tensor de entrada
        out = tflm_output_ptr(&out_bytes); // ponteiro pro tensor de saída
        in_scale = tflm_input_scale();
        in_zp = tflm_input_zero_point();
        out_scale = tflm_output_scale();
        out_zp = tflm_output_zero_point();
        
        printf("\nTFLM config:\n");
        printf("  Input: scale=%.6f, zero_point=%d\n", in_scale, in_zp);
        printf("  Output: scale=%.6f, zero_point=%d\n\n", out_scale, out_zp);
        initialized = true;
    }
    
    printf("\n--- Nova inferencia ---\n");
    printf("Label real: %d\n", label);
    printf("Primeiros pixels: %d,%d,%d,%d,%d\n", 
           pixels[0], pixels[1], pixels[2], pixels[3], pixels[4]);
    
    // Normaliza pixels [0-255] -> [0-1] e quantiza pra int8
    for (int i = 0; i < MNIST_SIZE; i++) {
        float normalized = (float)pixels[i] / 255.0f;
        in[i] = quantize_f32_to_i8(normalized, in_scale, in_zp);
    }
    
    // Roda a inferência
    int rc = tflm_invoke();
    if (rc != 0) {
        printf("ERRO tflm_invoke: %d\n", rc);
        return;
    }
    
    printf("Invoke OK\n\n");
    
    // Converte saída int8 pra probabilidades em %
    float probs[10];
    softmax_i8_to_probs(out, out_scale, out_zp, probs, 10);
    
    // Exibe todas as probabilidades na serial
    printf("Probabilidades:\n");
    for (int i = 0; i < 10; i++) {
        printf("  %d: %6.2f%%", i, probs[i]);
        if (i == label) printf(" <- real");  // marca qual é o label verdadeiro
        printf("\n");
    }
    
    // Calcula predição (classe com maior probabilidade)
    int pred = argmax_i8(out, 10);
    bool correct = (pred == label);
    
    printf("\nResultado: pred=%d real=%d %s (confianca: %.1f%%)\n\n", 
           pred, label, correct ? "OK" : "ERRO", probs[pred]);
    
    // Atualiza display OLED
    show_results(probs, label);
}


int main() {
    stdio_init_all();
    sleep_ms(2000);  // aguarda inicialização da serial
    
    printf("\nMNIST CNN INT8 - Raspberry Pi Pico W + TFLite Micro\n");
    printf("Modo: Probabilidades em %%\n\n");
    
    // Configura I2C pro display OLED
    i2c_init(i2c1, 400 * 1000);           // 400kHz
    gpio_set_function(14, GPIO_FUNC_I2C);  // GP14 = SDA
    gpio_set_function(15, GPIO_FUNC_I2C);  // GP15 = SCL
    gpio_pull_up(14);
    gpio_pull_up(15);
    
    // Inicializa display SSD1306 128x64
    ssd1306_init(&display, 128, 64, false, 0x3C, i2c1);
    ssd1306_config(&display);
    ssd1306_fill(&display, false);
    ssd1306_draw_string(&display, "MNIST CNN", 0, 0, false);
    ssd1306_draw_string(&display, "Modo: Probs %", 0, 16, false);
    ssd1306_draw_string(&display, "Aguarde...", 0, 28, false);
    ssd1306_send_data(&display);
    
    // Inicializa TensorFlow Lite Micro
    printf("Inicializando TensorFlow Lite Micro...\n");
    int rc = tflm_init();
    if (rc != 0) {
        printf("ERRO tflm_init: %d\n", rc);
        ssd1306_fill(&display, false);
        ssd1306_draw_string(&display, "ERROR!", 0, 0, false);
        ssd1306_draw_string(&display, "TFLM Init Failed", 0, 20, false);
        ssd1306_send_data(&display);
        while (1) tight_loop_contents();
    }
    
    printf("TFLM OK - Arena usado: %d bytes\n", tflm_arena_used_bytes());
    
    // Atualiza display pra modo pronto
    ssd1306_fill(&display, false);
    ssd1306_draw_string(&display, "PRONTO!", 0, 0, false);
    ssd1306_draw_string(&display, "Envie linha CSV:", 0, 16, false);
    ssd1306_draw_string(&display, "label,p1,...,p784", 0, 28, false);
    ssd1306_send_data(&display);
    
    printf("\nFormato esperado: label,pixel1,pixel2,...,pixel784\n");
    printf("Cole uma linha do CSV de teste e pressione ENTER\n");
    printf("Aguardando dados...\n\n");
    
    csv_pos = 0;
    last_byte_time = get_absolute_time();
    
    // Loop principal: recebe dados via serial e processa
    while (1) {
        int ch = getchar_timeout_us(100);  // tenta ler char com timeout de 100us
        
        if (ch >= 0) {  // recebeu algum char
            last_byte_time = get_absolute_time();  // atualiza timestamp
            
            // Enter ou newline: processa a linha recebida
            if (ch == '\n' || ch == '\r') {
                if (csv_pos > 0) {
                    csv_buffer[csv_pos] = '\0';  // termina string
                    
                    printf("Recebido %d chars\n", csv_pos);
                    
                    uint8_t label;
                    uint8_t pixels[MNIST_SIZE];
                    
                    // Faz parse da linha CSV
                    if (parse_csv_line(csv_buffer, &label, pixels) == 0) {
                        printf("Parse OK\n");
                        run_inference(label, pixels);  // executa inferência
                    } else {
                        printf("Parse FALHOU - formato: label,p1,p2,...,p784\n\n");
                    }
                    
                    csv_pos = 0;  // reseta buffer
                }
            } 
            // Char normal: adiciona no buffer
            else if (csv_pos < CSV_BUFFER_SIZE - 1) {
                if (ch >= 32 && ch <= 126) {  // char imprimível
                    csv_buffer[csv_pos++] = (char)ch;
                } else if (ch == '\t') {
                    csv_buffer[csv_pos++] = ' ';  // converte tab em espaço
                }
                
                // Feedback visual a cada 500 chars (linha CSV é grande)
                if (csv_pos % 500 == 0) {
                    printf("Recebendo: %d chars...\n", csv_pos);
                }
            } else {
                // Buffer estourou, reseta
                printf("Buffer cheio! Resetando\n");
                csv_pos = 0;
            }
        } else {
            // Nenhum char recebido, verifica timeout
            int64_t elapsed = absolute_time_diff_us(last_byte_time, get_absolute_time());
            if (csv_pos > 0 && elapsed > 3000000) {  // 3 segundos sem receber nada
                printf("Timeout - resetando (%d chars)\n", csv_pos);
                csv_pos = 0;
            }
        }
        
        tight_loop_contents();  // yield pra watchdog
    }
    
    return 0;
}
