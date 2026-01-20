# CNN MNIST (INT8) no RP2040 / Raspberry Pi Pico W (TensorFlow Lite Micro)

Implementação de **inferência embarcada** de uma **CNN quantizada em INT8** treinada no **MNIST**, executando em **RP2040 (Pico W)** usando **TensorFlow Lite Micro (TFLM)** via **pico-tflmicro** (submódulo/pasta vendorizada no repositório).

> **Objetivo:** demonstrar um pipeline TinyML completo: modelo **.tflite** convertido para array C (`mnist_cnn_int8_model.h`), entrada de amostra (`mnist_sample.h`), execução do interpretador TFLM no RP2040 e impressão/telemetria via USB.

---

## Principais recursos

- ✅ **Modelo CNN INT8** (quantizado) para classificação de dígitos `0–9`
- ✅ Execução em **baremetal** no **RP2040** (Pico SDK)
- ✅ Integração com **pico-tflmicro** (TFLM + CMSIS-NN habilitado quando disponível)
- ✅ Exemplo de **entrada em C** (`mnist_sample.h`) para teste rápido sem sensores/câmera
- ✅ Estrutura pronta para substituir amostras e/ou trocar o modelo

---

## Estrutura do repositório (arquivos anexos)

- `cnn_mnist.c`  
  Código da aplicação (loop principal, preparação de entrada, chamada do wrapper e leitura da saída).
- `tflm_wrapper.h` / `tflm_wrapper.cpp`  
  Wrapper para encapsular a inicialização do TFLM (model → interpreter → tensor arena) e a inferência (`invoke`).
- `mnist_cnn_int8_model.h`  
  Array C com o modelo TFLite **INT8** (ex.: `const unsigned char ...[]`).
- `mnist_sample.h`  
  Amostra(s) de entrada (imagem 28×28) em formato compatível com o modelo.
- `CMakeLists.txt`  
  Build do projeto com Pico SDK + pico-tflmicro.
- `pico_sdk_import.cmake`  
  Import padrão do Pico SDK (conforme templates oficiais do SDK).

---

## Requisitos

### Software
- **Windows 10/11** (ou Linux/macOS)  
- **Pico SDK** configurado (no seu caso: `~/.pico-sdk/sdk/2.2.0`)
- Toolchain **arm-none-eabi-gcc** (no seu log: `14_2_Rel1`)
- **CMake** + **Ninja** (recomendado no Windows)
- (Opcional) **picotool** para gerar `.uf2`

### Hardware
- **Raspberry Pi Pico W** (ou Pico; ajuste `PICO_BOARD` conforme necessário)
- Cabo USB para gravação/Serial (CDC)

---

## Como compilar (Windows / PowerShell)

> A forma mais previsível é criar uma pasta `build` **curta** (e, no Windows, evitar caminhos longos).

1) Abra um PowerShell no diretório do projeto:

```powershell
cd "\cnn_mnist"
if (Test-Path .\build) { Remove-Item -Recurse -Force .\build }
mkdir build
cd build
```

2) Configure o build com Ninja (Pico W):

```powershell
cmake -G Ninja -DPICO_BOARD=pico_w ..
```

3) Compile:

```powershell
ninja
```

Se tudo estiver correto, serão gerados:
- `cnn_mnist.elf`
- `cnn_mnist.uf2`
- `cnn_mnist.bin` / `cnn_mnist.hex` / `cnn_mnist.dis`

---

## Gravar no Pico W

1) Coloque o Pico W em modo BOOTSEL (segure **BOOTSEL** e conecte o USB).  
2) Copie o arquivo `cnn_mnist.uf2` para o drive `RPI-RP2`.

---

## Monitor serial (USB)

- Se o projeto estiver com **stdio USB** habilitado, use:
  - Windows: PuTTY / TeraTerm / Arduino Serial Monitor / VSCode Serial Monitor
  - Linux/macOS: `screen /dev/ttyACM0 115200` (ou equivalente)

> Em muitos exemplos do Pico SDK, o baudrate não importa para USB CDC, mas mantenha `115200` por padrão.

---

## Detalhes do modelo (MNIST)

O MNIST usa imagens **28×28** em escala de cinza. Para CNNs quantizadas em INT8, normalmente o fluxo é:

1. Entrada original: `uint8` (0–255) ou `float` (0–1)
2. Quantização: `int8` usando `scale` e `zero_point` do tensor de entrada

**Importante:** a pré-processamento correto depende do *input tensor* do modelo:
- Se o tensor de entrada for `int8`, você deve quantizar a imagem para `int8`.
- Se for `uint8`, deve manter `uint8`.
- Se for `float32`, deve normalizar para `float`.

✅ **Dica prática:** verifique no wrapper (ou via `interpreter->input(0)->type` e parâmetros) qual é o tipo e como mapear a amostra.

---

## API do wrapper (tflm_wrapper)

O wrapper existe para esconder a “complexidade” padrão do TFLM:

- carregar o modelo (`tflite::GetModel(...)`)
- criar o `tflite::MicroInterpreter`
- alocar tensores (`AllocateTensors`)
- mapear `input` e `output`
- executar a inferência (`Invoke`)

### Interface sugerida (padrão)
> O cabeçalho/implementação anexos estão com *trechos indicativos* (reticências `...`). A ideia típica é algo como:

- `bool tflm_init();`
- `bool tflm_invoke(const int8_t* in, int8_t* out);`
- ou uma versão que retorna ponteiros para `input()`/`output()`

Se você padronizar assim, o `cnn_mnist.c` fica limpo e fácil de adaptar para novos modelos.

---

## Troubleshooting (erros comuns do seu log)

### 1) `fatal error: opening dependency file ... .obj.d: No such file or directory`
Isso é **clássico de Windows + caminho longo** (limite de path) durante builds grandes (como CMSIS-NN + testes).

**Correções recomendadas (faça pelo menos uma):**
- ✅ Mover o projeto para um caminho curto, ex.: `C:\p\cnn_mnist`
- ✅ Habilitar **Win32 long paths** no Windows (Política/Registro)
- ✅ Evitar compilar **tests/benchmarks** do pico-tflmicro no build do seu app

### 2) Muitos warnings de `CMAKE_OBJECT_PATH_MAX` e “build may not work”
Não são erros por si só, mas indicam que o caminho está no limite — e costuma levar ao erro do item (1).  
Solução: **caminho curto** + **desativar testes**.

### 3) `cannot find -lpico_tflmicro`
Isso ocorre quando o CMake tenta linkar com uma lib chamada `pico_tflmicro`, mas o alvo real no `pico-tflmicro` tem outro nome (varia por fork/versão) **ou a biblioteca não foi adicionada ao build**.

✅ Solução: detectar o target correto do pico-tflmicro e linkar pelo **nome do TARGET** (não por `-l...` manual).

> Uma versão robusta do `CMakeLists.txt` costuma:
- `add_subdirectory(pico-tflmicro pico-tflmicro-build EXCLUDE_FROM_ALL)`
- localizar o alvo real (ex.: `pico_tflmicro`, `pico-tflmicro`, etc.)
- `target_link_libraries(cnn_mnist PRIVATE ${TFLM_TARGET})`

### 4) `PICO_DEFAULT_LED_PIN was not declared`
Esse erro veio de exemplos do pico-tflmicro (ex.: `examples/hello_world/...`) que assumem LED “default” definido no board.

✅ Solução: **não compilar exemplos/testes do pico-tflmicro** junto do seu app, ou ajustar o exemplo.

---

## Recomendações de build (CMake) para evitar testes do pico-tflmicro

Para builds no Windows, é altamente recomendável **desativar testes/benchmarks** do pico-tflmicro.  
Exemplo de flags típicas (dependem do fork):

```cmake
set(PICO_TFLMICRO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PICO_TFLMICRO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(PICO_TFLMICRO_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
```

Além disso, use `EXCLUDE_FROM_ALL` ao adicionar o subdiretório:

```cmake
add_subdirectory(pico-tflmicro pico-tflmicro-build EXCLUDE_FROM_ALL)
```

---

## Como trocar a amostra / testar outros dígitos

1) Substitua o conteúdo de `mnist_sample.h` por outra imagem 28×28.
2) Garanta que o formato está compatível com o tensor de entrada do modelo:
   - `int8` com quantização correta, ou
   - `uint8`, ou
   - `float32` normalizado.

Sugestão: mantenha um script Python/Colab para:
- carregar uma imagem MNIST
- aplicar o mesmo pré-processamento do treino
- exportar para `.h` (array C) já no tipo correto do modelo

---

## Créditos e referências

- Raspberry Pi Pico SDK (RP2040)
- TensorFlow Lite Micro (TFLM)
- CMSIS-NN (otimizações para kernels INT8)

---

## Contato / manutenção

- Autor: Ricardo Menezes Prates (UNIVASF)  
- Objetivo educacional - Embarcatech.
