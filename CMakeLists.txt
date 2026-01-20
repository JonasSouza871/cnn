# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.1.0)
set(toolchainVersion 13_3_Rel1)
set(picotoolVersion 2.1.0)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
cmake_minimum_required(VERSION 3.13)

# == Pico SDK ==
include(pico_sdk_import.cmake)

project(cnn_mnist C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(PICO_BOARD pico_w CACHE STRING "Pico board")

pico_sdk_init()

# -------------------------------------------------------------------
# pico-tflmicro
set(BUILD_TESTING OFF CACHE BOOL "Disable building tests" FORCE)
set(PICO_TFLMICRO_BUILD_TESTS OFF CACHE BOOL "Disable pico-tflmicro tests" FORCE)
set(PICO_TFLMICRO_BUILD_EXAMPLES OFF CACHE BOOL "Disable pico-tflmicro examples" FORCE)
set(PICO_TFLMICRO_BUILD_BENCHMARKS OFF CACHE BOOL "Disable pico-tflmicro benchmarks" FORCE)

add_subdirectory(pico-tflmicro pico-tflmicro-build EXCLUDE_FROM_ALL)

# Detect library target
set(TFLM_TARGET "")
foreach(candidate IN ITEMS pico_tflmicro pico-tflmicro tflmicro pico_tflmicro_lib)
    if(TARGET ${candidate})
        set(TFLM_TARGET ${candidate})
        break()
    endif()
endforeach()

if(TFLM_TARGET STREQUAL "")
    message(FATAL_ERROR
        "Could not find a pico-tflmicro library target."
    )
endif()

# ---------------------------------------------------------------hard----
# Bibliotecas
add_library(ssd1306 STATIC
    Firmware/lib/ssd1306.c
)

target_include_directories(ssd1306 PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/Firmware/lib
)

target_link_libraries(ssd1306 PUBLIC
    hardware_i2c
    hardware_gpio
)

# -------------------------------------------------------------------
# Executável principal
add_executable(cnn_mnist
    Firmware/cnn_mnist.c
    Firmware/tflm_wrapper.cpp
)

pico_set_program_name(cnn_mnist "cnn_mnist")
pico_set_program_version(cnn_mnist "0.1")

# I/O
pico_enable_stdio_uart(cnn_mnist 1)
pico_enable_stdio_usb(cnn_mnist 1)

target_include_directories(cnn_mnist PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/Firmware
    ${CMAKE_CURRENT_LIST_DIR}/Firmware/lib
    ${CMAKE_CURRENT_LIST_DIR}/Models
)

target_link_libraries(cnn_mnist PRIVATE
    pico_stdlib
    hardware_i2c
    hardware_gpio
    ssd1306
    ${TFLM_TARGET}
)

pico_add_extra_outputs(cnn_mnist)
