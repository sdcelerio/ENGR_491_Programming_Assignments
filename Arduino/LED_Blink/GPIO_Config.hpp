#pragma once

// Pin definitions (PORTB)
#define LED_BIT_FRONT PB1
#define LED_BIT_LEFT  PB2
#define LED_BIT_RIGHT PB3

// Frequency Symbols
#define FREQ_IDLE  1200
#define FREQ_00    300
#define FREQ_01    400
#define FREQ_10    500
#define FREQ_11    600
#define FREQ_SEND  900

// Timings
#define SYMBOL_DUR_US    100000UL // Time dedicated for each symbol
#define PREAMBLE_DUR_US   500000UL
#define REPEAT_DELAY_US  2000000UL // Time between each message