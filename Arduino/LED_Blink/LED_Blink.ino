#include <Arduino.h>
#include <avr/pgmspace.h>
#include "LEDs.hpp"
#include "Symbols.hpp"

// ─── Setup / Loop ─────────────────────────────────────────────────────────────
void setup() {
  DDRB |= (1 << LED_BIT_FRONT) | (1 << LED_BIT_LEFT) | (1 << LED_BIT_RIGHT);
  Set_All_To_Frequency(FREQ_IDLE);
}

void loop() {
  unsigned long now = micros();
  Tick_Symbols(now);
  Tick_LEDs(now);
}