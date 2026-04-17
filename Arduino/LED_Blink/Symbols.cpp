#include "Symbols.hpp"
#include "LEDs.hpp"
#include <avr/pgmspace.h>

// ─── Message ──────────────────────────────────────────────────────────────────
static const char message[] PROGMEM = "Hello World";
#define MSG_LEN (sizeof(message) - 1)

// ─── Sequence layout ──────────────────────────────────────────────────────────
#define STEPS_PER_CHAR 8
#define TOTAL_STEPS    (1 + (uint16_t)MSG_LEN * STEPS_PER_CHAR + 1)

// ─── Sequence state ───────────────────────────────────────────────────────────
static uint16_t      stepIndex       = 0;
static unsigned long stepStart_us    = 0;
static bool          waitingToRepeat = false;
static unsigned long repeatStart_us  = 0;

// ─── Helpers ──────────────────────────────────────────────────────────────────
static uint16_t dibitsToFreq(uint8_t dibit) {
  switch (dibit & 0x03) {
    case 0: return FREQ_00;
    case 1: return FREQ_01;
    case 2: return FREQ_10;
    default: return FREQ_11;
  }
}

static Symbol_Info computeStep(uint16_t idx) {
  if (idx == 0)               return { FREQ_IDLE, PREAMBLE_DUR_US };
  if (idx == TOTAL_STEPS - 1) return { FREQ_IDLE, PREAMBLE_DUR_US };

  uint16_t charStep = idx - 1;
  uint8_t  charIdx  = charStep / STEPS_PER_CHAR;
  uint8_t  subStep  = charStep % STEPS_PER_CHAR;

  if (subStep % 2 == 0) {
    return { FREQ_SEND, SYMBOL_DUR_US };
  } else {
    uint8_t c      = pgm_read_byte(&message[charIdx]);
    uint8_t dibIdx = subStep / 2;
    uint8_t shift  = dibIdx * 2;          // LSB first
    uint8_t dibit  = (c >> shift) & 0x03;
    return { dibitsToFreq(dibit), SYMBOL_DUR_US };
  }
}

// ─── Sequence engine ──────────────────────────────────────────────────────────
void Tick_Symbols(unsigned long now) {
  if (waitingToRepeat) {
    if (now - repeatStart_us >= REPEAT_DELAY_US) {
      stepIndex       = 0;
      stepStart_us    = now;
      waitingToRepeat = false;
      Set_All_To_Frequency(computeStep(0).freq_hz);
    }
    return;
  }

  Symbol_Info cur = computeStep(stepIndex);
  if (now - stepStart_us >= cur.dur_us) {
    stepIndex++;
    if (stepIndex < TOTAL_STEPS) {
      Set_All_To_Frequency(computeStep(stepIndex).freq_hz);
      stepStart_us = now;
    } else {
      Set_All_To_Frequency(FREQ_IDLE);
      waitingToRepeat = true;
      repeatStart_us  = now;
    }
  }
}