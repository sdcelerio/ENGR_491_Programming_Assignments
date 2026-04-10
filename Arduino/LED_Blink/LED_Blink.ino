#include <Arduino.h>

// ─── Pin definitions (PORTB) ────────────────────────────────────────────────
#define LED_BIT_FRONT PB1
#define LED_BIT_LEFT  PB2
#define LED_BIT_RIGHT PB3

// ─── Frequency symbol definitions ────────────────────────────────────────────
#define FREQ_IDLE  400.0f
#define FREQ_00    500.0f
#define FREQ_01    600.0f
#define FREQ_10    700.0f
#define FREQ_11    800.0f
#define FREQ_SEND  1000.0f
#define FREQ_STOP  1200.0f

#define REPEAT_DELAY_US 2000000UL

// ─── Structs ─────────────────────────────────────────────────────────────────
struct LEDChannel {
  uint8_t       bit;
  unsigned long halfPeriod_us;
  unsigned long lastToggle_us;
};

struct Symbol {
  float         freq;
  unsigned long duration_us;
};

// ─── Global LED channels ──────────────────────────────────────────────────────
LEDChannel channels[3] = {
  { LED_BIT_FRONT, 0, 0 },
  { LED_BIT_LEFT,  0, 0 },
  { LED_BIT_RIGHT, 0, 0 },
};

// ─── Sequence state ───────────────────────────────────────────────────────────
const Symbol* activeSymbol    = nullptr;
uint8_t       activeSymbolLen = 0;
uint8_t       symbolIndex     = 0;
unsigned long stepStart_us    = 0;

// ─── Repeat state ─────────────────────────────────────────────────────────────
bool          waitingToRepeat = false;
unsigned long repeatStart_us  = 0;
const Symbol* pendingSymbol   = nullptr;
uint8_t       pendingSymbolLen = 0;

// ─── Helpers ──────────────────────────────────────────────────────────────────
void setAllFreq(float hz) {
  unsigned long half = (unsigned long)(1000000.0f / (2.0f * hz));
  for (auto& ch : channels) ch.halfPeriod_us = half;
}

void startSymbol(const Symbol* sym, uint8_t len) {
  activeSymbol    = sym;
  activeSymbolLen = len;
  symbolIndex     = 0;
  stepStart_us    = micros();
  setAllFreq(sym[0].freq);
  waitingToRepeat = false;
}

void loopSymbol(const Symbol* sym, uint8_t len) {
  pendingSymbol    = sym;
  pendingSymbolLen = len;
  startSymbol(sym, len);
}

// ─── Sequence engine ──────────────────────────────────────────────────────────
void tickSymbol(unsigned long now) {
  if (waitingToRepeat) {
    if (now - repeatStart_us >= REPEAT_DELAY_US) {
      startSymbol(pendingSymbol, pendingSymbolLen);
    }
    return;
  }

  if (!activeSymbol || symbolIndex >= activeSymbolLen) return;

  if (now - stepStart_us >= activeSymbol[symbolIndex].duration_us) {
    symbolIndex++;
    if (symbolIndex < activeSymbolLen) {
      setAllFreq(activeSymbol[symbolIndex].freq);
      stepStart_us = now;
    } else {
      setAllFreq(FREQ_IDLE);
      activeSymbol = nullptr;
      if (pendingSymbol) {
        waitingToRepeat = true;
        repeatStart_us  = now;
      }
    }
  }
}

// ─── LED ticker ───────────────────────────────────────────────────────────────
void tickLEDs(unsigned long now) {
  for (auto& ch : channels) {
    if (ch.halfPeriod_us > 0 && now - ch.lastToggle_us >= ch.halfPeriod_us) {
      PINB = (1 << ch.bit);
      ch.lastToggle_us = now;
    }
  }
}

// ─── Message definition ───────────────────────────────────────────────────────
static const Symbol myMessage[] = {
  { FREQ_IDLE, 500000UL  },
  { FREQ_SEND, 1000000UL },
  { FREQ_00,   1000000UL },
  { FREQ_SEND, 1000000UL },
  { FREQ_01,   1000000UL },
  { FREQ_SEND, 1000000UL },
  { FREQ_10,   1000000UL },
  { FREQ_SEND, 1000000UL },
  { FREQ_11,   1000000UL },
  { FREQ_STOP, 1000000UL },
  { FREQ_IDLE, 500000UL  },
};

// ─── Setup / Loop ─────────────────────────────────────────────────────────────
void setup() {
  DDRB |= (1 << LED_BIT_FRONT) | (1 << LED_BIT_LEFT) | (1 << LED_BIT_RIGHT);
  setAllFreq(FREQ_IDLE);
  loopSymbol(myMessage, sizeof(myMessage) / sizeof(myMessage[0]));
}

void loop() {
  unsigned long now = micros();
  tickSymbol(now);
  tickLEDs(now);
}