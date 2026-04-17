#pragma once
#include <Arduino.h>
#include "GPIO_Config.hpp"

struct Symbol_Info {
  uint16_t      freq_hz;
  unsigned long dur_us;
};

void Tick_Symbols(unsigned long Curr_Time);