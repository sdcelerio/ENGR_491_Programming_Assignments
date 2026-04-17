#pragma once
#include <Arduino.h>
#include "GPIO_Config.hpp"

typedef struct {
  uint8_t       Pin_Bit;
  unsigned long Half_Period_us;
  unsigned long Last_Toggle_us;
} LED_Channel;

extern LED_Channel Channels[3];

void Set_All_To_Frequency(uint16_t Hz);
void Tick_LEDs(unsigned long Curr_Time);