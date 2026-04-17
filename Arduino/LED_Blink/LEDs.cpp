#include "GPIO_Config.hpp"
#include "LEDs.hpp"

LED_Channel Channels[3] = {
  { LED_BIT_FRONT, 0, 0 },
  { LED_BIT_LEFT,  0, 0 },
  { LED_BIT_RIGHT, 0, 0 },
};

void Set_All_To_Frequency(uint16_t Hz) {
  unsigned long New_Half_Period = 500000UL / Hz;
  for (auto& Curr_Channel : Channels) 
    Curr_Channel.Half_Period_us = New_Half_Period;
}

void Tick_LEDs(unsigned long Curr_Time) {
  for (auto& Curr_Channel : Channels) {
    if (Curr_Channel.Half_Period_us > 0 && Curr_Time - Curr_Channel.Last_Toggle_us >= Curr_Channel.Half_Period_us) {
      PINB = (1 << Curr_Channel.Pin_Bit);
      Curr_Channel.Last_Toggle_us = Curr_Time;
    }
  }
}