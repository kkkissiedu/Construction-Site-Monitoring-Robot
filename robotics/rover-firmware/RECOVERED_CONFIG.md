# Rover STM32 — recovered configuration

Recovered on 2026-11-17 by disassembling `bluepill_flash_read1.bin` (read-only static analysis; the
firmware was not executed). Source: STM32F103 Blue Pill, 31,188 of 32,768 bytes used, HAL-generated
(CubeMX) code. Backup SHA-256: `4e788ff8917552ed1ba65531ef4cecad634b561a5f0d4113701bde9a10232620`.

## Configuration

| Area | Setting |
|---|---|
| Clock | HSE 8 MHz, PLL x9, SYSCLK 72 MHz, AHB 72, APB1 36 (timer clock 72), APB2 72, flash latency 2 |
| TIM1 | PWM CH1-CH4 on PA8-PA11 (AF push-pull, 2 MHz), prescaler 0, period 65535, PWM mode 1, active high, automatic output on |
| TIM2 | PWM CH1-CH4 on PA0-PA3 (AF push-pull, 2 MHz), prescaler 0, period 65535, PWM mode 1, active high |
| PWM rate | 72 MHz / 65536 = about 1.1 kHz, 16-bit resolution |
| I2C1 | PB6 = SCL, PB7 = SDA (AF open-drain), 100 kHz, 7-bit addressing, initialised but never used |
| Debug | JTAG disabled, SWD enabled (PA13/PA14). PB3, PB4, PA15 free |
| GPIO clocks | GPIOA, GPIOB, GPIOD enabled; no pin configured by MX_GPIO_Init |
| Absent | USART, USB, ADC, DMA use, watchdog, EXTI |

## Behaviour

The main loop is a fixed demo: forward 10 s, stop 1 s, backward 10 s, stop 1 s, repeat. Every motor is
either at 100% duty (CCR = 65535) or 0. No serial commands, no speed control, no IMU reads.

## Motor pin map (from the user, matches the code)

| Motor | Forward pin | Reverse pin |
|---|---|---|
| 1 | PA8 (TIM1 CH1) | PA9 (TIM1 CH2) |
| 2 | PA10 (TIM1 CH3) | PA11 (TIM1 CH4) |
| 3 | PA2 (TIM2 CH3) | PA3 (TIM2 CH4) |
| 4 | PA0 (TIM2 CH1) | PA1 (TIM2 CH2) |

Forward pin = RPWM, reverse pin = LPWM on the BTS7960 (HW-039) drivers; EN pins tied high.

## Function addresses (flash base 0x08000000)

| Address | Function |
|---|---|
| 0x0800053C | main |
| 0x080004D8 | SystemClock_Config |
| 0x0800014C | MX_GPIO_Init |
| 0x080002C0 | MX_I2C1_Init |
| 0x080002F8 | MX_TIM1_Init |
| 0x08000404 | MX_TIM2_Init |
| 0x08000234 | forward (all) |
| 0x08000278 | backward (all) |
| 0x08000210 | stop (all) |
| 0x08000190 | set pair: zero two channels, drive one at full duty |

## Notes

- IMU (GY-521) was wired SCL = PB8, SDA = PB7. The firmware expects SCL = PB6. Move SCL to PB6.
- Restore the original firmware: `python -m stm32loader -p COM8 -f F1 -b 115200 -a 0x08000000 -e -w -v bluepill_flash_read1.bin`
  (BOOT0 = 1, then back to 0). Run with `%USERPROFILE%\.platformio\penv\Scripts\python.exe`.
- The original `.bin` files are third-party firmware and are kept outside this repo.
