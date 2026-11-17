# Rover Firmware — drive controller for the Construction Site Monitoring Robot

STM32F103C6 firmware that drives the rover's four wheel motors from text commands sent by the Jetson over a UART.

## Hardware

- Blue Pill with a **low-density STM32F103C6** (chip ID 0x412, 32 KiB flash, 10 KB RAM, only USART1 and USART2). It is not a C8: a C8 build puts the stack past the real RAM.
- Four BTS7960 (HW-039) drivers, one per motor, enable pins tied high. Each motor is a PWM pin pair: the first pin drives forward (RPWM), the second reverse (LPWM), and never both at once.
- Jetson link on **USART1 remapped to PB6 (TX) and PB7 (RX)**, 115200 8N1, 3.3 V logic, common ground.

| Motor | Wheel | Forward pin | Reverse pin | Timer |
|-------|-------|-------------|-------------|-------|
| 1 | front left  | PA8  | PA9  | TIM1 CH1 / CH2 |
| 2 | back left   | PA10 | PA11 | TIM1 CH3 / CH4 |
| 3 | front right | PA2  | PA3  | TIM2 CH3 / CH4 |
| 4 | back right  | PA0  | PA1  | TIM2 CH1 / CH2 |

PWM defaults to about 1.1 kHz with a 16-bit period, as in the original firmware. `C PWM` raises it up to 20 kHz.

## Serial protocol

Lines end in `\n` and are case-insensitive. Speeds are signed integers from -255 to 255.

| Command | Effect | Reply |
|---------|--------|-------|
| `V <left> <right>` | set side speeds; repeat faster than the failsafe time | `OK` |
| `T <motor> [speed]` | bench test: one motor for 1.5 s (default speed 60) | `OK` |
| `S` | stop everything now | `OK` |
| `P` | ping | `PONG` |
| `?` | current ramped speeds | `S <tick_ms> <m1> <m2> <m3> <m4>` |
| `ID` | firmware version and last reset cause | `ID rover-fw 1.0 F103C6 rst=<cause>` |
| `BOOT` | reboot into the ST bootloader (no BOOT0 jumper) | `OK` |
| `C ?` | print the settings | `C LIMIT=… RAMP=… FAIL=… PWM=… SIDE=… SIGN=…` |
| `C LIMIT <1-255>` | speed cap | `OK` |
| `C RAMP <1-255>` | speed step per 10 ms | `OK` |
| `C FAIL <100-5000>` | failsafe time in ms | `OK` |
| `C PWM <1100-20000>` | PWM frequency in Hz | `OK` |
| `C SIGN <motor> <1\|-1>` | flip a motor's direction | `OK` |
| `C SIDE <motor> <0\|1>` | 0 = left side, 1 = right side | `OK` |
| `C RESET` | restore the defaults | `OK` |

Anything else replies `ERR`. If `V` commands stop for the failsafe time (500 ms by default) the motors stop and the chip sends `FAILSAFE`. An independent watchdog (about 1 s) resets the chip if the firmware hangs. Settings live in RAM only, so send the `C` lines at start-up.

## Build

Needs CMake, Ninja and `arm-none-eabi-gcc` (tested with GCC 7.2). The HAL and CMSIS headers are included.

```bash
cmake -S Projects -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE="$PWD/Projects/cmake/gcc-arm-none-eabi.cmake" -DCMAKE_BUILD_TYPE=Release
cmake --build build
arm-none-eabi-objcopy -O binary build/rover-firmware.elf rover-firmware.bin
```

The image is about 12.6 KB of flash and 2.4 KB of RAM.

## Flash

With the wheels lifted, using a USB-UART adapter and the chip's built-in bootloader:

1. Disconnect the motor wires on PA9 and PA10, which share the bootloader pins.
2. Adapter RXD to PA9, TXD to PA10, ground to ground. Put BOOT0 on 1 and press reset, or send `BOOT` over the Jetson link.
3. `python -m stm32loader -p COM8 -f F1 -b 115200 -a 0x08000000 -e -w -v rover-firmware.bin`
4. BOOT0 back to 0, reconnect the motor wires, reset.

A debug probe on the SWD header (PA13, PA14) avoids the wire changes.

## Notes

- `rover-firmware.ioc` is the CubeMX project. It does not make CubeMX generate `SysTick_Handler` or the fault handlers, so they are defined in `Src/rover.c`. Delete them there if you turn generation on.
- `Src/rover.c` and `Inc/rover.h` hold all the application code; `main.c` only has three hooks in its USER CODE blocks.
- `RECOVERED_CONFIG.md` records the clock, timer and pin setup recovered from the original firmware.

## Status

Bench-tested with the wheels lifted: each motor alone, forward and spin commands, the failsafe, `ID`, and the `C` settings (16 kHz PWM, speed cap, direction flip and side swap all behave as described). `BOOT` is not yet tested on hardware. The Jetson side is not connected yet.
