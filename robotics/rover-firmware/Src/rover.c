/*
 * rover.c - UART command interface and motor control for the rover (STM32F103C6, HAL).
 *
 * USART1 on PB6 (TX) / PB7 (RX), 115200 8N1, text lines ending in '\n', case-insensitive:
 *   V <left> <right>   side speeds, -255..255 (repeat faster than the failsafe time)
 *   T <motor> [speed]  bench test: one motor for TEST_MS, motor 1..4, default TEST_SPEED
 *   S                  stop everything now
 *   P                  reply PONG
 *   ?                  reply "S <tick_ms> <m1> <m2> <m3> <m4>" (current ramped speeds)
 *   ID                 reply firmware version and the last reset cause
 *   BOOT               reboot into the ST system bootloader (no BOOT0 jumper needed)
 *   C ?                print the settings
 *   C LIMIT <1-255>    speed cap            C RAMP <1-255>    speed step per 10 ms
 *   C FAIL <100-5000>  failsafe in ms       C PWM <1100-20000> PWM frequency in Hz
 *   C SIGN <m> <1|-1>  flip a motor         C SIDE <m> <0|1>  0 = left, 1 = right
 *   C RESET            restore the defaults (settings are kept in RAM only)
 *
 * Each motor is one PWM pin pair on a BTS7960: forward pin and reverse pin, never both on.
 */

#include "rover.h"
#include "main.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern TIM_HandleTypeDef  htim1;
extern TIM_HandleTypeDef  htim2;
extern UART_HandleTypeDef huart1;

TIM_HandleTypeDef htim3;   /* motor 2 only; set up here, not in CubeMX, so PA9/PA10 stay free for the flasher */


/* ---- Config -------------------------------------------------------------- */
#define FW_VERSION      "1.1"
#define NUM_MOTORS      4
#define DEF_LIMIT       255     /* cap on any commanded speed */
#define DEF_RAMP        10      /* speed units added or removed per RAMP_PERIOD_MS */
#define DEF_FAILSAFE_MS 500     /* stop if V commands stop arriving */
#define DEF_PWM_ARR     65535   /* timer period of the original firmware: about 1.1 kHz */
#define RAMP_PERIOD_MS  10
#define TEST_MS         1500    /* length of a T pulse */
#define TEST_SPEED      60
#define TIMER_HZ        72000000UL
#define RX_LINE_MAX     32
#define RX_RING_SIZE    64
#define MAX_TOKENS      4
#define IWDG_RELOAD     625     /* 40 kHz LSI / 64 = 625 Hz, so about 1 s */
#define BOOT_MAGIC_ADDR 0x200023F0u   /* free RAM below the stack; survives a soft reset */
#define BOOT_MAGIC      0xB007B007u
#define SYSMEM_BASE     0x1FFFF000u

typedef enum { SIDE_LEFT = 0, SIDE_RIGHT = 1 } Side;

typedef struct {
    TIM_HandleTypeDef *tim_fwd;
    uint32_t           ch_fwd;
    TIM_HandleTypeDef *tim_rev;
    uint32_t           ch_rev;
} MotorPins;

/* First pin of each pair is forward. Wheel positions confirmed with the T test, 2026-11-17. */
static const MotorPins pins[NUM_MOTORS] = {
    { &htim1, TIM_CHANNEL_1, &htim3, TIM_CHANNEL_4 },   /* motor 1 front left:  PA8  fwd, PB1  rev */
    { &htim3, TIM_CHANNEL_3, &htim1, TIM_CHANNEL_4 },   /* motor 2 back left:   PB0  fwd, PA11 rev */
    { &htim2, TIM_CHANNEL_3, &htim2, TIM_CHANNEL_4 },   /* motor 3 front right: PA2  fwd, PA3  rev */
    { &htim2, TIM_CHANNEL_1, &htim2, TIM_CHANNEL_2 },   /* motor 4 back right:  PA0  fwd, PA1  rev */
};


/* ---- State --------------------------------------------------------------- */
static Side     m_side[NUM_MOTORS] = { SIDE_LEFT, SIDE_LEFT, SIDE_RIGHT, SIDE_RIGHT };
static int8_t   m_sign[NUM_MOTORS] = { 1, 1, 1, 1 };
static int16_t  speed_limit        = DEF_LIMIT;
static int16_t  ramp_step          = DEF_RAMP;
static uint16_t failsafe_ms        = DEF_FAILSAFE_MS;
static uint32_t pwm_arr            = DEF_PWM_ARR;

static int16_t  target[NUM_MOTORS];
static int16_t  current[NUM_MOTORS];

static uint8_t  v_active;
static uint32_t v_deadline;

static uint8_t  test_active;
static uint8_t  test_motor;
static uint32_t test_end;

static uint8_t           rx_byte;
static uint8_t           rx_ring[RX_RING_SIZE];
static volatile uint8_t  rx_head;
static volatile uint8_t  rx_tail;

static const char *reset_cause = "?";


/* ---- Helpers ------------------------------------------------------------- */
static void reply(const char *s)
{
    HAL_UART_Transmit(&huart1, (uint8_t *)s, (uint16_t)strlen(s), 20);
}


static int16_t clamp_speed(int v)
{
    if (v >  speed_limit) return  speed_limit;
    if (v < -speed_limit) return -speed_limit;
    return (int16_t)v;
}


static int parse_int(const char *s, int *out)
{
    char *end;
    long  v = strtol(s, &end, 10);

    if (end == s || *end != '\0') return 0;
    *out = (int)v;
    return 1;
}


static const char *read_reset_cause(void)
{
    uint32_t    csr = RCC->CSR;
    const char *c;

    if      (csr & RCC_CSR_IWDGRSTF) c = "IWDG";
    else if (csr & RCC_CSR_WWDGRSTF) c = "WWDG";
    else if (csr & RCC_CSR_SFTRSTF)  c = "SOFT";
    else if (csr & RCC_CSR_PORRSTF)  c = "POWER";
    else if (csr & RCC_CSR_PINRSTF)  c = "PIN";
    else                             c = "OTHER";

    RCC->CSR |= RCC_CSR_RMVF;
    return c;
}


/* Turn off the pin that is not driving first, so both are never on together */
static void motor_apply(int idx, int16_t speed)
{
    const MotorPins *m    = &pins[idx];
    uint32_t         duty = (uint32_t)(speed < 0 ? -speed : speed) * pwm_arr / 255u;

    if (speed > 0) {
        __HAL_TIM_SET_COMPARE(m->tim_rev, m->ch_rev, 0);
        __HAL_TIM_SET_COMPARE(m->tim_fwd, m->ch_fwd, duty);
    } else if (speed < 0) {
        __HAL_TIM_SET_COMPARE(m->tim_fwd, m->ch_fwd, 0);
        __HAL_TIM_SET_COMPARE(m->tim_rev, m->ch_rev, duty);
    } else {
        __HAL_TIM_SET_COMPARE(m->tim_fwd, m->ch_fwd, 0);
        __HAL_TIM_SET_COMPARE(m->tim_rev, m->ch_rev, 0);
    }
}


static void stop_all_now(void)
{
    for (int i = 0; i < NUM_MOTORS; i++) {
        target[i]  = 0;
        current[i] = 0;
        motor_apply(i, 0);
    }
    v_active    = 0;
    test_active = 0;
}


static void set_side(Side side, int16_t speed)
{
    for (int i = 0; i < NUM_MOTORS; i++) {
        if (m_side[i] == side) target[i] = (int16_t)(speed * m_sign[i]);
    }
}


static void ramp_update(void)
{
    static uint32_t last;
    uint32_t        now = HAL_GetTick();

    if (now - last < RAMP_PERIOD_MS) return;
    last = now;

    for (int i = 0; i < NUM_MOTORS; i++) {
        int16_t diff = (int16_t)(target[i] - current[i]);

        if (diff > ramp_step)       diff =  ramp_step;
        else if (diff < -ramp_step) diff = -ramp_step;

        if (diff) {
            current[i] = (int16_t)(current[i] + diff);
            motor_apply(i, current[i]);
        }
    }
}


static void set_defaults(void)
{
    for (int i = 0; i < NUM_MOTORS; i++) {
        m_side[i] = (i < 2) ? SIDE_LEFT : SIDE_RIGHT;
        m_sign[i] = 1;
    }
    speed_limit = DEF_LIMIT;
    ramp_step   = DEF_RAMP;
    failsafe_ms = DEF_FAILSAFE_MS;
    pwm_arr     = DEF_PWM_ARR;
    __HAL_TIM_SET_AUTORELOAD(&htim1, pwm_arr);
    __HAL_TIM_SET_AUTORELOAD(&htim2, pwm_arr);
    __HAL_TIM_SET_AUTORELOAD(&htim3, pwm_arr);
}


/* ---- Commands ------------------------------------------------------------ */
static void cmd_config(char **t, int n)
{
    int  a, b;
    char buf[96];

    if (n == 2 && strcmp(t[1], "?") == 0) {
        snprintf(buf, sizeof buf, "C LIMIT=%d RAMP=%d FAIL=%u PWM=%lu SIDE=%d%d%d%d SIGN=%d,%d,%d,%d\n",
                 speed_limit, ramp_step, (unsigned)failsafe_ms, (unsigned long)(TIMER_HZ / (pwm_arr + 1)),
                 m_side[0], m_side[1], m_side[2], m_side[3],
                 m_sign[0], m_sign[1], m_sign[2], m_sign[3]);
        reply(buf);
        return;
    }

    stop_all_now();   /* never change settings while moving */

    if (n == 2 && strcmp(t[1], "RESET") == 0) {
        set_defaults();
        reply("OK\n");
    }
    else if (n == 3 && parse_int(t[2], &a)) {
        if      (strcmp(t[1], "LIMIT") == 0 && a >= 1    && a <= 255)   { speed_limit = (int16_t)a;  reply("OK\n"); }
        else if (strcmp(t[1], "RAMP")  == 0 && a >= 1    && a <= 255)   { ramp_step   = (int16_t)a;  reply("OK\n"); }
        else if (strcmp(t[1], "FAIL")  == 0 && a >= 100  && a <= 5000)  { failsafe_ms = (uint16_t)a; reply("OK\n"); }
        else if (strcmp(t[1], "PWM")   == 0 && a >= 1100 && a <= 20000) {
            pwm_arr = TIMER_HZ / (uint32_t)a - 1;
            __HAL_TIM_SET_AUTORELOAD(&htim1, pwm_arr);
            __HAL_TIM_SET_AUTORELOAD(&htim2, pwm_arr);
            __HAL_TIM_SET_AUTORELOAD(&htim3, pwm_arr);
            reply("OK\n");
        }
        else reply("ERR\n");
    }
    else if (n == 4 && parse_int(t[2], &a) && parse_int(t[3], &b) && a >= 1 && a <= NUM_MOTORS) {
        if      (strcmp(t[1], "SIGN") == 0 && (b == 1 || b == -1)) { m_sign[a - 1] = (int8_t)b; reply("OK\n"); }
        else if (strcmp(t[1], "SIDE") == 0 && (b == 0 || b == 1))  { m_side[a - 1] = (Side)b;   reply("OK\n"); }
        else reply("ERR\n");
    }
    else {
        reply("ERR\n");
    }
}


static void handle_line(char *line)
{
    char *tok[MAX_TOKENS];
    int   n = 0;

    for (char *p = strtok(line, " \t"); p && n < MAX_TOKENS; p = strtok(NULL, " \t")) tok[n++] = p;
    if (n == 0) return;

    if (strcmp(tok[0], "V") == 0) {
        int l, r;

        if (n != 3 || !parse_int(tok[1], &l) || !parse_int(tok[2], &r)) { reply("ERR\n"); return; }

        test_active = 0;
        set_side(SIDE_LEFT,  clamp_speed(l));
        set_side(SIDE_RIGHT, clamp_speed(r));
        v_active   = (l != 0 || r != 0);
        v_deadline = HAL_GetTick() + failsafe_ms;
        reply("OK\n");
    }
    else if (strcmp(tok[0], "T") == 0) {
        int m, s = TEST_SPEED;

        if (n < 2 || !parse_int(tok[1], &m) || m < 1 || m > NUM_MOTORS) { reply("ERR\n"); return; }
        if (n == 3 && !parse_int(tok[2], &s)) { reply("ERR\n"); return; }

        stop_all_now();
        test_motor         = (uint8_t)(m - 1);
        target[test_motor] = (int16_t)(clamp_speed(s) * m_sign[test_motor]);
        test_end           = HAL_GetTick() + TEST_MS;
        test_active        = 1;
        reply("OK\n");
    }
    else if (strcmp(tok[0], "S") == 0) {
        stop_all_now();
        reply("OK\n");
    }
    else if (strcmp(tok[0], "P") == 0) {
        reply("PONG\n");
    }
    else if (strcmp(tok[0], "?") == 0) {
        char buf[48];

        snprintf(buf, sizeof buf, "S %lu %d %d %d %d\n", (unsigned long)HAL_GetTick(),
                 current[0], current[1], current[2], current[3]);
        reply(buf);
    }
    else if (strcmp(tok[0], "ID") == 0) {
        char buf[48];

        snprintf(buf, sizeof buf, "ID rover-fw %s F103C6 rst=%s\n", FW_VERSION, reset_cause);
        reply(buf);
    }
    else if (strcmp(tok[0], "C") == 0) {
        cmd_config(tok, n);
    }
    else if (strcmp(tok[0], "BOOT") == 0) {
        stop_all_now();
        reply("OK\n");
        *(volatile uint32_t *)BOOT_MAGIC_ADDR = BOOT_MAGIC;
        NVIC_SystemReset();
    }
    else {
        reply("ERR\n");
    }
}


static void read_commands(void)
{
    static char    line[RX_LINE_MAX];
    static uint8_t len;
    static uint8_t overflow;

    while (rx_tail != rx_head) {
        char c = (char)rx_ring[rx_tail];

        rx_tail = (uint8_t)((rx_tail + 1) % RX_RING_SIZE);

        if (c == '\n' || c == '\r') {
            if (overflow)  reply("ERR\n");
            else if (len)  { line[len] = '\0'; handle_line(line); }
            len      = 0;
            overflow = 0;
        } else if (len < RX_LINE_MAX - 1) {
            line[len++] = (char)toupper((unsigned char)c);
        } else {
            overflow = 1;
        }
    }
}


/* ---- Public -------------------------------------------------------------- */
/* First thing in main(): after a BOOT command, jump into the ST system bootloader */
void rover_boot_check(void)
{
    volatile uint32_t *magic = (volatile uint32_t *)BOOT_MAGIC_ADDR;

    if (*magic != BOOT_MAGIC) return;

    *magic = 0;
    __set_MSP(*(volatile uint32_t *)SYSMEM_BASE);
    ((void (*)(void))(*(volatile uint32_t *)(SYSMEM_BASE + 4)))();
}


/* TIM3 CH3/CH4 on PB0/PB1 (default pin map, no remap): motor 2, same clock and period as TIM1/TIM2 */
static void tim3_init(void)
{
    GPIO_InitTypeDef   gpio = {0};
    TIM_OC_InitTypeDef oc   = {0};

    __HAL_RCC_TIM3_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();

    htim3.Instance               = TIM3;
    htim3.Init.Prescaler         = 0;
    htim3.Init.CounterMode       = TIM_COUNTERMODE_UP;
    htim3.Init.Period            = DEF_PWM_ARR;
    htim3.Init.ClockDivision     = TIM_CLOCKDIVISION_DIV1;
    htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
    if (HAL_TIM_PWM_Init(&htim3) != HAL_OK) Error_Handler();

    oc.OCMode     = TIM_OCMODE_PWM1;
    oc.Pulse      = 0;
    oc.OCPolarity = TIM_OCPOLARITY_HIGH;
    oc.OCFastMode = TIM_OCFAST_DISABLE;
    if (HAL_TIM_PWM_ConfigChannel(&htim3, &oc, TIM_CHANNEL_3) != HAL_OK) Error_Handler();
    if (HAL_TIM_PWM_ConfigChannel(&htim3, &oc, TIM_CHANNEL_4) != HAL_OK) Error_Handler();

    gpio.Pin   = GPIO_PIN_0 | GPIO_PIN_1;
    gpio.Mode  = GPIO_MODE_AF_PP;
    gpio.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOB, &gpio);
}


void rover_init(void)
{
    reset_cause = read_reset_cause();

    tim3_init();

    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_1);
    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_4);
    HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_3);
    HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_4);
    HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_1);
    HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_2);
    HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_3);
    HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_4);

    stop_all_now();

    {   /* pull PB7 (USART1 RX) high so a loose or unplugged wire reads as idle, not noise */
        GPIO_InitTypeDef rx = {0};

        rx.Pin  = GPIO_PIN_7;
        rx.Mode = GPIO_MODE_INPUT;
        rx.Pull = GPIO_PULLUP;
        HAL_GPIO_Init(GPIOB, &rx);
    }
    HAL_UART_Receive_IT(&huart1, &rx_byte, 1);

    DBGMCU->CR |= DBGMCU_CR_DBG_IWDG_STOP;   /* freeze the watchdog while a debugger halts the core */
    IWDG->KR  = 0x5555;
    IWDG->PR  = 4;                           /* prescaler /64 */
    IWDG->RLR = IWDG_RELOAD;
    IWDG->KR  = 0xAAAA;
    IWDG->KR  = 0xCCCC;
}


#ifdef SELFTEST
/* Self-test build (-DSELFTEST): no commands needed. SELFTEST_LEAD_MS after reset each motor runs forward, then
 * reverse, one pulse per SELFTEST_STEP_MS, in the order front-left, back-left, front-right, back-right. */
#define SELFTEST_LEAD_MS  15000u
#define SELFTEST_STEP_MS  5000u
#define SELFTEST_SPEED    120

static void selftest_poll(uint32_t now)
{
    static int32_t last = -1;
    int32_t        step;

    if (now < SELFTEST_LEAD_MS) return;

    step = (int32_t)((now - SELFTEST_LEAD_MS) / SELFTEST_STEP_MS);
    if (step >= 2 * NUM_MOTORS || step == last) return;

    last = step;
    stop_all_now();
    test_motor         = (uint8_t)(step / 2);
    target[test_motor] = (int16_t)(((step % 2) ? -SELFTEST_SPEED : SELFTEST_SPEED) * m_sign[test_motor]);
    test_end           = now + TEST_MS;
    test_active        = 1;
}
#endif


void rover_poll(void)
{
    uint32_t now = HAL_GetTick();

    IWDG->KR = 0xAAAA;

    read_commands();
#ifdef SELFTEST
    selftest_poll(now);
#endif

    if (test_active && (int32_t)(now - test_end) >= 0) {
        target[test_motor] = 0;
        test_active        = 0;
    }

    if (v_active && (int32_t)(now - v_deadline) > 0) {
        stop_all_now();
        reply("FAILSAFE\n");
    }

    ramp_update();
}


/* Crash path: motor outputs off, then wait for the watchdog to reset the chip */
void rover_fault(void)
{
    TIM1->BDTR &= ~TIM_BDTR_MOE;
    TIM2->CCER  = 0;
    TIM3->CCER  = 0;
    for (;;) {}
}


/* ---- UART callbacks ------------------------------------------------------ */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance != USART1) return;

    uint8_t next = (uint8_t)((rx_head + 1) % RX_RING_SIZE);

    if (next != rx_tail) {
        rx_ring[rx_head] = rx_byte;
        rx_head          = next;
    }
    HAL_UART_Receive_IT(&huart1, &rx_byte, 1);
}


void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART1) HAL_UART_Receive_IT(&huart1, &rx_byte, 1);
}


/* ---- Exception handlers -------------------------------------------------- */
/* The .ioc does not make CubeMX generate these, and without SysTick_Handler the HAL tick
 * never advances and the chip locks up at the first 1 ms tick. If you ever tick "generate
 * handler" for them in CubeMX, delete these to avoid duplicate definitions. */
void SysTick_Handler(void)    { HAL_IncTick(); }
void NMI_Handler(void)        { rover_fault(); }
void HardFault_Handler(void)  { rover_fault(); }
void MemManage_Handler(void)  { rover_fault(); }
void BusFault_Handler(void)   { rover_fault(); }
void UsageFault_Handler(void) { rover_fault(); }
