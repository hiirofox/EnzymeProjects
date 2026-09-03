#pragma once
#include <cmath>
#include <cstdint>

void GenerateTestData(float* initX, float* targetY, int numSamples)
{
    if (!initX || !targetY || numSamples <= 0)
        return;

    constexpr float kFs = 48000.0f;   // 如果你的工程不是 48k，在这里改
    constexpr float kPi = 3.14159265358979323846f;
    constexpr float kTwoPi = 2.0f * kPi;

    struct Biquad
    {
        float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f;
        float a1 = 0.0f, a2 = 0.0f;
        float z1 = 0.0f, z2 = 0.0f;

        float Process(float x)
        {
            const float y = b0 * x + z1;
            z1 = b1 * x - a1 * y + z2;
            z2 = b2 * x - a2 * y;
            return y;
        }
    };

    auto SetNotch = [&](Biquad& q, float f, float Q)
        {
            const float w0 = kTwoPi * f / kFs;
            const float c = std::cos(w0);
            const float s = std::sin(w0);
            const float alpha = s / (2.0f * Q);
            const float a0 = 1.0f + alpha;

            q.b0 = 1.0f / a0;
            q.b1 = -2.0f * c / a0;
            q.b2 = 1.0f / a0;
            q.a1 = -2.0f * c / a0;
            q.a2 = (1.0f - alpha) / a0;
        };

    auto SetPeak = [&](Biquad& q, float f, float Q, float gainDb)
        {
            const float A = std::pow(10.0f, gainDb / 40.0f);
            const float w0 = kTwoPi * f / kFs;
            const float c = std::cos(w0);
            const float s = std::sin(w0);
            const float alpha = s / (2.0f * Q);

            const float b0 = 1.0f + alpha * A;
            const float b1 = -2.0f * c;
            const float b2 = 1.0f - alpha * A;

            const float a0 = 1.0f + alpha / A;
            const float a1 = -2.0f * c;
            const float a2 = 1.0f - alpha / A;

            q.b0 = b0 / a0;
            q.b1 = b1 / a0;
            q.b2 = b2 / a0;
            q.a1 = a1 / a0;
            q.a2 = a2 / a0;
        };

    auto SetHP = [&](Biquad& q, float f, float Q)
        {
            const float w0 = kTwoPi * f / kFs;
            const float c = std::cos(w0);
            const float s = std::sin(w0);
            const float alpha = s / (2.0f * Q);
            const float a0 = 1.0f + alpha;

            q.b0 = ((1.0f + c) * 0.5f) / a0;
            q.b1 = (-(1.0f + c)) / a0;
            q.b2 = ((1.0f + c) * 0.5f) / a0;
            q.a1 = (-2.0f * c) / a0;
            q.a2 = (1.0f - alpha) / a0;
        };

    auto SetLP = [&](Biquad& q, float f, float Q)
        {
            const float w0 = kTwoPi * f / kFs;
            const float c = std::cos(w0);
            const float s = std::sin(w0);
            const float alpha = s / (2.0f * Q);
            const float a0 = 1.0f + alpha;

            q.b0 = ((1.0f - c) * 0.5f) / a0;
            q.b1 = (1.0f - c) / a0;
            q.b2 = ((1.0f - c) * 0.5f) / a0;
            q.a1 = (-2.0f * c) / a0;
            q.a2 = (1.0f - alpha) / a0;
        };

    // ================================================================
    // 1. 生成测试干声
    //
    // 不再使用简单 sine sweep。
    //
    // 包含：
    //   - guitar-like power chord
    //   - 略微失谐的谐波
    //   - pluck transient
    //   - 跳频 probe
    //   - colored noise
    //   - amplitude modulation
    //   - silence/recovery
    //
    // 这些东西对听感模型通常比单个扫频更有价值。
    // ================================================================

    uint32_t rng = 0x31415926u;

    auto RandSigned = [&]() -> float
        {
            // deterministic xorshift32
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;

            return
                (float)((rng >> 8) & 0x00ffffffu)
                * (2.0f / 16777215.0f)
                - 1.0f;
        };

    const float roots[] =
    {
        82.4069f,     // E2
        98.0000f,     // G2
        110.0000f,    // A2
        123.4708f,    // B2
        146.8324f,    // D3
        164.8138f,    // E3
        196.0000f,    // G3
        220.0000f     // A3
    };

    // 故意不是规则 harmonic sequence。
    // 对 notch / Q / IMD 识别比较有帮助。
    const float probeFreqs[] =
    {
        73.0f,
        127.0f,
        211.0f,
        347.0f,
        563.0f,
        911.0f,
        1471.0f,
        2381.0f,
        3821.0f,
        5213.0f
    };

    float p1 = 0.0f;
    float p2 = 0.37f;
    float p3 = 1.11f;
    float p4 = 2.03f;

    float probePhase = 0.0f;
    float noiseLP = 0.0f;

    float rawPeak = 1.0e-12f;

    const int chordLen = (int)(0.230f * kFs);
    const int probeLen = (int)(0.137f * kFs);
    const int pluckLen = (int)(0.173f * kFs);
    const int macroLen = (int)(2.400f * kFs);

    for (int i = 0; i < numSamples; ++i)
    {
        const int chordIndex = (i / chordLen) & 7;
        const float root = roots[chordIndex];

        // 基频 + 五度 + 略失谐 octave + 近似三次谐波。
        // 略失谐很重要，会产生慢 beat 和大量 IMD 测试条件。
        const float f1 = root;
        const float f2 = root * 1.4983071f;
        const float f3 = root * 2.0030f;
        const float f4 = root * 2.9970f;

        p1 += kTwoPi * f1 / kFs;
        p2 += kTwoPi * f2 / kFs;
        p3 += kTwoPi * f3 / kFs;
        p4 += kTwoPi * f4 / kFs;

        if (p1 > kTwoPi) p1 -= kTwoPi;
        if (p2 > kTwoPi) p2 -= kTwoPi;
        if (p3 > kTwoPi) p3 -= kTwoPi;
        if (p4 > kTwoPi) p4 -= kTwoPi;

        const float chord =
            0.42f * std::sin(p1)
            + 0.28f * std::sin(p2)
            + 0.18f * std::sin(p3)
            + 0.10f * std::sin(p4);

        // ------------------------------------------------------------
        // 跳频 probe
        //
        // 相比纯 sine sweep，它会产生频率突然变化，
        // 对带记忆非线性、滤波器状态和瞬态识别比较有用。
        // ------------------------------------------------------------

        const int probeIndex = (i / probeLen) % 10;

        probePhase +=
            kTwoPi * probeFreqs[probeIndex] / kFs;

        if (probePhase > kTwoPi)
            probePhase -= kTwoPi;

        const float probe =
            0.13f * std::sin(probePhase);

        // ------------------------------------------------------------
        // Guitar-ish pick / pluck
        // ------------------------------------------------------------

        const int pp = i % pluckLen;

        const float pluckEnv =
            std::exp(
                -(float)pp /
                (0.043f * kFs));

        const float pluckPhase =
            kTwoPi * root * (float)pp / kFs;

        const float pluck =
            pluckEnv *
            (
                0.55f * std::sin(pluckPhase)
                + 0.27f * std::sin(2.01f * pluckPhase)
                + 0.12f * std::sin(3.98f * pluckPhase)
                );

        // ------------------------------------------------------------
        // 少量 colored noise
        //
        // 可以帮助模型识别：
        //   高频谐波
        //   intermodulation
        //   notch
        //   非线性的局部曲率
        // ------------------------------------------------------------

        const float white = RandSigned();

        noiseLP =
            0.985f * noiseLP +
            0.015f * white;

        const float coloredNoise =
            0.055f * noiseLP
            + 0.018f * (white - noiseLP);

        // 慢速幅度变化。
        //
        // 同一个波形会不断经过不同的 nonlinear operating point。
        const float slowAM =
            0.72f +
            0.28f *
            std::sin(
                kTwoPi *
                0.71f *
                (float)i / kFs);

        float x =
            slowAM *
            (
                0.62f * chord
                + 0.40f * pluck
                + probe
                )
            + coloredNoise;

        // ------------------------------------------------------------
        // 偶尔接近静音
        //
        // 让模型学：
        //   bias recovery
        //   envelope release
        //   filter ringing
        //   memory decay
        // ------------------------------------------------------------

        const int mp = i % macroLen;

        if (mp > (int)(2.18f * kFs))
            x *= 0.015f;

        initX[i] = x;

        const float ax = std::fabs(x);

        if (ax > rawPeak)
            rawPeak = ax;
    }

    // 输入峰值强制控制到约 1e-2。
    const float inScale =
        0.0095f / rawPeak;

    for (int i = 0; i < numSamples; ++i)
        initX[i] *= inScale;


    // ================================================================
    // 2. 怪异的固定 Volterra guitar pedal
    //
    // signal flow:
    //
    // HP
    //   ↓
    // presence EQ
    //   ↓
    // high internal gain
    //   ↓
    // finite-memory Volterra
    //   ↓
    // envelope / sag
    //   ↓
    // asymmetric diode
    //   ↓
    // nested tanh
    //   ↓
    // polynomial wavefolder
    //   ↓
    // thermal bias memory
    //   ↓
    // notch
    //   ↓
    // notch
    //   ↓
    // high-Q resonance
    //   ↓
    // cab-ish LP
    //
    // ================================================================

    Biquad preHP;
    Biquad prePresence;

    Biquad notch1;
    Biquad notch2;

    Biquad postPeak;
    Biquad cabLP;

    SetHP(
        preHP,
        52.0f,
        0.707f);

    SetPeak(
        prePresence,
        1180.0f,
        1.15f,
        +5.5f);

    // 很窄的第一个 notch
    SetNotch(
        notch1,
        735.0f,
        7.5f);

    // 稍宽一点的第二个 notch
    SetNotch(
        notch2,
        2860.0f,
        4.2f);

    // 类似 cocked-wah / 某些 distortion pedal 的鼻音峰。
    SetPeak(
        postPeak,
        1580.0f,
        3.8f,
        +7.0f);

    // cab / speaker-ish rolloff
    SetLP(
        cabLP,
        7600.0f,
        0.72f);


    // ================================================================
    // Volterra memory
    // ================================================================

    constexpr int M = 32;

    float mem[M] = {};

    int memPos = 0;

    float env = 0.0f;
    float thermal = 0.0f;
    float feedback = 0.0f;

    float targetPeak = 1.0e-12f;


    for (int i = 0; i < numSamples; ++i)
    {
        // ------------------------------------------------------------
        // Pre-EQ
        // ------------------------------------------------------------

        float u =
            preHP.Process(initX[i]);

        u =
            prePresence.Process(u);

        // initX 只有 ~0.01，所以进入虚拟 pedal 前提高 gain。
        //
        // 内部通常落在 +-0.5 ~ +-1 附近。
        u *= 78.0f;

        // 防止以后你改 input generator 后突然把 Volterra 炸飞。
        if (u > 1.6f) u = 1.6f;
        if (u < -1.6f) u = -1.6f;


        // ------------------------------------------------------------
        // delay line
        // ------------------------------------------------------------

        mem[memPos] = u;

        auto D = [&](int delay) -> float
            {
                int idx =
                    memPos - delay;

                if (idx < 0)
                    idx += M;

                return mem[idx];
            };


        const float d0 = D(0);
        const float d2 = D(2);
        const float d3 = D(3);
        const float d5 = D(5);
        const float d7 = D(7);
        const float d11 = D(11);
        const float d13 = D(13);
        const float d19 = D(19);


        // ============================================================
        // 怪 Volterra
        //
        // 不是单纯：
        //
        // y = ax + bx² + cx³
        //
        // 而是：
        //
        // x[n]x[n-5]
        // x[n-2]x[n-11]
        // x[n]x[n-3]x[n-7]
        // ...
        //
        // 因此非线性本身带 temporal memory。
        // ============================================================

        float v =
            1.00f * d0
            - 0.24f * d3
            + 0.16f * d7
            - 0.09f * d13
            + 0.045f * d19

            // second order
            + 0.31f * d0 * d0
            - 0.23f * d3 * d3
            + 0.22f * d0 * d5
            - 0.17f * d2 * d11
            + 0.09f * d7 * d13

            // third order
            + 0.20f * d0 * d0 * d0
            - 0.16f * d0 * d3 * d7
            + 0.11f * d2 * d5 * d13
            - 0.055f * d3 * d3 * d11

            // fourth order
            + 0.045f *
            d0 * d0 *
            d5 * d5

            - 0.025f *
            d2 * d3 *
            d7 * d11;


        // ============================================================
        // Envelope dependent sag
        // ============================================================

        const float au =
            std::fabs(u);

        // attack 快，release 慢
        const float envCoef =
            (au > env)
            ? 0.91f
            : 0.99925f;

        env =
            envCoef * env +
            (1.0f - envCoef) * au;

        // 输入越大，后级 drive 略微掉下去。
        //
        // 类似：
        // power sag / compressor / transistor operating point movement
        const float sag =
            1.0f /
            (1.0f +
                0.42f * env * env);


        // ============================================================
        // 一个非常弱的 state feedback
        //
        // 不是为了做 delay，
        // 只是让失真具有一点粘滞性。
        // ============================================================

        v +=
            0.075f * feedback;

        v *= sag;


        // ============================================================
        // 套娃非线性 #1
        //
        // asymmetric diode-like
        // ============================================================

        const float bias =
            0.105f +
            0.035f * thermal;

        // 减掉静态偏置导致的 DC。
        const float diode0 =
            std::tanh(
                1.85f * bias);

        const float diode =
            std::tanh(
                1.85f *
                (v + bias))
            - diode0;


        // ============================================================
        // 套娃非线性 #2
        //
        // tanh(
        //     x
        //     + tanh(x)
        //     + x²
        // )
        //
        // 偶数次项故意加入，做 asymmetric distortion。
        // ============================================================

        const float n1 =
            std::tanh(
                1.55f *
                (
                    diode
                    + 0.28f *
                    std::tanh(
                        3.10f * diode)

                    - 0.11f *
                    diode * diode
                    ));


        // ============================================================
        // 套娃非线性 #3
        //
        // polynomial wave-folder-ish
        // 再丢进 tanh
        // ============================================================

        const float n1_2 =
            n1 * n1;

        const float n1_3 =
            n1_2 * n1;

        const float n1_5 =
            n1_3 * n1_2;

        float n2 =
            std::tanh(
                1.28f *
                (
                    n1
                    + 0.34f * n1_3
                    - 0.10f * n1_5
                    ));


        // 再叠一个更硬的小分支。
        n2 +=
            0.085f *
            std::tanh(
                5.2f *
                (
                    n1
                    - 0.18f * n1_3
                    ));


        // ============================================================
        // thermal / bias memory
        //
        // 非线性输出长期平均值反过来移动工作点。
        // ============================================================

        thermal =
            0.99955f * thermal +
            0.00045f * n2;

        n2 -=
            0.10f * thermal;


        feedback =
            0.90f * feedback +
            0.10f * n2;


        // ============================================================
        // Weird EQ / notch section
        // ============================================================

        float y =
            notch1.Process(n2);

        y =
            notch2.Process(y);

        y =
            postPeak.Process(y);

        y =
            cabLP.Process(y);


        // ------------------------------------------------------------
        // 留一点相对直接的 signal path。
        //
        // 这样 pick transient 不会被完全抹掉。
        // ------------------------------------------------------------

        y =
            0.93f * y +
            0.075f * u;


        // 最终 safety saturation。
        //
        // 即使前面参数乱改，一般也不会产生 INF。
        y =
            std::tanh(
                1.12f * y);


        targetY[i] = y;

        const float ay =
            std::fabs(y);

        if (ay > targetPeak)
            targetPeak = ay;


        // M == 32，所以可以直接 bit mask。
        memPos =
            (memPos + 1) &
            (M - 1);
    }


    // ================================================================
    // 3. 输出尺度
    //
    // 保证 targetY peak ~= 0.98
    // ================================================================

    const float outScale =
        (targetPeak > 1.0e-9f)
        ? (0.98f / targetPeak)
        : 1.0f;

    for (int i = 0; i < numSamples; ++i)
        targetY[i] *= outScale;
}