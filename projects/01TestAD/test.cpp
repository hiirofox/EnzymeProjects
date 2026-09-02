#define OPTIM_ENABLE_EIGEN_WRAPPERS

#include <optim.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>


extern int enzyme_dup;
extern int enzyme_const;

template <typename Return, typename... Args>
Return __enzyme_autodiff(void*, Args...);


constexpr double PI = 3.14159265358979323846;
constexpr double SAMPLE_RATE = 8000.0;

constexpr int SIGNAL_N = 512;
constexpr int HARMONICS = 5;

/*
²ÎÊý£º

0      log(f0)

1~5    log(harmonic amplitude)

6      log(attack)
7      log(decay)
8      log(drive)

9      vibrato depth (Hz)
10     log(vibrato rate)

*/

constexpr int PARAM_COUNT = 11;


// ============================================================
// Complex
// ============================================================

struct Complex
{
    double re;
    double im;
};


inline Complex cadd(Complex a, Complex b)
{
    return {
        a.re + b.re,
        a.im + b.im
    };
}


inline Complex csub(Complex a, Complex b)
{
    return {
        a.re - b.re,
        a.im - b.im
    };
}


inline Complex cmul(Complex a, Complex b)
{
    return {
        a.re * b.re - a.im * b.im,
        a.re * b.im + a.im * b.re
    };
}


// ============================================================
// constexpr log2
// ============================================================

constexpr int ct_log2(int n)
{
    return (n <= 1)
        ? 0
        : 1 + ct_log2(n / 2);
}


// ============================================================
// bit reverse
// ============================================================

unsigned reverse_bits(
    unsigned x,
    int bits
)
{
    unsigned y = 0;

    for (int i = 0; i < bits; ++i)
    {
        y <<= 1;
        y |= x & 1u;
        x >>= 1;
    }

    return y;
}


// ============================================================
// Radix-2 FFT
// ============================================================

template<int N>
void fft(Complex* x)
{
    constexpr int LOG_N = ct_log2(N);

    for (unsigned i = 0; i < N; ++i)
    {
        const unsigned j =
            reverse_bits(i, LOG_N);

        if (j > i)
        {
            Complex temp = x[i];

            x[i] = x[j];
            x[j] = temp;
        }
    }


    for (int len = 2; len <= N; len *= 2)
    {
        const double angle =
            -2.0 * PI /
            static_cast<double>(len);

        const Complex wlen{
            std::cos(angle),
            std::sin(angle)
        };


        for (int i = 0; i < N; i += len)
        {
            Complex w{
                1.0,
                0.0
            };


            for (int j = 0; j < len / 2; ++j)
            {
                const Complex u =
                    x[i + j];

                const Complex v =
                    cmul(
                        x[i + j + len / 2],
                        w
                    );


                x[i + j] =
                    cadd(u, v);

                x[i + j + len / 2] =
                    csub(u, v);


                w =
                    cmul(w, wlen);
            }
        }
    }
}


// ============================================================
// Hann window
// ============================================================

template<int N>
double hann(int n)
{
    return
        0.5
        -
        0.5 *
        std::cos(
            2.0 * PI *
            static_cast<double>(n) /
            static_cast<double>(N - 1)
        );
}


// ============================================================
// DDSP synthesizer
//
// Harmonic oscillator
// + amplitude envelope
// + vibrato
// + tanh waveshaper
// ============================================================

double synth_sample(
    const double* p,
    int sample_index
)
{
    const double t =
        static_cast<double>(sample_index)
        / SAMPLE_RATE;


    const double f0 =
        std::exp(p[0]);


    const double attack =
        std::exp(p[6]);

    const double decay =
        std::exp(p[7]);

    const double drive =
        std::exp(p[8]);


    const double vibrato_depth =
        p[9];

    const double vibrato_rate =
        std::exp(p[10]);


    // --------------------------------------------------------
    // amplitude envelope
    // --------------------------------------------------------

    const double attack_env =
        1.0
        -
        std::exp(
            -t / attack
        );


    const double decay_env =
        std::exp(
            -t / decay
        );


    const double envelope =
        attack_env * decay_env;


    // --------------------------------------------------------
    // integrated sinusoidal vibrato
    //
    // instantaneous frequency:
    //
    // f(t) =
    // f0 + depth * sin(2*pi*rate*t)
    //
    // phase is its integral
    // --------------------------------------------------------

    const double vibrato_omega =
        2.0 * PI * vibrato_rate;


    const double vibrato_phase_term =
        vibrato_depth
        *
        (
            1.0
            -
            std::cos(
                vibrato_omega * t
            )
            )
        /
        vibrato_omega;


    const double phase =
        2.0 * PI
        *
        (
            f0 * t
            +
            vibrato_phase_term
            );


    // --------------------------------------------------------
    // harmonic oscillator bank
    // --------------------------------------------------------

    double signal = 0.0;


    for (int h = 0; h < HARMONICS; ++h)
    {
        const double amplitude =
            std::exp(
                p[1 + h]
            );


        const double harmonic =
            static_cast<double>(h + 1);


        signal +=
            amplitude
            *
            std::sin(
                harmonic * phase
            );
    }


    // --------------------------------------------------------
    // differentiable waveshaper
    // --------------------------------------------------------

    signal =
        std::tanh(
            drive * signal
        );


    return
        envelope * signal;
}


// ============================================================
// One STFT resolution
//
// log-power loss
// +
// spectral convergence
// ============================================================

template<int FFT_N, int HOP>
double stft_loss(
    const double* prediction,
    const double* target
)
{
    constexpr double EPS = 1e-8;

    double total_loss = 0.0;

    int frame_count = 0;


    for (
        int start = 0;
        start + FFT_N <= SIGNAL_N;
        start += HOP
        )
    {
        Complex pred_fft[FFT_N];
        Complex target_fft[FFT_N];


        for (int i = 0; i < FFT_N; ++i)
        {
            const double window =
                hann<FFT_N>(i);


            pred_fft[i].re =
                prediction[start + i]
                *
                window;

            pred_fft[i].im = 0.0;


            target_fft[i].re =
                target[start + i]
                *
                window;

            target_fft[i].im = 0.0;
        }


        fft<FFT_N>(pred_fft);
        fft<FFT_N>(target_fft);


        double log_loss = 0.0;

        double convergence_num = 0.0;
        double convergence_den = 0.0;


        for (int k = 0; k <= FFT_N / 2; ++k)
        {
            const double pred_power =
                pred_fft[k].re
                * pred_fft[k].re
                +
                pred_fft[k].im
                * pred_fft[k].im
                +
                EPS;


            const double target_power =
                target_fft[k].re
                * target_fft[k].re
                +
                target_fft[k].im
                * target_fft[k].im
                +
                EPS;


            // ------------------------------------------------
            // log spectrum
            // ------------------------------------------------

            const double pred_log =
                std::log(pred_power);

            const double target_log =
                std::log(target_power);


            const double log_error =
                pred_log
                -
                target_log;


            log_loss +=
                log_error
                *
                log_error;


            // ------------------------------------------------
            // spectral convergence
            // ------------------------------------------------

            const double pred_mag =
                std::sqrt(pred_power);

            const double target_mag =
                std::sqrt(target_power);


            const double mag_error =
                pred_mag
                -
                target_mag;


            convergence_num +=
                mag_error
                *
                mag_error;


            convergence_den +=
                target_mag
                *
                target_mag;
        }


        log_loss /=
            static_cast<double>(
                FFT_N / 2 + 1
                );


        const double convergence =
            std::sqrt(
                convergence_num
                /
                (
                    convergence_den
                    +
                    EPS
                    )
            );


        total_loss +=
            0.7 * log_loss
            +
            0.3 * convergence;


        ++frame_count;
    }


    return
        total_loss
        /
        static_cast<double>(frame_count);
}


// ============================================================
// DDSP loss
//
// multi-resolution STFT
// +
// small waveform loss
// ============================================================

double ddsp_loss(
    const double* params,
    const double* target
)
{
    double prediction[SIGNAL_N];


    for (int i = 0; i < SIGNAL_N; ++i)
    {
        prediction[i] =
            synth_sample(
                params,
                i
            );
    }


    // --------------------------------------------------------
    // waveform MSE
    // --------------------------------------------------------

    double waveform_loss = 0.0;


    for (int i = 0; i < SIGNAL_N; ++i)
    {
        const double error =
            prediction[i]
            -
            target[i];


        waveform_loss +=
            error * error;
    }


    waveform_loss /=
        static_cast<double>(
            SIGNAL_N
            );


    // --------------------------------------------------------
    // multi-resolution spectral loss
    // --------------------------------------------------------

    const double loss_64 =
        stft_loss<
        64,
        16
        >(
            prediction,
            target
        );


    const double loss_128 =
        stft_loss<
        128,
        32
        >(
            prediction,
            target
        );


    const double loss_256 =
        stft_loss<
        256,
        64
        >(
            prediction,
            target
        );


    // --------------------------------------------------------
    // final DDSP loss
    // --------------------------------------------------------

    return
        0.05 * waveform_loss
        +
        0.20 * loss_64
        +
        0.30 * loss_128
        +
        0.45 * loss_256;
}


// ============================================================
// Dataset
// ============================================================

struct Dataset
{
    std::vector<double> target;
};


// ============================================================
// Optim objective
// ============================================================

double objective(
    const optim::ColVec_t& params,
    optim::ColVec_t* gradient,
    void* user_data
)
{
    auto* data =
        static_cast<Dataset*>(
            user_data
            );


    if (gradient != nullptr)
    {
        gradient->resize(
            params.size()
        );

        gradient->setZero();


        __enzyme_autodiff<void>(
            reinterpret_cast<void*>(
                ddsp_loss
                ),

            enzyme_dup,
            params.data(),
            gradient->data(),

            enzyme_const,
            data->target.data()
        );
    }


    return
        ddsp_loss(
            params.data(),
            data->target.data()
        );
}


// ============================================================
// target synthesizer
// ============================================================

void generate_target(
    Dataset& data
)
{
    double p[PARAM_COUNT];


    p[0] =
        std::log(220.0);


    p[1] =
        std::log(0.80);

    p[2] =
        std::log(0.42);

    p[3] =
        std::log(0.22);

    p[4] =
        std::log(0.11);

    p[5] =
        std::log(0.055);


    p[6] =
        std::log(0.006);

    p[7] =
        std::log(0.18);

    p[8] =
        std::log(1.4);


    p[9] =
        3.5;


    p[10] =
        std::log(5.0);


    data.target.resize(
        SIGNAL_N
    );


    for (int i = 0; i < SIGNAL_N; ++i)
    {
        data.target[i] =
            synth_sample(
                p,
                i
            );
    }
}


// ============================================================
// print parameters
// ============================================================

void print_params(
    const optim::ColVec_t& p
)
{
    std::cout
        << "f0 = "
        << std::exp(p(0))
        << " Hz\n";


    for (int h = 0; h < HARMONICS; ++h)
    {
        std::cout
            << "harmonic "
            << h + 1
            << " = "
            << std::exp(
                p(1 + h)
            )
            << "\n";
    }


    std::cout
        << "attack = "
        << std::exp(p(6))
        << " s\n";


    std::cout
        << "decay = "
        << std::exp(p(7))
        << " s\n";


    std::cout
        << "drive = "
        << std::exp(p(8))
        << "\n";


    std::cout
        << "vibrato depth = "
        << p(9)
        << " Hz\n";


    std::cout
        << "vibrato rate = "
        << std::exp(p(10))
        << " Hz\n";
}


// ============================================================
// main
// ============================================================

int main()
{
    Dataset data;

    generate_target(data);


    // --------------------------------------------------------
    // intentionally incorrect initialization
    // --------------------------------------------------------

    optim::ColVec_t params(
        PARAM_COUNT
    );


    params(0) =
        std::log(205.0);


    params(1) =
        std::log(0.60);

    params(2) =
        std::log(0.30);

    params(3) =
        std::log(0.15);

    params(4) =
        std::log(0.08);

    params(5) =
        std::log(0.03);


    params(6) =
        std::log(0.015);

    params(7) =
        std::log(0.12);

    params(8) =
        std::log(1.0);


    params(9) =
        1.0;


    params(10) =
        std::log(4.0);


    // --------------------------------------------------------
    // initial loss
    // --------------------------------------------------------

    std::cout
        << "Initial loss: "
        << objective(
            params,
            nullptr,
            &data
        )
        << "\n\n";


    // --------------------------------------------------------
    // Enzyme gradient
    // --------------------------------------------------------

    optim::ColVec_t gradient;


    objective(
        params,
        &gradient,
        &data
    );


    std::cout
        << "Initial Enzyme gradient:\n";


    for (int i = 0; i < PARAM_COUNT; ++i)
    {
        std::cout
            << "grad["
            << i
            << "] = "
            << gradient(i)
            << "\n";
    }


    // --------------------------------------------------------
    // Adam
    // --------------------------------------------------------

    optim::algo_settings_t settings;


    settings.gd_settings.method =
        6;


    settings.gd_settings.par_step_size =
        0.003;


    settings.gd_settings.par_adam_beta_1 =
        0.9;


    settings.gd_settings.par_adam_beta_2 =
        0.999;


    settings.iter_max =
        10000;


    settings.grad_err_tol =
        1e-5;


    settings.rel_sol_change_tol =
        1e-8;


    // --------------------------------------------------------
    // optimize
    // --------------------------------------------------------

    const bool success =
        optim::gd(
            params,
            objective,
            &data,
            settings
        );


    // --------------------------------------------------------
    // result
    // --------------------------------------------------------

    std::cout
        << "\nOptimizer success: "
        << std::boolalpha
        << success
        << "\n";


    std::cout
        << "Final loss: "
        << objective(
            params,
            nullptr,
            &data
        )
        << "\n\n";


    std::cout
        << "Learned parameters:\n";


    print_params(params);


    // --------------------------------------------------------
    // compare waveform
    // --------------------------------------------------------

    std::cout
        << "\n"
        << "target\tprediction\n";


    for (int i = 0; i < 32; ++i)
    {
        const double prediction =
            synth_sample(
                params.data(),
                i
            );


        std::cout
            << data.target[i]
            << "\t"
            << prediction
            << "\n";
    }


    return 0;
}