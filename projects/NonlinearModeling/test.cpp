#define OPTIM_ENABLE_EIGEN_WRAPPERS

#include <future>
#include <optim.hpp>
#include <cmath>
#include <utility>
#include "NLModeling.h"
#include "DatGen.h"
#include "wavfile.h"

extern int enzyme_dup;
extern int enzyme_const;
template <typename... Args>
void __enzyme_autodiff(Args...);

template<int N>
void FFT(float* re, float* im, int inv)
{
	static_assert((N & (N - 1)) == 0, "N must be power of 2");
	for (int i = 1, j = 0; i < N; ++i) {
		int bit = N >> 1;
		for (; j & bit; bit >>= 1) j ^= bit;
		j ^= bit;
		if (i < j) {
			std::swap(re[i], re[j]);
			std::swap(im[i], im[j]);
		}
	}
	for (int len = 2; len <= N; len <<= 1) {
		float ang = 2.0f * 3.14159265358979323846f / len
			* (inv ? 1.0f : -1.0f);
		float wlen_re = std::cos(ang);
		float wlen_im = std::sin(ang);
		for (int i = 0; i < N; i += len) {
			float wr = 1.0f;
			float wi = 0.0f;
			for (int j = 0; j < len / 2; ++j) {
				int a = i + j;
				int b = a + len / 2;
				float vr = re[b] * wr - im[b] * wi;
				float vi = re[b] * wi + im[b] * wr;
				float ur = re[a];
				float ui = im[a];
				re[a] = ur + vr;
				im[a] = ui + vi;
				re[b] = ur - vr;
				im[b] = ui - vi;
				float t = wr;
				wr = t * wlen_re - wi * wlen_im;
				wi = t * wlen_im + wi * wlen_re;
			}
		}
	}
	if (inv) {
		for (int i = 0; i < N; ++i) {
			re[i] /= N;
			im[i] /= N;
		}
	}
}
#define SelectedNL NLModelingGRU
constexpr static int NumParams = SelectedNL::NumParams;
constexpr static int BatchSampleLen = 48000 * 2;
float testX[BatchSampleLen], targetY[BatchSampleLen];

constexpr static int WindowSize = 1024;
constexpr static int HopSize = 512;
float window[WindowSize];//Harris-Blackman
void loss_wrapper(
	float* params,
	int n, int startPos, int batchLen,
	float* outLoss,
	float* specPeak,
	float* outRMS,
	float* outMax)
{
	SelectedNL::NLModelParams& nlParams = *new SelectedNL::NLModelParams;
	SelectedNL::NLModelProcess& nlproc = *new SelectedNL::NLModelProcess;
	float* y = new float[batchLen];
	float* tmpre1 = new float[WindowSize];
	float* tmpim1 = new float[WindowSize];
	float* tmpre2 = new float[WindowSize];
	float* tmpim2 = new float[WindowSize];

	nlParams.VecToParams(params);
	nlproc.Init();
	nlproc.ProcessBlock(nlParams, testX, y, batchLen);

	float sumd8 = 0;
	float sumt8 = 0;
	for (int i = 0; i < batchLen - WindowSize; i += HopSize)
	{
		for (int j = 0; j < WindowSize; ++j)
		{
			tmpre1[j] = y[i + j] * window[j];
			tmpim1[j] = 0.0f;
			tmpre2[j] = targetY[i + j + startPos] * window[j];
			tmpim2[j] = 0.0f;
		}
		FFT<WindowSize>(tmpre1, tmpim1, 0);
		FFT<WindowSize>(tmpre2, tmpim2, 0);
		for (int j = 1; j < WindowSize / 2 - 1; ++j)
		{
			//float mag1 = sqrtf(tmpre1[j] * tmpre1[j] + tmpim1[j] * tmpim1[j] + 1e-12f);
			//float mag2 = sqrtf(tmpre2[j] * tmpre2[j] + tmpim2[j] * tmpim2[j] + 1e-12f);
			float mag1 = tmpre1[j] * tmpre1[j] + tmpim1[j] * tmpim1[j];
			float mag2 = tmpre2[j] * tmpre2[j] + tmpim2[j] * tmpim2[j];

			float d = mag1 - mag2;
			float d2 = d * d * 0.001f;
			float d4 = d2 * d2;
			float d8 = d4 * d4;
			sumd8 += d8;

			float t = mag2;
			float t2 = t * t * 0.001f;
			float t4 = t2 * t2;
			float t8 = t4 * t4;
			sumt8 += t8;
		}
	}
	sumd8 = powf(sumd8, 1.0 / 8.0);
	sumt8 = powf(sumt8, 1.0 / 8.0);
	float specSoftPeak = sumd8 / sumt8;

	float errSq = 0.0f;
	float errMax = 0.0f;
	float targetSq = 0.0f;
	float targetMax = 0.0f;

	float err8 = 0.0;
	float target8 = 0.0;
	for (int i = 0; i < batchLen; ++i)
	{
		const float t = targetY[i + startPos];
		const float d = y[i] - t;
		errSq += d * d;
		errMax = std::max(errMax, fabsf(d));
		targetSq += t * t;
		targetMax = std::max(targetMax, fabsf(t));

		float d2 = d * d * 0.0001;
		float d4 = d2 * d2;
		float d8 = d4 * d4;
		float t2 = t * t * 0.0001;
		float t4 = t2 * t2;
		float t8 = t4 * t4;
		err8 += d8;
		target8 += t8;
	}
	const float invN = 1.0f / static_cast<float>(batchLen);
	const float errRMS = sqrtf(errSq * invN);
	const float targetRMS = sqrtf(targetSq * invN);
	constexpr float eps = 1e-12f;
	const float rms = errRMS / (targetRMS + eps);
	const float max = errMax / (targetMax + eps);

	const float errR8 = powf(err8, 1.0 / 8.0);
	const float targetR8 = powf(target8, 1.0 / 8.0);
	const float timeSoftPeak = errR8 / (targetR8 + eps);

	float loss = rms * 50.0 + timeSoftPeak * 100.0 + specSoftPeak * 50.0;

	*outLoss = loss;
	*specPeak = specSoftPeak;
	*outRMS = rms;
	*outMax = max;
}

std::tuple<float, float, float, float > get_gradient(float* params, float* grad, int n,
	int startPos, int batchLen)
{
	std::fill(grad, grad + n, 0.0f);

	float lossValue = 0.0f;
	float specPeakValue = 0.0f;
	float rmsValue = 0.0f;
	float maxValue = 0.0f;

	float dLoss = 1.0f;
	float dSpecPeak = 1.0f;
	float dRMS = 0.0f;
	float dMax = 0.0f;
	__enzyme_autodiff(
		(void*)loss_wrapper,
		// params -> 求 dLoss/dparams
		enzyme_dup,
		params, grad,
		// n
		enzyme_const,
		n,
		enzyme_const,
		startPos, batchLen,
		// loss
		enzyme_dup,
		&lossValue, &dLoss,
		// specPeak
		enzyme_dup,
		&specPeakValue, &dSpecPeak,
		// RMS
		enzyme_dup,
		&rmsValue, &dRMS,
		// Max
		enzyme_dup,
		&maxValue, &dMax
	);
	return {
		lossValue,
		specPeakValue,
		rmsValue,
		maxValue
	};
}

int iter = 0;
double objective(
	const Eigen::VectorXd& x,
	Eigen::VectorXd* grad_out,
	void* data)
{
	Eigen::VectorXf params = x.cast<float>();
	Eigen::VectorXf grad(x.size());

	int startPos = 0, batchLen = BatchSampleLen;
	auto [loss, specPeak, rms, max] =
		get_gradient(params.data(), grad.data(), NumParams,
			startPos, batchLen);//性能项

	if (grad_out)  *grad_out = grad.cast<double>();
	if (iter % 20 == 0)
	{
		printf("Iter%5d loss=%03.5f specPeak=%03.5f rms=%03.5f max=%03.5f\n",
			iter, loss, specPeak, rms, max);
	}
	iter++;
	return loss;
}
float WindowFunc(float x)//nice window near Harries-Blackman
{
	float z = x * x;
	float q = std::fma(0.34950587, z, -1.27143058);
	q = std::fma(q, z, 2.32561793);
	q = std::fma(q, z, -2.33733549);
	q = std::fma(q, z, 1.0);
	q = q * q;
	return std::fma(-z, q, q);
}
int main()
{
	for (int i = 0; i < WindowSize; ++i)
	{
		window[i] = WindowFunc((float)i / WindowSize * 2.0 - 1.0);
	}
	GenerateTestData(testX, targetY, BatchSampleLen);

	float directParams[NumParams];
	SelectedNL::NLModelParams::InitVecDirect(directParams);
	Eigen::VectorXd x(NumParams);
	for (int i = 0; i < NumParams; ++i)
		x[i] = directParams[i];

	optim::algo_settings_t settings;
	settings.gd_settings.method = 6;
	settings.gd_settings.par_step_size = 0.0000025;
	settings.gd_settings.par_adam_beta_1 = 0.9;
	settings.gd_settings.par_adam_beta_2 = 0.999;
	settings.iter_max = 2000000;
	bool success = optim::gd(x, objective, NULL, settings);

	iter = 0;
	settings.iter_max = 1000000;
	success = optim::lbfgs(x, objective, nullptr, settings);
}
