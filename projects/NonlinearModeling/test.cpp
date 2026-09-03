#define OPTIM_ENABLE_EIGEN_WRAPPERS

#include <optim.hpp>
#include <cmath>
#include <utility>
#include "NLModeling.h"

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

constexpr static int NumParams = NLModeling::NumParams;
constexpr static int BatchSampleLen = 65536;
float testX[BatchSampleLen], targetY[BatchSampleLen];

constexpr static int WindowSize = 1024;
constexpr static int HopSize = 512;

NLModeling::NLModelParams nlParams;
NLModeling::NLModelProcess nlproc;
float y[BatchSampleLen];
float tmpre1[WindowSize];
float tmpim1[WindowSize];
float tmpre2[WindowSize];
float tmpim2[WindowSize];
float window[WindowSize];
float loss(float* params, int n)
{
	nlParams.VecToParams(params);
	nlproc.Init();
	nlproc.ProcessBlock(nlParams, testX, y, BatchSampleLen);

	float loss = 0;

	/*
	for (int i = 0; i < BatchSampleLen; ++i)
	{
		float d = y[i] - targetY[i];
		loss += d * d;
	}*/
	for (int i = 0; i < BatchSampleLen - WindowSize; i += HopSize)
	{
		for (int j = 0; j < WindowSize; ++j)
		{
			tmpre1[j] = y[i + j] * window[j];
			tmpim1[j] = 0;
			tmpre2[j] = targetY[i + j] * window[j];
			tmpim2[j] = 0;
		}
		FFT<WindowSize>(tmpre1, tmpim1, 0);
		FFT<WindowSize>(tmpre2, tmpim2, 0);
		for (int j = 1; j < WindowSize / 2; ++j)
		{
			float mag1 = sqrtf(tmpre1[j] * tmpre1[j] + tmpim1[j] * tmpim1[j]);
			float mag2 = sqrtf(tmpre2[j] * tmpre2[j] + tmpim2[j] * tmpim2[j]);
			float d = mag1 - mag2;

			float e = d * d * 0.01f;
			float e2 = e * e;
			float e4 = e2 * e2;
			float e8 = e4 * e4;
			float e16 = e8 * e8;
			loss += e16;
		}
	}
	return powf(loss, 1.0 / 16.0);
	//return loss;
}


void loss_wrapper(float* params, int n, float* out)
{
	*out = loss(params, n);
}
float get_gradient(float* params, float* grad, int n)
{
	std::fill(grad, grad + n, 0.0f);
	float loss_value = 0.0f;
	float dloss = 1.0f;
	__enzyme_autodiff(
		(void*)loss_wrapper,
		// params + gradient
		enzyme_dup,
		params, grad,
		// n ²»Çóµ¼
		enzyme_const,
		n,
		// loss output
		enzyme_dup,
		&loss_value, &dloss
	);
	return loss_value;
}

int iter = 0;
double objective(
	const Eigen::VectorXd& x,
	Eigen::VectorXd* grad_out,
	void* data)
{
	Eigen::VectorXf params = x.cast<float>();
	Eigen::VectorXf grad(x.size());
	float loss = get_gradient(params.data(), grad.data(), NumParams);
	if (grad_out)  *grad_out = grad.cast<double>();
	iter++;
	if (iter % 20 == 0)
		std::cout << "Iter" << iter << "  loss = " << loss << '\n';
	return loss;
}
float WindowFunc(float x)
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
	float t = 0;
	for (int i = 0; i < BatchSampleLen; ++i)
	{
		testX[i] = sin(t * 2.0 * M_PI) * 0.01;
		t += (float)i / BatchSampleLen;
	}
	srand(31415926);
	float initp[NumParams];
	for (int i = 0; i < NumParams; ++i)
		initp[i] = (rand() % 10000) / 10000.0 * (rand() % 2 ? 1 : -1);
	nlParams.VecToParams(initp);
	nlproc.Init();
	nlproc.ProcessBlock(nlParams, testX, targetY, BatchSampleLen);

	Eigen::VectorXd x(NumParams);
	for (int i = 0; i < NumParams; ++i)
		x[i] = (rand() % 10000) / 10000.0 * (rand() % 2 ? 1 : -1);


	optim::algo_settings_t settings;
	settings.gd_settings.method = 6;
	settings.gd_settings.par_step_size = 0.005;
	settings.gd_settings.par_adam_beta_1 = 0.9;
	settings.gd_settings.par_adam_beta_2 = 0.999;
	settings.iter_max = 1000;
	bool success = optim::gd(
		x,
		objective,
		NULL,
		settings
	);
	iter = 0;
	settings.iter_max = 1000000;
	success = optim::lbfgs(
		x,
		objective,
		nullptr,
		settings
	);
}
