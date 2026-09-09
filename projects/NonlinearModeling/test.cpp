#define OPTIM_ENABLE_EIGEN_WRAPPERS

#include <future>
#include <algorithm>
#include <queue>
#include <ensmallen.hpp>
#include <Eigen/Dense>
//#include <optim.hpp>
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
#define SelectedNL NLModeling3

constexpr static int NumParams = SelectedNL::NumParams;
constexpr static int bootSize = 48000 / 480;//留一些采样供响应稳定
constexpr static int delaySample = SelectedNL::NLModelProcess::GetTargetDelaySample();
//延迟一些采样让模型好优化，而不是学预测
//典型值：灰盒1sample，gru 2sample
int BatchSampleLen = 65536;
//float* testX, * targetY;
std::vector<float> testX, targetY;

constexpr static int WindowSize = 1024;
constexpr static int HopSize = 512;
constexpr static int AliasTestSize = 64;

float window[WindowSize];//Harris-Blackman
void loss_wrapper(
	float* params,
	int n, int startPos, int batchLen,
	const float* testX,
	const float* targetY,
	float* outLoss,
	float* specPeak,
	float* outRMS,
	float* outMax,
	float* outAlias,
	float* outStruct)
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
	nlproc.ProcessBlock(nlParams, &testX[startPos], y, batchLen);

	float sumd8 = 0;
	float sumt8 = 0;
	for (int i = bootSize; i < batchLen - WindowSize; i += HopSize)
	{
		for (int j = 0; j < WindowSize; ++j)
		{
			tmpre1[j] = y[i + j] * window[j];
			tmpim1[j] = 0.0f;
			tmpre2[j] = targetY[i + j + startPos - delaySample] * window[j];
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

			float d = logf(mag1 + 1e-2) - logf(mag2 + 1e-2);
			float d2 = d * d * 0.000001;
			sumd8 += d2;

			/*
			float d = mag1 - mag2;
			float d2 = d * d * 0.0001f;
			float d4 = d2 * d2;
			float d8 = d4 * d4;
			sumd8 += d8;
			float t = mag2;
			float t2 = t * t * 0.0001f;
			float t4 = t2 * t2;
			float t8 = t4 * t4;
			sumt8 += t8;
			*/
		}
	}
	//sumd8 = powf(sumd8, 1.0 / 8.0);
	//sumt8 = powf(sumt8, 1.0 / 8.0);
	//float specSoftPeak = sumd8 / (sumt8 + 1e-3);
	float specSoftPeak = sumd8;

	float errSq = 0.0f;
	float errMax = 0.0f;
	float targetSq = 0.0f;
	float targetMax = 0.0f;

	float err8 = 0.0;
	float target8 = 0.0;
	for (int i = bootSize; i < batchLen; ++i)
	{
		int targIdx = i + startPos - delaySample;
		float vty = targetY[targIdx] - 0.85 * targetY[targIdx - 1];//hp
		float vy = y[i] - 0.85 * y[i - 1];//hp
		//float vty = targetY[targIdx];
		//float vy = y[i];
		const float t = vty;
		const float d = vy - vty;
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
	const float invN = 1.0f / static_cast<float>(batchLen - bootSize);
	const float errRMS = sqrtf(errSq * invN);
	const float targetRMS = sqrtf(targetSq * invN);
	constexpr float eps = 1e-3f;
	const float rms = errRMS / (targetRMS + eps);
	const float max = errMax / (targetMax + eps);
	const float errR8 = powf(err8, 1.0 / 8.0);
	const float targetR8 = powf(target8, 1.0 / 8.0);
	const float timeSoftPeak = errR8 / (targetR8 + eps);

	//aliasing analyze
	float aTotalEnergy = 0;
	float aliasEnergy = 0;
	for (int i = 2; i < AliasTestSize / 2 - 1; ++i)
	{
		nlproc.Init();
		for (int j = 0; j < AliasTestSize; ++j)
		{
			float x = (float)j / AliasTestSize;
			//float w = 0.5 - 0.5 * std::cos(2.0 * 3.1415926535 * x);
			float w = 1.0;
			tmpre2[j] = std::sin(2.0 * 3.1415926535 * x * i) * w * 0.25;
		}
		nlproc.ProcessBlock(nlParams, tmpre2, tmpre1, AliasTestSize);//boot
		nlproc.ProcessBlock(nlParams, tmpre2, tmpre1, AliasTestSize);
		for (int j = 0; j < AliasTestSize; ++j)
		{
			float x = (float)j / AliasTestSize;
			//float w = 0.5 - 0.5 * std::cos(2.0 * 3.1415926535 * x);
			float w = 1.0;
			tmpre1[j] *= w;
			tmpim1[j] = 0;
		}
		FFT<AliasTestSize>(tmpre1, tmpim1, 0);
		for (int j = 1; j < AliasTestSize / 2; ++j)
		{
			float mag = tmpre1[j] * tmpre1[j] + tmpim1[j] * tmpim1[j];
			aTotalEnergy += mag;
			if (j < i) aliasEnergy += mag;
		}
	}
	float cleanEnergy = aTotalEnergy - aliasEnergy;
	if (cleanEnergy < 0)cleanEnergy = 0;
	float aliasloss = aliasEnergy / (cleanEnergy + 1e-4) * 100.0;

	float structloss = nlproc.GetSructureLoss(nlParams);

	//loss eval
	//float loss = rms * 50.0 + timeSoftPeak * 100.0 + specSoftPeak * 50.0;
	//float loss = rms * 10.0 + timeSoftPeak * 10.0 + specSoftPeak * 180.0;
	float loss = rms * 10.0 +
		timeSoftPeak * 10.0 +
		aliasloss * 10.0 +
		specSoftPeak * 170.0 +
		structloss * 10.0;
	//float loss = specSoftPeak * 200.0;
	//float loss = timeSoftPeak * 200.0;
	//float loss = aliasloss * 200.0;

	*outLoss = loss;
	*specPeak = specSoftPeak;
	*outRMS = rms;
	*outMax = max;
	*outAlias = aliasloss;
	*outStruct = structloss;
}

std::tuple<float, float, float, float, float, float > get_gradient(const float* params, float* grad, int n,
	int startPos, int batchLen)
{
	std::fill(grad, grad + n, 0.0f);

	float lossValue = 0.0f;
	float specPeakValue = 0.0f;
	float rmsValue = 0.0f;
	float maxValue = 0.0f;
	float aliasValue = 0.0f;
	float structValue = 0.0f;

	float dLoss = 1.0f;
	float dSpecPeak = 0.0f;
	float dRMS = 0.0f;
	float dMax = 0.0f;
	float dAlias = 0.0f;
	float dStruct = 0.0f;
	__enzyme_autodiff(
		(void*)loss_wrapper,
		enzyme_dup, params, grad,
		enzyme_const, n,
		enzyme_const, startPos,
		enzyme_const, batchLen,
		enzyme_const, testX.data(),
		enzyme_const, targetY.data(),
		enzyme_dup, &lossValue, &dLoss,
		enzyme_dup, &specPeakValue, &dSpecPeak,
		enzyme_dup, &rmsValue, &dRMS,
		enzyme_dup, &maxValue, &dMax,
		enzyme_dup, &aliasValue, &dAlias,
		enzyme_dup, &structValue, &dStruct
	);
	return {
		lossValue,
		specPeakValue,
		rmsValue,
		maxValue,
		aliasValue,
		structValue
	};
}

const int NumTasks = 12;
int numTrainBlocks = 1;
std::vector<int> blockStart;
std::vector<int> blockLen;
class ADThreadPool
{
public:
	using Metrics = std::tuple<float, float, float, float, float, float>;
	struct TaskResult
	{
		Metrics metrics;
		Eigen::VectorXf grad;
	};
private:
	std::array<std::thread, NumTasks> pool;
	std::array<TaskResult, NumTasks> results;
	std::array<int, NumTasks> starts{};
	std::array<int, NumTasks> lengths{};
	std::array<bool, NumTasks> pending{};
	std::array<bool, NumTasks> done{};
	std::mutex mutex;
	std::condition_variable cv;
	bool stop = false;
	const float* params = nullptr;
	void ADTask(int id)
	{
		while (true)
		{
			int startPos, batchLen;
			const float* p;
			{
				std::unique_lock lock(mutex);
				cv.wait(lock, [&] { return stop || pending[id]; });
				if (stop) return;
				startPos = starts[id];
				batchLen = lengths[id];
				p = params;
				pending[id] = false;
			}
			auto& result = results[id];
			result.metrics = get_gradient(p, result.grad.data(), NumParams, startPos, batchLen);

			{
				std::lock_guard lock(mutex);
				done[id] = true;
			}
			cv.notify_all();
		}
	}
public:
	ADThreadPool()
	{
		for (auto& r : results)
			r.grad = Eigen::VectorXf::Zero(NumParams);
		for (int i = 0; i < NumTasks; ++i)
			pool[i] = std::thread(&ADThreadPool::ADTask, this, i);
	}
	~ADThreadPool()
	{
		{
			std::lock_guard lock(mutex);
			stop = true;
		}
		cv.notify_all();

		for (auto& t : pool)
			if (t.joinable()) t.join();
	}
	void SetParams(const float* p)
	{
		params = p;
	}
	void AddTask(int id, int startPos, int batchLen)
	{
		{
			std::lock_guard lock(mutex);
			starts[id] = startPos;
			lengths[id] = batchLen;
			done[id] = false;
			pending[id] = true;
		}
		cv.notify_all();
	}
	TaskResult GetResult(int id)
	{
		std::unique_lock lock(mutex);
		cv.wait(lock, [&] { return done[id]; });
		return results[id];
	}
};
//ADThreadPool adPool;
int iter = 0;
float bestLoss = 999999999;
float smoothLoss = 200.0;
float bestParams[SelectedNL::NumParams];
int newScoreFlag = 0;
std::string saveParamsFile = "";
void SaveParams(float* params, int NumParams)
{
	FILE* pf = fopen(saveParamsFile.c_str(), "w");
	fprintf(pf, "%d\n", NumParams);
	for (int i = 0; i < NumParams; ++i)
		fprintf(pf, "%.8f,", params[i]);
	fclose(pf);
}
int ReadParams(float* params, int NumParams)
{
	FILE* pf = fopen(saveParamsFile.c_str(), "r");
	if (!pf)return 0;
	int n, tmp;
	fscanf(pf, "%d", &n);
	for (int i = 0; i < NumParams; ++i) fscanf(pf, "%f,", &params[i]);
	fclose(pf);
	return 1;
}
double objective(const Eigen::VectorXd& x, Eigen::VectorXd* grad_out, void* data)
{
	Eigen::VectorXf params = x.cast<float>();
	float sumLoss = 0.0f;
	float sumSpecPeak = 0.0f;
	float sumRms = 0.0f;
	float sumMax = 0.0f;
	float sumAlias = 0.0f;
	float sumStruct = 0.0f;
	Eigen::VectorXf grad = Eigen::VectorXf::Zero(NumParams);

	/*
	adPool.SetParams(params.data());
	for (int i = 0; i < NumTasks; ++i)
		adPool.AddTask(i, blockStart[i], blockLen[i]);
	for (int i = 0; i < NumTasks; ++i)
	{
		auto result = adPool.GetResult(i);
		auto [loss, specPeak, rms, max,alias,struct] = result.metrics;
		sumLoss += loss;
		sumSpecPeak += specPeak;
		sumRms += rms;
		sumMax += max;
		sumAlias += alias;
		grad += result.grad;
	}
	sumLoss /= NumTasks;
	sumSpecPeak /= NumTasks;
	sumRms /= NumTasks;
	sumMax /= NumTasks;
	grad /= NumTasks;
	*/
	//int selectBlockID = rand() % numTrainBlocks;
	int selectBlockID = 0;
	auto [loss, specPeak, rmsValue, maxValue, aliasValue, structValue] =
		get_gradient(params.data(), grad.data(), NumParams, blockStart[selectBlockID], blockLen[selectBlockID]);

	//if (iter == 0)smoothLoss = loss;
	//if (!std::isinf(loss) && !std::isnan(loss))
	//	smoothLoss += 0.0125 * (loss - smoothLoss);
	//sumLoss = smoothLoss;
	sumLoss = loss;
	sumSpecPeak = specPeak;
	sumRms = rmsValue;
	sumMax = maxValue;
	sumAlias = aliasValue;
	sumStruct = structValue;
	if (grad_out)
		*grad_out = grad.cast<double>();

	if (sumLoss < bestLoss)
	{
		bestLoss = sumLoss;
		for (int i = 0; i < SelectedNL::NumParams; ++i)
			bestParams[i] = params[i];
		newScoreFlag = 1;
	}
	if (iter % 5 == 0)
	{
		printf("Iter%5d loss=%03.3f spec=%03.3f rms=%03.3f max=%03.3f alias=%03.3f struct:%03.3f %s\n",
			iter, sumLoss, sumSpecPeak, sumRms, sumMax, sumAlias, sumStruct, newScoreFlag ? "(NEW!)" : "");
		if (newScoreFlag)
		{
			SaveParams(params.data(), SelectedNL::NumParams);
		}
		newScoreFlag = 0;
	}
	++iter;
	return sumLoss;
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
int SegmentBlock(std::vector<float>& samples, int numSamples, std::vector<int>& start, std::vector<int>& len)
{
	int blockSize = numSamples / numTrainBlocks;
	for (int i = 0; i < numTrainBlocks; ++i)
	{
		printf("block:%d:%d (%.2fs)\n", blockSize * i, blockSize, blockSize / 48000.0);
		int startPos = blockSize * i;
		start.push_back(startPos);
		len.push_back(blockSize);
		for (int j = startPos, k = 0; j < startPos + bootSize; ++j, ++k)
		{
			float x = (float)k / bootSize;
			samples[j] *= WindowFunc(x - 1.0);//给数据集一个缓慢上升的窗
		}
	}
	return blockSize;
}

class EnsmallenObjective
{
public:
	size_t NumFunctions()
	{
		return 1;
	}

	void Shuffle()
	{
	}

	double EvaluateWithGradient(const arma::mat& x, arma::mat& grad)
	{
		Eigen::VectorXd xEigen(NumParams);
		Eigen::VectorXd gradEigen(NumParams);

		for (int i = 0; i < NumParams; ++i)
			xEigen[i] = x[i];

		double loss = objective(xEigen, &gradEigen, nullptr);

		grad.set_size(NumParams, 1);
		for (int i = 0; i < NumParams; ++i)
			grad[i] = gradEigen[i];

		return loss;
	}

	double EvaluateWithGradient(
		const arma::mat& x,
		const size_t begin,
		arma::mat& grad,
		const size_t batchSize)
	{
		return EvaluateWithGradient(x, grad);
	}
};

WavReader wr;
int main()
{
	for (int i = 0; i < WindowSize; ++i)
	{
		window[i] = WindowFunc((float)i / WindowSize * 2.0 - 1.0);
	}

	//testX.resize(BatchSampleLen);
	//targetY.resize(BatchSampleLen);
	//GenerateTestData(testX.data(), targetY.data(), BatchSampleLen);

	//std::string root = "/home/hiirofox/TestEnzyme/projects/NonlinearModeling/builds/";
	std::string root = "";

	std::ifstream f(root + "tasks.txt");
	int tasks = 0, maxiter = 1; float minloss = 0;
	f >> tasks >> minloss >> maxiter; f.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	std::vector<std::string> targetsName(tasks);
	std::vector<int> okflags(tasks, 0);
	printf("tasks:%d\nMaxIter:%d\nminLoss:%.5f\ntargets:\n", tasks, maxiter, minloss);
	for (int i = 0; i < tasks; ++i) {
		std::getline(f, targetsName[i]);
		if (!targetsName[i].empty() && targetsName[i].back() == '\r')
			targetsName[i].pop_back();
		std::cout << "\"" << targetsName[i] << "\"\n";
	}

	printf("\n---TASKS START---\n");

	for (;;) for (int taskid = 0; taskid < tasks; ++taskid)
	{
		if (okflags[taskid])continue;

		wr.OpenWAV(root + "nd-input.wav");//input.wav
		//std::string target = "nd-d10-t05";//target
		std::string target = targetsName[taskid];
		BatchSampleLen = wr.GetNumSamples();
		printf("wav NumSamples:%d\n", BatchSampleLen);
		testX.resize(BatchSampleLen);
		targetY.resize(BatchSampleLen);
		wr.ReadBlockMono(testX.data(), BatchSampleLen);
		wr.OpenWAV(root + target + ".wav");
		saveParamsFile = "bestloss-nl3-" + target + ".txt";

		wr.ReadBlockMono(targetY.data(), BatchSampleLen);
		float xrms = 0;
		float yrms = 0;
		float yAvgEnergy = 0;
		for (int i = 0; i < BatchSampleLen; ++i)
		{
			testX[i] *= 1.0;
			targetY[i] *= 1.0 / 1.414213562;
			xrms += testX[i] * testX[i] * 0.01;
			yrms += targetY[i] * targetY[i] * 0.01;
		}
		yAvgEnergy = yrms / BatchSampleLen / 0.0001;
		SegmentBlock(testX, BatchSampleLen, blockStart, blockLen);

		xrms = sqrtf(xrms / 0.0001);
		yrms = sqrtf(yrms / 0.0001);
		printf("boot sample:%d(%.2fs)\n", bootSize, (float)bootSize / 48000.0);
		printf("read ok. xrms=%.5f yrms=%.5f\n", xrms, yrms);

		//open file
		float directParams[NumParams];
		int result = ReadParams(directParams, NumParams);
		if (!result)SelectedNL::NLModelParams::InitVecDirect(directParams);

		arma::vec x(NumParams);
		for (int i = 0; i < NumParams; ++i)
			bestParams[i] = x[i] = directParams[i];

		EnsmallenObjective objectiveFunction;
		float lrstart = 0.001;
		ens::Adam adam;
		adam.StepSize() = lrstart;
		adam.BatchSize() = 1;
		adam.Beta1() = 0.9;
		adam.Beta2() = 0.999;
		adam.Epsilon() = 1e-8;
		adam.MaxIterations() = maxiter;
		adam.Tolerance() = 0.0;
		adam.Shuffle() = false;
		ens::L_BFGS lbfgs;
		lbfgs.MaxIterations() = maxiter;

		/*
		float adamloss = bestLoss;
		iter = 0;
		adam.Optimize(objectiveFunction, x);
		for (int i = 0; i < NumParams; ++i) x[i] = bestParams[i];
		printf("adam loss: %.3f->%.3f (%.3f%%)\n", adamloss, bestLoss, (adamloss - bestLoss) / adamloss * 100.0);
		*/

		float lbfgsloss = bestLoss;
		iter = 0;
		lbfgs.Optimize(objectiveFunction, x);
		for (int i = 0; i < NumParams; ++i) x[i] = bestParams[i];
		printf("lbfgs loss: %.3f->%.3f (%.3f%%)\n", lbfgsloss, bestLoss, (lbfgsloss - bestLoss) / lbfgsloss * 100.0);

		if (bestLoss < minloss) okflags[taskid] = 1;
		int allokflag = 1;
		for (int i = 0; i < tasks; ++i)
			if (!okflags[i])allokflag = 0;
		if (allokflag) goto done;
	}
done:
	//goto有什么不好的
}
