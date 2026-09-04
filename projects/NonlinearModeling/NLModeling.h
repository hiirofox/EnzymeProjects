#pragma once

#include <math.h>
#include <vector>

namespace NLModeling
{
	constexpr static int NumLayers = 4;
	constexpr static int NLOrder = 4;
	constexpr static int FiltOrder = 4;
	constexpr static int NumParams = NumLayers * NLOrder * (FiltOrder * 2 + 1);
	struct NLModelParams
	{
		float ks[NumLayers][NLOrder][FiltOrder];
		float gs[NumLayers][NLOrder][FiltOrder + 1];

		constexpr static float GScale = 200.0f;
		constexpr static float kScale = 0.975f;
		static inline float MapG(float x)
		{
			return GScale * std::tanh(x);
		}
		static inline float UnmapG(float x)
		{
			return std::atanh(x / GScale);
		}
		static void InitVecDirect(float* out)
		{
			int p = 0;
			for (int layer = 0; layer < NumLayers; ++layer)
			{
				for (int nl = 0; nl < NLOrder; ++nl)
				{
					for (int i = 0; i < FiltOrder; ++i)
						out[p++] = 0.0f;

					for (int i = 0; i < FiltOrder + 1; ++i)
					{
						if (nl == 0 && i == FiltOrder)
							out[p++] = UnmapG(1.0f);
						else
							out[p++] = UnmapG(1e-6f);
					}
				}
			}
		}
		void ParamsToVec(float* out) const
		{
			int p = 0;
			for (int layer = 0; layer < NumLayers; ++layer)
			{
				for (int nl = 0; nl < NLOrder; ++nl)
				{
					for (int i = 0; i < FiltOrder; ++i)
						out[p++] = std::atanh(ks[layer][nl][i] / kScale);
					for (int i = 0; i < FiltOrder + 1; ++i)
						out[p++] = UnmapG(gs[layer][nl][i]);
				}
			}
		}
		void VecToParams(const float* in)
		{
			int p = 0;
			for (int layer = 0; layer < NumLayers; ++layer)
			{
				for (int nl = 0; nl < NLOrder; ++nl)
				{
					for (int i = 0; i < FiltOrder; ++i)
						ks[layer][nl][i] = std::tanh(in[p++]) * kScale;
					for (int i = 0; i < FiltOrder + 1; ++i)
						gs[layer][nl][i] = MapG(in[p++]);
				}
			}
		}
	};

	template<int i, int Order, typename Sample>
	inline std::tuple<Sample, Sample> ProcessLattice(Sample x, Sample* z, Sample* k, Sample* g)
	{
		if constexpr (i >= Order) return { x,x * g[i] };
		else
		{
			Sample a = x - k[i] * z[i];
			auto [pass, total] =
				ProcessLattice<i + 1, Order, Sample>(a, z, k, g);
			Sample y = a * k[i] + z[i];
			z[i] = pass;
			return { y,total + y * g[i] };
		}
	}

	class NLModelProcess
	{
	private:
		float zs[NumLayers][NLOrder][FiltOrder];
	public:
		void Init()
		{
			for (int i = 0; i < NumLayers; ++i)
				for (int j = 0; j < NLOrder; ++j)
					for (int k = 0; k < FiltOrder; ++k)
						zs[i][j][k] = 0;
		}

		inline float NonlinearChebyshev(float x, float& x0, float& x1, float k)
		{
			float nextx = 2.0 * x * x1 - x0;
			x0 = x1, x1 = nextx;
			return x0;
		}
		inline float NonlinearLegendre(float x, float& x0, float& x1, float k)
		{
			float nextx = ((2.0 * k + 1.0) * x * x1 - k * x0) / (k + 1.0);
			x0 = x1, x1 = nextx;
			return x0;
		}
		inline float NonlinearHermite(float x, float& x0, float& x1, float k)
		{
			float nextx = x * x1 - k * x0;
			x0 = x1, x1 = nextx;
			return x0;
		}
		inline float NonlinearMy(float x, float& x0, float& x1, float k)
		{
			x0 = x1 / (1.0 + fabsf(x1));
			x1 *= x;
			return x0;
		}
		inline float NonlinearSimple(float x, float& x0, float& x1, float k)
		{
			x0 *= x;
			return x0;
		}

		template<typename Sample>
		static inline Sample Clip(Sample x, Sample lo, Sample hi)
		{
			return x < lo ? lo : (x > hi ? hi : x);
		}

		void ProcessBlock(NLModelParams& p, const float* in, float* out, int NumSamples)
		{
			for (int i = 0; i < NumSamples; ++i)
			{
				float x = in[i];
				for (int n = 0; n < NumLayers; ++n)
				{
					float y = 0;
					float x0 = 1.0, x1 = x;
					for (int j = 0; j < NLOrder; ++j)
					{
						float nlout = NonlinearChebyshev(x, x0, x1, j + 1);
						auto [pass, total] = ProcessLattice<0, FiltOrder, float>(nlout, zs[n][j], p.ks[n][j], p.gs[n][j]);
						y += total;
					}
					x = Clip(y, -1.0f, 1.0f);
				}
				out[i] = x;
			}
		}
	};
}

namespace NLModeling2
{
	constexpr static int NumLayers = 8;
	constexpr static int NLOrder = 4;
	constexpr static int FiltOrder = 4;
	constexpr static int NumParams =
		NumLayers * NLOrder * (FiltOrder * 2 + 1)
		+ NumLayers
		+ 1
		+ NumLayers * NLOrder;

	struct NLModelParams
	{
		float ks[NumLayers][NLOrder][FiltOrder];
		float gs[NumLayers][NLOrder][FiltOrder + 1];

		float pass[NumLayers];
		float gain;
		float sp[NumLayers][NLOrder];

		constexpr static float GScale = 200.0f;
		constexpr static float kScale = 0.975f;

		static inline float MapG(float x)
		{
			return GScale * std::tanh(x);
		}
		static inline float UnmapG(float x)
		{
			return std::atanh(x / GScale);
		}

		static void InitVecDirect(float* out)
		{
			int p = 0;
			for (int layer = 0; layer < NumLayers; ++layer)
			{
				for (int nl = 0; nl < NLOrder; ++nl)
				{
					for (int i = 0; i < FiltOrder; ++i)
						out[p++] = 0.0f;

					for (int i = 0; i < FiltOrder + 1; ++i)
					{
						if (nl == 0 && i == FiltOrder)
							out[p++] = UnmapG(1.0f);
						else
							out[p++] = UnmapG(1e-6f);
					}
				}
			}

			for (int layer = 0; layer < NumLayers; ++layer)
				out[p++] = 0.0f;

			out[p++] = 1.0f;

			for (int layer = 0; layer < NumLayers; ++layer)
				for (int nl = 0; nl < NLOrder; ++nl)
					out[p++] = 1.0f;
		}

		void ParamsToVec(float* out) const
		{
			int p = 0;
			for (int layer = 0; layer < NumLayers; ++layer)
			{
				for (int nl = 0; nl < NLOrder; ++nl)
				{
					for (int i = 0; i < FiltOrder; ++i)
						out[p++] = std::atanh(ks[layer][nl][i] / kScale);

					for (int i = 0; i < FiltOrder + 1; ++i)
						out[p++] = UnmapG(gs[layer][nl][i]);
				}
			}

			for (int layer = 0; layer < NumLayers; ++layer)
				out[p++] = pass[layer];

			out[p++] = gain;

			for (int layer = 0; layer < NumLayers; ++layer)
				for (int nl = 0; nl < NLOrder; ++nl)
					out[p++] = sp[layer][nl];
		}

		void VecToParams(const float* in)
		{
			int p = 0;
			for (int layer = 0; layer < NumLayers; ++layer)
			{
				for (int nl = 0; nl < NLOrder; ++nl)
				{
					for (int i = 0; i < FiltOrder; ++i)
						ks[layer][nl][i] = std::tanh(in[p++]) * kScale;

					for (int i = 0; i < FiltOrder + 1; ++i)
						gs[layer][nl][i] = MapG(in[p++]);
				}
			}

			for (int layer = 0; layer < NumLayers; ++layer)
				pass[layer] = in[p++];

			gain = in[p++];

			for (int layer = 0; layer < NumLayers; ++layer)
				for (int nl = 0; nl < NLOrder; ++nl)
					sp[layer][nl] = in[p++];
		}
	};

	template<int i, int Order, typename Sample>
	inline std::tuple<Sample, Sample> ProcessLattice(Sample x, Sample* z, Sample* k, Sample* g)
	{
		if constexpr (i >= Order) return { x,x * g[i] };
		else
		{
			Sample a = x - k[i] * z[i];
			auto [pass, total] =
				ProcessLattice<i + 1, Order, Sample>(a, z, k, g);
			Sample y = a * k[i] + z[i];
			z[i] = pass;
			return { y,total + y * g[i] };
		}
	}

	class NLModelProcess
	{
	private:
		float zs[NumLayers][NLOrder][FiltOrder];

	public:
		void Init()
		{
			for (int i = 0; i < NumLayers; ++i)
				for (int j = 0; j < NLOrder; ++j)
					for (int k = 0; k < FiltOrder; ++k)
						zs[i][j][k] = 0;
		}

		inline float NonlinearChebyshev(float x, float& x0, float& x1, float k)
		{
			float nextx = 2.0 * x * x1 - x0;
			x0 = x1, x1 = nextx;
			return x0;
		}

		inline float NonlinearLegendre(float x, float& x0, float& x1, float k)
		{
			float nextx = ((2.0 * k + 1.0) * x * x1 - k * x0) / (k + 1.0);
			x0 = x1, x1 = nextx;
			return x0;
		}

		inline float NonlinearHermite(float x, float& x0, float& x1, float k)
		{
			float nextx = x * x1 - k * x0;
			x0 = x1, x1 = nextx;
			return x0;
		}

		inline float NonlinearMy(float x, float& x0, float& x1, float k)
		{
			x0 = x1 / (1.0 + fabsf(x1));
			x1 *= x;
			return x0;
		}

		inline float NonlinearSimple(float x, float& x0, float& x1, float k)
		{
			x0 *= x;
			return x0;
		}

		template<typename Sample>
		static inline Sample Clip(Sample x, Sample lo, Sample hi)
		{
			return x < lo ? lo : (x > hi ? hi : x);
		}

		void ProcessBlock(NLModelParams& p, const float* in, float* out, int NumSamples)
		{
			for (int i = 0; i < NumSamples; ++i)
			{
				float x = in[i] * p.gain;

				for (int n = 0; n < NumLayers; ++n)
				{
					float input = x;
					float y = 0;

					for (int j = 0; j < NLOrder; ++j)
					{
						float nx = x * p.sp[n][j];
						float x0 = 1.0, x1 = nx;
						float nlout = NonlinearMy(nx, x0, x1, j + 1);

						auto [pass, total] =
							ProcessLattice<0, FiltOrder, float>(
								nlout,
								zs[n][j],
								p.ks[n][j],
								p.gs[n][j]);

						y += total;
					}

					x = Clip(y + input * p.pass[n], -1.0f, 1.0f);
				}

				out[i] = x;
			}
		}
	};
}

namespace NLModelingGRU
{
	constexpr static int HiddenSize = 4;

	constexpr static int GateParams = HiddenSize + HiddenSize * HiddenSize + HiddenSize;
	constexpr static int NumParams = GateParams * 3 + HiddenSize + 1;

	constexpr static float InputScale = 1000.0f;

	constexpr static float InputWeightRange = 1000.0f;
	constexpr static float RecurrentWeightRange = 1000.0f;
	constexpr static float BiasRange = 1000.0f;
	constexpr static float OutputWeightRange = 1000.0f;
	constexpr static float OutputBiasRange = 1000.0f;


	template<typename Sample>
	static inline Sample Clip(Sample x, Sample lo, Sample hi)
	{
		return x < lo ? lo : (x > hi ? hi : x);
	}

	template<typename Sample>
	static inline Sample Sigmoid(Sample x)
	{
		return Sample(0.5) * (std::tanh(Sample(0.5) * x) + Sample(1));
	}
	template<typename Sample>
	static inline Sample Tanh(Sample x)
	{
		return std::tanh(x);
	}
	/*
	template<typename Sample>
	static inline Sample Tanh(Sample x)
	{
		return Clip(x, Sample(-1), Sample(1));
	}
	template<typename Sample>
	static inline Sample Sigmoid(Sample x)
	{
		return Clip(Sample(0.5) + Sample(0.25) * x, Sample(0), Sample(1));
	}*/
	/*
	template<typename Sample>
	static inline Sample Tanh(Sample x)
	{
		return x / (1.0f + fabsf(x));
	}
	template<typename Sample>
	static inline Sample Sigmoid(Sample x)
	{
		return Sample(0.5) * (std::tanh(Sample(0.5) * x) + Sample(1));
	}*/

	static inline float MapParam(float x, float range)
	{
		return range * std::tanh(x);
	}

	static inline float UnmapParam(float x, float range)
	{
		return std::atanh(x / range);
	}

	struct NLModelParams
	{
		float wz[HiddenSize];
		float uz[HiddenSize][HiddenSize];
		float bz[HiddenSize];

		float wr[HiddenSize];
		float ur[HiddenSize][HiddenSize];
		float br[HiddenSize];

		float wh[HiddenSize];
		float uh[HiddenSize][HiddenSize];
		float bh[HiddenSize];

		float wo[HiddenSize];
		float bo;

		static void InitVecDirect(float* out)
		{
			static_assert((HiddenSize & 1) == 0);

			int p = 0;

			for (int i = 0; i < HiddenSize; ++i)
				out[p++] = UnmapParam(0.0f, InputWeightRange);

			for (int i = 0; i < HiddenSize; ++i)
				for (int j = 0; j < HiddenSize; ++j)
					out[p++] = UnmapParam(0.0f, RecurrentWeightRange);

			for (int i = 0; i < HiddenSize; ++i)
				out[p++] = UnmapParam(-2.0f, BiasRange);

			for (int i = 0; i < HiddenSize; ++i)
				out[p++] = UnmapParam(0.0f, InputWeightRange);

			for (int i = 0; i < HiddenSize; ++i)
				for (int j = 0; j < HiddenSize; ++j)
					out[p++] = UnmapParam(0.0f, RecurrentWeightRange);

			for (int i = 0; i < HiddenSize; ++i)
				out[p++] = UnmapParam(0.0f, BiasRange);

			for (int i = 0; i < HiddenSize; ++i)
			{
				float w = 0.7f + 0.2f * (float)(i / 2);
				out[p++] = UnmapParam(w, InputWeightRange);
			}

			for (int i = 0; i < HiddenSize; ++i)
				for (int j = 0; j < HiddenSize; ++j)
					out[p++] = UnmapParam(0.0f, RecurrentWeightRange);

			for (int i = 0; i < HiddenSize; ++i)
				out[p++] = UnmapParam(0.0f, BiasRange);

			for (int i = 0; i < HiddenSize; ++i)
			{
				float w = (i & 1) ? -0.05f : 0.05f;
				out[p++] = UnmapParam(w, OutputWeightRange);
			}

			out[p++] = UnmapParam(0.0f, OutputBiasRange);
		}

		void ParamsToVec(float* out) const
		{
			int p = 0;

			for (int i = 0; i < HiddenSize; ++i)
				out[p++] = UnmapParam(wz[i], InputWeightRange);

			for (int i = 0; i < HiddenSize; ++i)
				for (int j = 0; j < HiddenSize; ++j)
					out[p++] = UnmapParam(uz[i][j], RecurrentWeightRange);

			for (int i = 0; i < HiddenSize; ++i)
				out[p++] = UnmapParam(bz[i], BiasRange);

			for (int i = 0; i < HiddenSize; ++i)
				out[p++] = UnmapParam(wr[i], InputWeightRange);

			for (int i = 0; i < HiddenSize; ++i)
				for (int j = 0; j < HiddenSize; ++j)
					out[p++] = UnmapParam(ur[i][j], RecurrentWeightRange);

			for (int i = 0; i < HiddenSize; ++i)
				out[p++] = UnmapParam(br[i], BiasRange);

			for (int i = 0; i < HiddenSize; ++i)
				out[p++] = UnmapParam(wh[i], InputWeightRange);

			for (int i = 0; i < HiddenSize; ++i)
				for (int j = 0; j < HiddenSize; ++j)
					out[p++] = UnmapParam(uh[i][j], RecurrentWeightRange);

			for (int i = 0; i < HiddenSize; ++i)
				out[p++] = UnmapParam(bh[i], BiasRange);

			for (int i = 0; i < HiddenSize; ++i)
				out[p++] = UnmapParam(wo[i], OutputWeightRange);

			out[p++] = UnmapParam(bo, OutputBiasRange);
		}

		void VecToParams(const float* in)
		{
			int p = 0;

			for (int i = 0; i < HiddenSize; ++i)
				wz[i] = MapParam(in[p++], InputWeightRange);

			for (int i = 0; i < HiddenSize; ++i)
				for (int j = 0; j < HiddenSize; ++j)
					uz[i][j] = MapParam(in[p++], RecurrentWeightRange);

			for (int i = 0; i < HiddenSize; ++i)
				bz[i] = MapParam(in[p++], BiasRange);

			for (int i = 0; i < HiddenSize; ++i)
				wr[i] = MapParam(in[p++], InputWeightRange);

			for (int i = 0; i < HiddenSize; ++i)
				for (int j = 0; j < HiddenSize; ++j)
					ur[i][j] = MapParam(in[p++], RecurrentWeightRange);

			for (int i = 0; i < HiddenSize; ++i)
				br[i] = MapParam(in[p++], BiasRange);

			for (int i = 0; i < HiddenSize; ++i)
				wh[i] = MapParam(in[p++], InputWeightRange);

			for (int i = 0; i < HiddenSize; ++i)
				for (int j = 0; j < HiddenSize; ++j)
					uh[i][j] = MapParam(in[p++], RecurrentWeightRange);

			for (int i = 0; i < HiddenSize; ++i)
				bh[i] = MapParam(in[p++], BiasRange);

			for (int i = 0; i < HiddenSize; ++i)
				wo[i] = MapParam(in[p++], OutputWeightRange);

			bo = MapParam(in[p++], OutputBiasRange);
		}
	};

	class NLModelProcess
	{
	private:
		float hs[HiddenSize];

		float h0[HiddenSize];
		float z[HiddenSize];
		float r[HiddenSize];
		float ht[HiddenSize];
	public:
		void Init()
		{
			for (int i = 0; i < HiddenSize; ++i)
				hs[i] = 0.0f;
		}

		template<typename Sample>
		static inline Sample Clip(Sample x, Sample lo, Sample hi)
		{
			return x < lo ? lo : (x > hi ? hi : x);
		}
		void ProcessBlock(NLModelParams& p, const float* in, float* out, int NumSamples)
		{
			for (int i = 0; i < NumSamples; ++i)
			{
				float x = in[i] * InputScale;


				for (int j = 0; j < HiddenSize; ++j)
					h0[j] = Clip(hs[j], -10.0f, 10.0f);

				for (int j = 0; j < HiddenSize; ++j)
				{
					float v = p.wz[j] * x + p.bz[j];

					for (int k = 0; k < HiddenSize; ++k)
						v += p.uz[j][k] * h0[k];

					z[j] = Sigmoid(v);
				}

				for (int j = 0; j < HiddenSize; ++j)
				{
					float v = p.wr[j] * x + p.br[j];

					for (int k = 0; k < HiddenSize; ++k)
						v += p.ur[j][k] * h0[k];

					r[j] = Sigmoid(v);
				}

				for (int j = 0; j < HiddenSize; ++j)
				{
					float v = p.wh[j] * x + p.bh[j];

					for (int k = 0; k < HiddenSize; ++k)
						v += p.uh[j][k] * (r[k] * h0[k]);

					ht[j] = Tanh(v);
				}

				for (int j = 0; j < HiddenSize; ++j)
					hs[j] = (1.0f - z[j]) * ht[j] + z[j] * h0[j];

				float y = in[i] + p.bo;

				for (int j = 0; j < HiddenSize; ++j)
					y += p.wo[j] * hs[j];

				out[i] = Clip(y, -1.0f, 1.0f);
			}
		}
	};
}