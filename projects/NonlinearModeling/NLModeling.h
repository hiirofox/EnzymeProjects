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
	constexpr static int NumLayers = 6;
	constexpr static int NLOrder = 8;
	constexpr static int FiltOrder = 8;
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
						float nlout = NonlinearSimple(nx, x0, x1, j + 1);

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
namespace NLModeling3
{
	constexpr static int NumLayers = 8;
	constexpr static int FiltOrder = 8;
	constexpr static int ParamsPerLayer = FiltOrder * 2 + 1 + 5 + 3;
	constexpr static int NumParams = NumLayers * ParamsPerLayer;
	constexpr static float kScale = 0.9995;

	struct NLModelParams
	{
		float k[NumLayers][FiltOrder];
		float gf[NumLayers][FiltOrder];
		float gfx[NumLayers];

		float a1[NumLayers];
		float a2[NumLayers];
		float a3[NumLayers];
		float b1[NumLayers];
		float b2[NumLayers];

		float gdry[NumLayers];
		float gnlin[NumLayers];
		float gnlout[NumLayers];

		template<typename Sample>
		static inline Sample Clip(Sample x, Sample lo, Sample hi)
		{
			return x < lo ? lo : (x > hi ? hi : x);
		}
		static void InitVecDirect(float* out)
		{
			int p = 0;
			for (int layer = 0; layer < NumLayers; ++layer)
			{
				for (int i = 0; i < FiltOrder; ++i)
					out[p++] = 0.0f;

				for (int i = 0; i < FiltOrder; ++i)
					out[p++] = 0.0f;

				out[p++] = 1.0f;
				out[p++] = 0.0f;
				out[p++] = 1.0f;
				out[p++] = 0.0f;
				out[p++] = 1.0f;
				out[p++] = 0.0f;
				out[p++] = 0.0f;
				out[p++] = 1.0f;
				out[p++] = 1.0f;
			}
		}

		void ParamsToVec(float* out) const
		{
			int p = 0;
			for (int layer = 0; layer < NumLayers; ++layer)
			{
				for (int i = 0; i < FiltOrder; ++i)
					out[p++] = k[layer][i];

				for (int i = 0; i < FiltOrder; ++i)
					out[p++] = gf[layer][i];

				out[p++] = gfx[layer];
				out[p++] = a1[layer];
				out[p++] = a2[layer];
				out[p++] = a3[layer];
				out[p++] = b1[layer];
				out[p++] = b2[layer];
				out[p++] = gdry[layer];
				out[p++] = gnlin[layer];
				out[p++] = gnlout[layer];
			}
		}

		void VecToParams(const float* in)
		{
			int p = 0;
			for (int layer = 0; layer < NumLayers; ++layer)
			{
				for (int i = 0; i < FiltOrder; ++i)
					k[layer][i] = Clip(in[p++], -kScale, kScale);

				for (int i = 0; i < FiltOrder; ++i)
					gf[layer][i] = in[p++];

				gfx[layer] = in[p++];
				a1[layer] = in[p++];
				a2[layer] = in[p++];
				a3[layer] = in[p++];
				b1[layer] = in[p++];
				b2[layer] = in[p++];
				gdry[layer] = in[p++];
				gnlin[layer] = in[p++];
				gnlout[layer] = in[p++];
			}
		}
	};

	template<int layer, typename Sample>
	inline std::tuple<Sample, Sample> ProcessLattice(
		Sample x, Sample* z, const Sample* k,
		const Sample* gf, Sample gfx)
	{
		if constexpr (layer >= FiltOrder)
			return { x, x * gfx };
		else
		{
			auto z0 = z[layer];
			auto a = z0 * k[layer] + x;
			auto [nextz, out] = ProcessLattice<layer + 1, Sample>(a, z, k, gf, gfx);
			z[layer] = nextz;
			auto y = a * -k[layer] + z0;
			return { y, out + y * gf[layer] };
		}
	}

	class NLModelProcess
	{
	private:
		float z[NumLayers][FiltOrder];
		float nlz[NumLayers];
	public:
		NLModelProcess()
		{
			Init();
		}

		void Init()
		{
			for (int layer = 0; layer < NumLayers; ++layer)
				for (int i = 0; i < FiltOrder; ++i)
					z[layer][i] = 0.0f;

			for (int layer = 0; layer < NumLayers; ++layer)
				nlz[layer] = 0;
		}

		static inline float Nonlinear(
			float x, const NLModelParams& p, int layer)
		{
			auto x2 = x * x;
			auto absx = std::abs(x);
			auto num = x * (x * (x + p.a1[layer]) + p.a2[layer] + p.a3[layer] * absx);
			auto den = p.b1[layer] * x2 + p.b2[layer] * x2 * absx + 1.0f;
			return num / den;
		}

		inline float ProcessCell(float x, const NLModelParams& p, int layer)
		{
			auto [nextz, latticeOut] = ProcessLattice<0, float>(
				x, z[layer], p.k[layer], p.gf[layer], p.gfx[layer]);
			auto nlo = Nonlinear(latticeOut * p.gnlin[layer], p, layer);
			return x * p.gdry[layer] + nlo * p.gnlout[layer];
		}

		void ProcessBlock(NLModelParams& p, const float* in, float* out, int NumSamples)
		{
			for (int i = 0; i < NumSamples; ++i)
			{
				float x = in[i];
				for (int layer = 0; layer < NumLayers; ++layer)
					x = ProcessCell(x, p, layer);
				out[i] = x;
			}
		}
	};
}

namespace NLModelingGRU
{
	constexpr static int NumInputs = 1;
	constexpr static int NumHiddens = 8;

	constexpr static int NumParams =
		3 * NumHiddens * NumInputs +
		3 * NumHiddens * NumHiddens +
		4 * NumHiddens + 1;

	struct NLModelParams
	{
		std::array<std::array<float, NumInputs>, NumHiddens> wr;
		std::array<std::array<float, NumInputs>, NumHiddens> wz;
		std::array<std::array<float, NumInputs>, NumHiddens> wh;
		std::array<std::array<float, NumHiddens>, NumHiddens> ur;
		std::array<std::array<float, NumHiddens>, NumHiddens> uz;
		std::array<std::array<float, NumHiddens>, NumHiddens> uh;
		std::array<float, NumHiddens> br;
		std::array<float, NumHiddens> bz;
		std::array<float, NumHiddens> bh;
		std::array<float, NumHiddens> wo;
		float b;

		static void InitVecRandom(float* out)
		{
			static std::mt19937 rng(12345);
			static std::normal_distribution<float> nd(0.0f, 1.0f);
			int p = 0;
			auto R = [&](float m, float s)
				{
					return std::clamp(m + s * nd(rng), m - 2.0f * s, m + 2.0f * s);
				};
			auto F = [&](int n, float m, float s)
				{
					for (int i = 0; i < n; ++i)out[p++] = R(m, s);
				};
			F(NumHiddens * NumInputs, 0.363f, 0.447f); // wr
			F(NumHiddens * NumInputs, 0.115f, 0.402f); // wz
			F(NumHiddens * NumInputs, 0.071f, 0.585f); // wh
			F(NumHiddens * NumHiddens, -0.016f, 0.439f); // ur
			F(NumHiddens * NumHiddens, -0.022f, 0.397f); // uz
			F(NumHiddens * NumHiddens, 0.037f, 0.486f); // uh
			F(NumHiddens, 0.134f, 0.059f); // br
			F(NumHiddens, 1.239f, 0.116f); // bz
			F(NumHiddens, 0.051f, 0.117f); // bh
			F(NumHiddens, 0.125f, 0.677f); // wo
			out[p++] = 0.0f; // b
		}
		static void InitVecRandom2(float* out)
		{
			std::mt19937 rng(31415926);
			auto Rand = [&](float a)
				{
					std::uniform_real_distribution<float> d(-a, a);
					return d(rng);
				};
			NLModelParams p;
			const float wi = std::sqrt(6.0f / (NumInputs + NumHiddens));
			const float wh = std::sqrt(6.0f / (NumHiddens + NumHiddens));
			const float wo = std::sqrt(6.0f / (NumHiddens + 1));
			for (int i = 0; i < NumHiddens; ++i)
			{
				for (int j = 0; j < NumInputs; ++j)
				{
					p.wr[i][j] = Rand(wi);
					p.wz[i][j] = Rand(wi);
					p.wh[i][j] = Rand(wi);
				}
				for (int j = 0; j < NumHiddens; ++j)
				{
					p.ur[i][j] = Rand(wh);
					p.uz[i][j] = Rand(wh);
					p.uh[i][j] = Rand(wh);
				}
				p.br[i] = 0.0f;
				p.bz[i] = 1.0f;
				p.bh[i] = 0.0f;
				p.wo[i] = Rand(wo);
			}
			p.b = 0.0f;
			p.ParamsToVec(out);
		}
		static void InitVecDirect(float* out)
		{
			int p = 0;
			constexpr float s = 0.1f;
			for (int i = 0; i < NumHiddens; ++i)
				for (int j = 0; j < NumInputs; ++j)	out[p++] = 0.0f;
			for (int i = 0; i < NumHiddens; ++i)
				for (int j = 0; j < NumInputs; ++j)	out[p++] = 0.0f;
			for (int i = 0; i < NumHiddens; ++i)
				for (int j = 0; j < NumInputs; ++j)	out[p++] = (i == 0 && j == 0) ? s : 0.0f;
			for (int i = 0; i < NumHiddens * NumHiddens; ++i)out[p++] = 0.0f;
			for (int i = 0; i < NumHiddens * NumHiddens; ++i)out[p++] = 0.0f;
			for (int i = 0; i < NumHiddens * NumHiddens; ++i)out[p++] = 0.0f;
			for (int i = 0; i < NumHiddens; ++i)out[p++] = 0.0f;
			for (int i = 0; i < NumHiddens; ++i)out[p++] = -8.0f;
			for (int i = 0; i < NumHiddens; ++i)out[p++] = 0.0f;
			for (int i = 0; i < NumHiddens; ++i)out[p++] = (i == 0) ? 1.0f / s : 0.0f;
			out[p++] = 0.0f;
		}
		void ParamsToVec(float* out) const
		{
			int p = 0;
			for (int i = 0; i < NumHiddens; ++i)for (int j = 0; j < NumInputs; ++j)out[p++] = wr[i][j];
			for (int i = 0; i < NumHiddens; ++i)for (int j = 0; j < NumInputs; ++j)out[p++] = wz[i][j];
			for (int i = 0; i < NumHiddens; ++i)for (int j = 0; j < NumInputs; ++j)out[p++] = wh[i][j];
			for (int i = 0; i < NumHiddens; ++i)for (int j = 0; j < NumHiddens; ++j)out[p++] = ur[i][j];
			for (int i = 0; i < NumHiddens; ++i)for (int j = 0; j < NumHiddens; ++j)out[p++] = uz[i][j];
			for (int i = 0; i < NumHiddens; ++i)for (int j = 0; j < NumHiddens; ++j)out[p++] = uh[i][j];
			for (int i = 0; i < NumHiddens; ++i)out[p++] = br[i];
			for (int i = 0; i < NumHiddens; ++i)out[p++] = bz[i];
			for (int i = 0; i < NumHiddens; ++i)out[p++] = bh[i];
			for (int i = 0; i < NumHiddens; ++i)out[p++] = wo[i];
			out[p++] = b;
		}
		void VecToParams(const float* in)
		{
			int p = 0;
			for (int i = 0; i < NumHiddens; ++i)for (int j = 0; j < NumInputs; ++j)wr[i][j] = in[p++];
			for (int i = 0; i < NumHiddens; ++i)for (int j = 0; j < NumInputs; ++j)wz[i][j] = in[p++];
			for (int i = 0; i < NumHiddens; ++i)for (int j = 0; j < NumInputs; ++j)wh[i][j] = in[p++];
			for (int i = 0; i < NumHiddens; ++i)for (int j = 0; j < NumHiddens; ++j)ur[i][j] = in[p++];
			for (int i = 0; i < NumHiddens; ++i)for (int j = 0; j < NumHiddens; ++j)uz[i][j] = in[p++];
			for (int i = 0; i < NumHiddens; ++i)for (int j = 0; j < NumHiddens; ++j)uh[i][j] = in[p++];
			for (int i = 0; i < NumHiddens; ++i)br[i] = in[p++];
			for (int i = 0; i < NumHiddens; ++i)bz[i] = in[p++];
			for (int i = 0; i < NumHiddens; ++i)bh[i] = in[p++];
			for (int i = 0; i < NumHiddens; ++i)wo[i] = in[p++];
			b = in[p++];
		}
	};

	class NLModelProcess
	{
	private:
		std::array<float, NumInputs> inputp{ 0 };
		std::array<float, NumHiddens> h{ 0 };
		std::array<float, NumHiddens> tmp1{ 0 };
		std::array<float, NumHiddens> tmp2{ 0 };
		std::array<float, NumHiddens> s1o{ 0 };
		std::array<float, NumHiddens> s2o{ 0 };
		std::array<float, NumHiddens> tho{ 0 };
		inline static float Tanh(float x)
		{
			//return tanhf(x);
			return x / (1.0f + x * x) + 0.125f * x;
		}
		inline static float Sigmoid(float x)
		{
			return 0.5f * (Tanh(0.5f * x) + 1.0f);
		}
	public:
		void Init()
		{
			for (auto& v : h)v = 0;
		}
		void SetRuntimeParams(float* inputp)
		{
			for (int i = 0; i < NumInputs - 1; ++i)
				this->inputp[i] = inputp[i];
		}
		template<int W, int H>
		inline void MatMul(
			std::array<float, W>& x,
			std::array<std::array<float, W>, H>& mat,
			std::array<float, H>& y)
		{
			for (int m = 0; m < H; ++m)
			{
				float v = 0.0;
				for (int n = 0; n < W; ++n)
					v += x[n] * mat[m][n];
				y[m] = v;
			}
		}
		template<int W>
		inline void VecAdd(
			std::array<float, W>& x1,
			std::array<float, W>& x2,
			std::array<float, W>& y)
		{
			for (int n = 0; n < W; ++n)y[n] = x1[n] + x2[n];
		}
		template<int W>
		inline void VecTanh(std::array<float, W>& x, std::array<float, W>& y)
		{
			for (int n = 0; n < W; ++n)y[n] = Tanh(x[n]);
		}
		template<int W>
		inline void VecSigmoid(std::array<float, W>& x, std::array<float, W>& y)
		{
			for (int n = 0; n < W; ++n)y[n] = Sigmoid(x[n]);
		}
		void ProcessBlock(NLModelParams& p, const float* in, float* out, int NumSamples)
		{
			std::array<float, NumInputs> x;
			for (int i = 0; i < NumSamples; ++i)
			{
				x[0] = in[i];
				for (int j = 1; j < NumInputs; ++j) x[j] = inputp[j - 1];
				//r
				MatMul<NumInputs, NumHiddens>(x, p.wr, tmp1);
				MatMul<NumHiddens, NumHiddens>(h, p.ur, tmp2);
				VecAdd<NumHiddens>(tmp1, tmp2, tmp1);
				VecAdd<NumHiddens>(tmp1, p.br, tmp1);//bias
				VecSigmoid<NumHiddens>(tmp1, s1o);
				//z
				MatMul<NumInputs, NumHiddens>(x, p.wz, tmp1);
				MatMul<NumHiddens, NumHiddens>(h, p.uz, tmp2);
				VecAdd<NumHiddens>(tmp1, tmp2, tmp1);
				VecAdd<NumHiddens>(tmp1, p.bz, tmp1);//bias
				VecSigmoid<NumHiddens>(tmp1, s2o);
				//h
				MatMul<NumInputs, NumHiddens>(x, p.wh, tmp1);
				MatMul<NumHiddens, NumHiddens>(h, p.uh, tmp2);
				for (int n = 0; n < NumHiddens; ++n) tmp2[n] = s1o[n] * tmp2[n];
				VecAdd<NumHiddens>(tmp1, tmp2, tmp1);
				VecAdd<NumHiddens>(tmp1, p.bh, tmp1);//bias
				VecTanh<NumHiddens>(tmp1, tho);
				//next
				for (int n = 0; n < NumHiddens; ++n)tmp1[n] = s2o[n] * h[n];
				for (int n = 0; n < NumHiddens; ++n)tmp2[n] = (1.0f - s2o[n]) * tho[n];
				for (int n = 0; n < NumHiddens; ++n)h[n] = tmp1[n] + tmp2[n];
				//out
				float v = 0;
				for (int n = 0; n < NumHiddens; ++n) v += h[n] * p.wo[n];
				out[i] = v + p.b;
			}
		}
	};
}

namespace NLModeling4//NLModeling4 powered by ai!
{
	constexpr static int NumLayers = 12;

	constexpr static int ParamsPerLayer = 15;
	constexpr static int NumParams = NumLayers * ParamsPerLayer;
	constexpr static float PoleScale = 0.9999f;

	static inline float MapPole(float x)
	{
		return PoleScale * std::tanh(x);
	}

	static inline float UnmapPole(float x)
	{
		return std::atanh(x / PoleScale);
	}

	struct NLModelParams
	{
		float p1[NumLayers], p2[NumLayers];
		float g1[NumLayers], g2[NumLayers];

		float pe[NumLayers];
		float gbias[NumLayers];
		float gin[NumLayers];

		float a1[NumLayers], a2[NumLayers];
		float b1[NumLayers], b2[NumLayers];

		float pp[NumLayers];
		float gp[NumLayers];

		float gdry[NumLayers];
		float gout[NumLayers];

		static void InitVecDirect(float* out)
		{
			int p = 0;

			for (int i = 0; i < NumLayers; ++i)
			{
				out[p++] = UnmapPole(0.5f);
				out[p++] = UnmapPole(0.95f);

				out[p++] = 0.0f;
				out[p++] = 0.0f;

				out[p++] = UnmapPole(0.999f);
				out[p++] = 0.0f;
				out[p++] = 1.0f;

				out[p++] = 0.0f;
				out[p++] = 1.0f;
				out[p++] = 0.0f;
				out[p++] = 1.0f;

				out[p++] = UnmapPole(0.5f);
				out[p++] = 0.0f;

				out[p++] = 0.0f;
				out[p++] = 1.0f;
			}
		}

		void ParamsToVec(float* out) const
		{
			int p = 0;

			for (int i = 0; i < NumLayers; ++i)
			{
				out[p++] = UnmapPole(p1[i]);
				out[p++] = UnmapPole(p2[i]);

				out[p++] = g1[i];
				out[p++] = g2[i];

				out[p++] = UnmapPole(pe[i]);
				out[p++] = gbias[i];
				out[p++] = gin[i];

				out[p++] = a1[i];
				out[p++] = a2[i];
				out[p++] = b1[i];
				out[p++] = b2[i];

				out[p++] = UnmapPole(pp[i]);
				out[p++] = gp[i];

				out[p++] = gdry[i];
				out[p++] = gout[i];
			}
		}

		void VecToParams(const float* in)
		{
			int p = 0;

			for (int i = 0; i < NumLayers; ++i)
			{
				p1[i] = MapPole(in[p++]);
				p2[i] = MapPole(in[p++]);

				g1[i] = in[p++];
				g2[i] = in[p++];

				pe[i] = MapPole(in[p++]);
				gbias[i] = in[p++];
				gin[i] = in[p++];

				a1[i] = in[p++];
				a2[i] = in[p++];
				b1[i] = in[p++];
				b2[i] = in[p++];

				pp[i] = MapPole(in[p++]);
				gp[i] = in[p++];

				gdry[i] = in[p++];
				gout[i] = in[p++];
			}
		}
	};

	class NLModelProcess
	{
		float z1[NumLayers];
		float z2[NumLayers];
		float ze[NumLayers];
		float zp[NumLayers];

	public:
		NLModelProcess()
		{
			Init();
		}

		void Init()
		{
			for (int i = 0; i < NumLayers; ++i)
			{
				z1[i] = 0.0f;
				z2[i] = 0.0f;
				ze[i] = 0.0f;
				zp[i] = 0.0f;
			}
		}

		static inline float Nonlinear(float x, const NLModelParams& p, int i)
		{
			float x2 = x * x;
			float num = x * (1.0f + p.a1[i] * x + p.a2[i] * x2);
			float den = 1.0f + p.b1[i] * std::abs(x) + p.b2[i] * x2;
			return num / den;
		}

		inline float ProcessCell(float x, const NLModelParams& p, int i)
		{
			z1[i] = p.p1[i] * z1[i] + (1.0f - p.p1[i]) * x;
			z2[i] = p.p2[i] * z2[i] + (1.0f - p.p2[i]) * x;
			float u = x + p.g1[i] * (z1[i] - x) + p.g2[i] * (z2[i] - x);
			ze[i] = p.pe[i] * ze[i] + (1.0f - p.pe[i]) * std::abs(u);
			float v = u * p.gin[i] + ze[i] * p.gbias[i];
			float n = Nonlinear(v, p, i);
			zp[i] = p.pp[i] * zp[i] + (1.0f - p.pp[i]) * n;
			float y = n + p.gp[i] * (zp[i] - n);
			return x * p.gdry[i] + y * p.gout[i];
		}

		void ProcessBlock(NLModelParams& p, const float* in, float* out, int NumSamples)
		{
			for (int i = 0; i < NumSamples; ++i)
			{
				float x = in[i];
				for (int layer = 0; layer < NumLayers; ++layer)
					x = ProcessCell(x, p, layer);
				out[i] = x;
			}
		}
	};
}
