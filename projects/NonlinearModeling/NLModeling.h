#pragma once

#include <math.h>
#include <vector>

namespace NLModeling
{
	constexpr static int NumLayers = 5;
	constexpr static int NLOrder = 4;
	constexpr static int FiltOrder = 4;
	constexpr static int NumParams = NumLayers * NLOrder * (FiltOrder * 2 + 1);
	struct NLModelParams
	{
		float ks[NumLayers][NLOrder][FiltOrder];
		float gs[NumLayers][NLOrder][FiltOrder + 1];
		void ParamsToVec(float* out) const
		{
			int p = 0;
			for (int layer = 0; layer < NumLayers; ++layer)
			{
				for (int nl = 0; nl < NLOrder; ++nl)
				{
					for (int i = 0; i < FiltOrder; ++i)
						out[p++] = std::atanh(ks[layer][nl][i]);
					for (int i = 0; i < FiltOrder + 1; ++i)
						out[p++] = gs[layer][nl][i];
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
						ks[layer][nl][i] = std::tanh(in[p++]);
					for (int i = 0; i < FiltOrder + 1; ++i)
						gs[layer][nl][i] = in[p++];
				}
			}
		}
	};

	template<int i, int Order, typename Sample>
	std::tuple<Sample, Sample> ProcessLattice(Sample x, Sample* z, Sample* k, Sample* g)
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
		void ProcessBlock(NLModelParams& p, const float* in, float* out, int NumSamples)
		{
			for (int i = 0; i < NumSamples; ++i)
			{
				float y = in[i];
				for (int n = 0; n < NumLayers; ++n)
				{
					float x0 = y;
					float y0 = 0;
					for (int j = 0; j < NLOrder; ++j)
					{
						auto [pass, total] = ProcessLattice<0, FiltOrder, float>(x0, zs[n][j], p.ks[n][j], p.gs[n][j]);
						y0 += total;
						x0 *= y;//nonlinear
					}
					y = y0;
				}
				out[i] = y;
			}
		}
	};
}