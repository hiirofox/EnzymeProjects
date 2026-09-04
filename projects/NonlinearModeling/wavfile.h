#pragma once

#include <fstream>
#include <string>
#include <cstdint>
#include <algorithm>

class WavWriter
{
public:
	void CreateWAV(std::string path, float sampleRate = 48000.0f)
	{
		Close();

		m_sampleRate = static_cast<uint32_t>(sampleRate);
		m_dataBytes = 0;

		m_file.open(path, std::ios::binary);
		if (!m_file.is_open())
			return;

		WriteHeader();
	}

	void WriteBlock(const float* bufl, const float* bufr, int numSamples)
	{
		if (!m_file.is_open() || bufl == nullptr || bufr == nullptr || numSamples <= 0)
			return;

		for (int i = 0; i < numSamples; ++i)
		{
			int16_t l = FloatToInt16(bufl[i]);
			int16_t r = FloatToInt16(bufr[i]);

			WriteValue(l);
			WriteValue(r);

			m_dataBytes += sizeof(int16_t) * 2;
		}
	}

	void Close()
	{
		if (!m_file.is_open())
			return;

		// 回填 RIFF chunk size
		m_file.seekp(4, std::ios::beg);
		uint32_t riffSize = 36 + m_dataBytes;
		WriteValue(riffSize);

		// 回填 data chunk size
		m_file.seekp(40, std::ios::beg);
		WriteValue(m_dataBytes);

		m_file.close();
	}

	~WavWriter()
	{
		Close();
	}

private:
	std::ofstream m_file;
	uint32_t m_sampleRate = 48000;
	uint32_t m_dataBytes = 0;

	static int16_t FloatToInt16(float v)
	{
		v = std::clamp(v, -1.0f, 1.0f);
		return static_cast<int16_t>(v * 32767.0f);
	}

	template <typename T>
	void WriteValue(T value)
	{
		m_file.write(reinterpret_cast<const char*>(&value), sizeof(T));
	}

	void WriteText(const char* text, int size)
	{
		m_file.write(text, size);
	}

	void WriteHeader()
	{
		const uint16_t numChannels = 2;
		const uint16_t bitsPerSample = 16;
		const uint16_t audioFormat = 1; // PCM
		const uint32_t byteRate = m_sampleRate * numChannels * bitsPerSample / 8;
		const uint16_t blockAlign = numChannels * bitsPerSample / 8;

		WriteText("RIFF", 4);
		WriteValue<uint32_t>(0); // 稍后回填
		WriteText("WAVE", 4);

		WriteText("fmt ", 4);
		WriteValue<uint32_t>(16); // fmt chunk size
		WriteValue<uint16_t>(audioFormat);
		WriteValue<uint16_t>(numChannels);
		WriteValue<uint32_t>(m_sampleRate);
		WriteValue<uint32_t>(byteRate);
		WriteValue<uint16_t>(blockAlign);
		WriteValue<uint16_t>(bitsPerSample);

		WriteText("data", 4);
		WriteValue<uint32_t>(0); // 稍后回填
	}
};
class WavReader
{
public:
	bool OpenWAV(const std::string& path)
	{
		Close();

		m_file.clear();
		m_file.open(path, std::ios::binary);

		if (!m_file.is_open())
			return false;

		return ReadHeader();
	}

	// 读取双声道。
	// 如果源文件是 mono，则同一个 mono 信号复制到左右声道。
	int ReadBlock(float* bufl, float* bufr, int numSamples)
	{
		if (!m_file.is_open() ||
			bufl == nullptr ||
			bufr == nullptr ||
			numSamples <= 0)
		{
			return 0;
		}

		int samplesRead = 0;

		for (int i = 0; i < numSamples; ++i)
		{
			float l = 0.0f;
			float r = 0.0f;

			if (!ReadFrame(l, r))
				break;

			bufl[i] = l;
			bufr[i] = r;

			++samplesRead;
		}

		// 不足部分补零
		for (int i = samplesRead; i < numSamples; ++i)
		{
			bufl[i] = 0.0f;
			bufr[i] = 0.0f;
		}

		return samplesRead;
	}

	// mono 文件：直接读取 mono
	// stereo 文件：默认取右声道
	int ReadBlockMono(float* buf, int numSamples)
	{
		if (!m_file.is_open() ||
			buf == nullptr ||
			numSamples <= 0)
		{
			return 0;
		}

		int samplesRead = 0;

		for (int i = 0; i < numSamples; ++i)
		{
			float l = 0.0f;
			float r = 0.0f;

			if (!ReadFrame(l, r))
				break;

			if (m_numChannels == 1)
				buf[i] = l;
			else
				buf[i] = r; // stereo 默认取右声道

			++samplesRead;
		}

		// 不足部分补零
		for (int i = samplesRead; i < numSamples; ++i)
		{
			buf[i] = 0.0f;
		}

		return samplesRead;
	}

	void Close()
	{
		if (m_file.is_open())
			m_file.close();

		m_file.clear();

		m_sampleRate = 0;
		m_numSamples = 0;
		m_dataBytesRemaining = 0;
		m_dataStartPos = 0;

		m_audioFormat = 0;
		m_numChannels = 0;
		m_bitsPerSample = 0;
		m_blockAlign = 0;
	}

	bool IsOpen() const
	{
		return m_file.is_open();
	}

	uint32_t GetSampleRate() const
	{
		return m_sampleRate;
	}

	uint64_t GetNumSamples() const
	{
		return m_numSamples;
	}

	uint64_t GetSamplesRemaining() const
	{
		if (m_blockAlign == 0)
			return 0;

		return m_dataBytesRemaining / m_blockAlign;
	}

	~WavReader()
	{
		Close();
	}

private:
	std::ifstream m_file;

	uint16_t m_audioFormat = 0;
	uint16_t m_numChannels = 0;
	uint16_t m_bitsPerSample = 0;
	uint16_t m_blockAlign = 0;

	uint32_t m_sampleRate = 0;

	uint64_t m_numSamples = 0;
	uint64_t m_dataBytesRemaining = 0;

	std::streampos m_dataStartPos = 0;

private:
	static float Int16ToFloat(int16_t v)
	{
		return static_cast<float>(v) / 32768.0f;
	}

	template <typename T>
	void ReadValue(T& value)
	{
		m_file.read(
			reinterpret_cast<char*>(&value),
			sizeof(T));
	}

	void ReadText(char* text, int size)
	{
		m_file.read(text, size);
	}

	static bool MatchText(const char* a, const char* b)
	{
		return a[0] == b[0] &&
			a[1] == b[1] &&
			a[2] == b[2] &&
			a[3] == b[3];
	}

	// 读取一帧音频。
	// mono:
	//   l == r == mono
	//
	// stereo:
	//   l / r 分别返回左右声道
	bool ReadFrame(float& l, float& r)
	{
		if (!m_file.is_open())
			return false;

		if (m_blockAlign == 0)
			return false;

		if (m_dataBytesRemaining < m_blockAlign)
			return false;

		// =========================================
		// 16-bit integer PCM
		// =========================================
		if (m_audioFormat == 1 &&
			m_bitsPerSample == 16)
		{
			if (m_numChannels == 1)
			{
				int16_t mono = 0;

				ReadValue(mono);

				if (!m_file)
					return false;

				float v = Int16ToFloat(mono);

				l = v;
				r = v;
			}
			else if (m_numChannels == 2)
			{
				int16_t left = 0;
				int16_t right = 0;

				ReadValue(left);
				ReadValue(right);

				if (!m_file)
					return false;

				l = Int16ToFloat(left);
				r = Int16ToFloat(right);
			}
			else
			{
				return false;
			}
		}

		// =========================================
		// 32-bit IEEE Float
		// =========================================
		else if (m_audioFormat == 3 &&
			m_bitsPerSample == 32)
		{
			if (m_numChannels == 1)
			{
				float mono = 0.0f;

				ReadValue(mono);

				if (!m_file)
					return false;

				l = mono;
				r = mono;
			}
			else if (m_numChannels == 2)
			{
				float left = 0.0f;
				float right = 0.0f;

				ReadValue(left);
				ReadValue(right);

				if (!m_file)
					return false;

				l = left;
				r = right;
			}
			else
			{
				return false;
			}
		}
		else
		{
			return false;
		}

		m_dataBytesRemaining -= m_blockAlign;

		return true;
	}

	bool ReadHeader()
	{
		char riff[4] = {};
		char wave[4] = {};

		ReadText(riff, 4);

		uint32_t riffSize = 0;
		ReadValue(riffSize);

		ReadText(wave, 4);

		if (!m_file ||
			!MatchText(riff, "RIFF") ||
			!MatchText(wave, "WAVE"))
		{
			Close();
			return false;
		}

		bool foundFmt = false;
		bool foundData = false;

		while (m_file && (!foundFmt || !foundData))
		{
			char chunkId[4] = {};
			uint32_t chunkSize = 0;

			ReadText(chunkId, 4);
			ReadValue(chunkSize);

			if (!m_file)
				break;

			std::streampos chunkDataStart = m_file.tellg();

			if (MatchText(chunkId, "fmt "))
			{
				// 标准 fmt 至少需要 16 字节
				if (chunkSize < 16)
				{
					Close();
					return false;
				}

				uint32_t byteRate = 0;

				ReadValue(m_audioFormat);
				ReadValue(m_numChannels);
				ReadValue(m_sampleRate);
				ReadValue(byteRate);
				ReadValue(m_blockAlign);
				ReadValue(m_bitsPerSample);

				if (!m_file)
				{
					Close();
					return false;
				}

				foundFmt = true;
			}
			else if (MatchText(chunkId, "data"))
			{
				m_dataStartPos = chunkDataStart;
				m_dataBytesRemaining = chunkSize;

				foundData = true;
			}

			// 跳到下一个 chunk
			std::streamoff skipSize =
				static_cast<std::streamoff>(chunkSize);

			// RIFF chunk 按 2 字节对齐
			if (skipSize & 1)
				++skipSize;

			m_file.seekg(
				chunkDataStart + skipSize,
				std::ios::beg);
		}

		if (!foundFmt || !foundData)
		{
			Close();
			return false;
		}

		// 只支持 mono / stereo
		if (m_numChannels != 1 &&
			m_numChannels != 2)
		{
			Close();
			return false;
		}

		// 支持：
		//
		// format 1 = integer PCM
		// format 3 = IEEE Float
		bool supported = false;

		if (m_audioFormat == 1 &&
			m_bitsPerSample == 16)
		{
			supported = true;
		}
		else if (m_audioFormat == 3 &&
			m_bitsPerSample == 32)
		{
			supported = true;
		}

		if (!supported)
		{
			Close();
			return false;
		}

		// 检查 blockAlign
		uint16_t expectedBlockAlign =
			static_cast<uint16_t>(
				m_numChannels *
				(m_bitsPerSample / 8));

		if (m_blockAlign != expectedBlockAlign)
		{
			Close();
			return false;
		}

		if (m_blockAlign == 0)
		{
			Close();
			return false;
		}

		// WAV 中这里实际上是 frame 数量。
		// mono 时一个 frame = 一个 sample
		// stereo 时一个 frame = L + R
		m_numSamples =
			m_dataBytesRemaining / m_blockAlign;

		// 回到 data chunk 开始位置
		m_file.clear();
		m_file.seekg(
			m_dataStartPos,
			std::ios::beg);

		if (!m_file)
		{
			Close();
			return false;
		}

		return true;
	}
};