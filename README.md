# Luganda Speech-to-Text API with Audio Cleaning

Enhanced Luganda Speech-to-Text API featuring advanced audio cleaning, noise reduction, and voice enhancement for improved transcription accuracy.

## Features

### 🎙️ Core Features
- **Real-time Luganda Speech-to-Text Transcription** using Wav2Vec2 model
- **Professional Audio Cleaning Pipeline**:
  - 🔇 **Noise Reduction** - Removes background noise while preserving speech
  - 🔊 **Voice Enhancement** - High-pass filtering and dynamic compression
  - 📊 **Audio Normalization** - Consistent loudness levels
  - ✂️ **Silence Trimming** - Removes silence from beginning/end
- **Multiple API Endpoints** for different use cases
- **Comprehensive Audio Analysis** - Quality metrics and statistics
- **Customizable Cleaning Parameters** - Fine-tune audio processing

## What's New in This Version

✨ **Audio Preprocessing**: Automatic noise reduction and voice enhancement before transcription
✨ **Quality Metrics**: Detailed audio statistics (RMS, peak amplitude, SNR, frequency analysis)
✨ **Flexible Configuration**: Control noise reduction strength and cleaning options
✨ **Multiple Endpoints**: Transcription, audio analysis, translation preparation
✨ **Production Ready**: Error handling, timeout management, configurable parameters

## Prerequisites

- Python 3.7+
- pip package manager
- 2GB+ RAM for model loading
- Audio files in WAV, MP3, or FLAC format

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/kpttiling-wq/git-clone-https-github.com-lucy-kevin-Luganda-Speech-To-Text-API.git-cd-Luganda-Speech-To-Text-API.git
cd git-clone-https-github.com-lucy-kevin-Luganda-Speech-To-Text-API.git-cd-Luganda-Speech-To-Text-API
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. First Run

The first time you run the API, it will download the pre-trained Wav2Vec2 model (~1-2GB). This may take a few minutes.

## Running the API

### Start the Flask Server

```bash
cd myproject
python myApp.py
```

The API will be accessible at `http://localhost:5000`

You should see output like:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

## API Endpoints

### 1. `/transcribe` (POST)
Transcribe audio with optional audio cleaning.

**Parameters:**
- `audio` (file, required): Audio file (WAV, MP3, FLAC)
- `clean_audio` (boolean, optional): Enable audio cleaning (default: true)
- `reduction_strength` (float, optional): Noise reduction strength 0.0-1.0 (default: 0.5)

**Example:**
```bash
curl -X POST -F 'audio=@your_audio.wav' \
  'http://localhost:5000/transcribe?clean_audio=true&reduction_strength=0.5'
```

**Response:**
```json
{
  "transcription": "Olwadde lwange nyo",
  "audio_cleaning_applied": true,
  "audio_stats": {
    "before_cleaning": {
      "duration_seconds": 5.2,
      "rms_db": -18.5,
      "snr_estimate_db": 8.2
    },
    "after_cleaning": {
      "duration_seconds": 4.9,
      "rms_db": -20.0,
      "snr_estimate_db": 18.7
    }
  }
}
```

### 2. `/audio-stats` (POST)
Analyze audio quality without transcription.

**Parameters:**
- `audio` (file, required): Audio file
- `clean_audio` (boolean, optional): Analyze after cleaning (default: false)

**Example:**
```bash
curl -X POST -F 'audio=@your_audio.wav' \
  'http://localhost:5000/audio-stats?clean_audio=true'
```

**Response:**
```json
{
  "status": "success",
  "audio_stats": {
    "duration_seconds": 5.2,
    "rms_db": -20.0,
    "peak_amplitude": 0.85,
    "crest_factor": 2.3,
    "snr_estimate_db": 18.7,
    "dominant_frequency_hz": 250,
    "sample_rate": 16000,
    "total_samples": 83200
  }
}
```

### 3. `/transcribe-and-translate` (POST)
Transcribe audio and prepare for translation.

**Parameters:**
- `audio` (file, required): Audio file
- `clean_audio` (boolean, optional): Enable audio cleaning (default: true)
- `target_language` (string, optional): Target language code (default: "en")

**Example:**
```bash
curl -X POST -F 'audio=@your_audio.wav' \
  'http://localhost:5000/transcribe-and-translate?target_language=en'
```

### 4. `/health` (GET)
Check API health status.

**Example:**
```bash
curl http://localhost:5000/health
```

## Usage Guide

### Using Python Client

```python
from example_usage import LugandaSTTClient

# Initialize client
client = LugandaSTTClient("http://localhost:5000")

# Check health
if client.health_check():
    print("API is running!")

# Transcribe audio
result = client.transcribe(
    "audio.wav",
    clean_audio=True,
    reduction_strength=0.7
)

print(f"Transcription: {result['transcription']}")
print(f"Audio Stats: {result['audio_stats']}")
```

### Using cURL

```bash
# Basic transcription
curl -X POST -F 'audio=@audio.wav' http://localhost:5000/transcribe

# With custom parameters
curl -X POST -F 'audio=@audio.wav' \
  'http://localhost:5000/transcribe?clean_audio=true&reduction_strength=0.8'

# Get audio stats only
curl -X POST -F 'audio=@audio.wav' \
  'http://localhost:5000/audio-stats'
```

### Using Postman

1. Open Postman
2. Create a new POST request to `http://localhost:5000/transcribe`
3. Go to Body tab → form-data
4. Add key `audio` with type `File` and select your audio file
5. Add key `clean_audio` with value `true`
6. Click Send

## Audio Cleaning Features

### Noise Reduction
- **Spectral Subtraction**: Removes stationary background noise
- **Configurable Strength**: 0.0 (no reduction) to 1.0 (maximum)
- **Preserves Speech**: Optimized to keep voice quality

### Voice Enhancement
- **High-Pass Filtering**: Removes low-frequency rumble and hum
- **Dynamic Compression**: Balances quiet and loud sections
- **Automatic Gain**: Prevents clipping while maintaining dynamics

### Audio Normalization
- **Target Loudness**: Normalized to -20dB (LUFS)
- **Peak Protection**: Prevents distortion
- **Consistent Output**: All transcriptions have similar volume

### Silence Trimming
- **Beginning/End Removal**: Removes leading/trailing silence
- **Configurable Threshold**: Can adjust sensitivity
- **Preserves Content**: Only removes true silence

## Noise Reduction Strength Guide

| Strength | Use Case | Effect |
|----------|----------|--------|
| 0.0 | Clean audio only | No noise reduction |
| 0.3 | Minimal noise | Light cleaning, maximum speech preservation |
| 0.5 | Moderate noise | **Recommended** - balanced approach |
| 0.7 | Heavy noise | Aggressive noise removal |
| 1.0 | Very heavy noise | Maximum reduction, may affect speech quality |

## Best Practices

### For Best Transcription Results:

1. **Use audio files 16kHz or higher** - Will be resampled to 16kHz automatically
2. **Enable audio cleaning** - Significantly improves accuracy in noisy environments
3. **Optimize noise reduction strength**:
   - Start with 0.5 (default)
   - Increase for noisier environments
   - Decrease if speech quality degrades
4. **Check audio stats** - Monitor SNR and quality metrics
5. **Use clear speech** - Enunciate clearly for best results
6. **Minimize background noise** - Better source audio = better results

### Audio Optimization Tips:

- Remove echo/reverb before sending
- Avoid multiple speakers in same file
- Keep audio duration reasonable (5-30 seconds per file)
- Use mono audio (single channel)
- Record at consistent volume

## Troubleshooting

### Issue: "API is not responding"
**Solution:**
```bash
# Check if server is running
curl http://localhost:5000/health

# If not, start the server:
cd myproject
python myApp.py
```

### Issue: "Missing 'audio' field in request"
**Solution:** Ensure you're uploading audio with key name `audio`:
```bash
curl -X POST -F 'audio=@file.wav' http://localhost:5000/transcribe
```

### Issue: Slow transcription on first run
**Solution:** First transcription takes longer as model loads. Subsequent requests are faster.

### Issue: "CUDA out of memory" or "RuntimeError: CUDA"
**Solution:** The API uses CPU by default. If you have GPU and want to use it:
```python
# In myApp.py, uncomment:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
```

### Issue: Poor transcription quality
**Solution:**
1. Check audio stats: `curl -X POST -F 'audio=@audio.wav' http://localhost:5000/audio-stats`
2. Increase noise reduction: `reduction_strength=0.7 or 0.8`
3. Ensure audio is 16kHz (mono preferred)
4. Check SNR (Signal-to-Noise Ratio) - should be > 10dB for good results

### Issue: Audio file format not supported
**Solution:** Convert to WAV first:
```bash
# Using FFmpeg
ffmpeg -i audio.mp3 -acodec pcm_s16le -ar 16000 audio.wav

# Using SoX
sox audio.mp3 -r 16000 audio.wav
```

## Configuration

### Adjusting Audio Cleaning Parameters

Edit `myproject/myApp.py` to change default parameters:

```python
# Line 50-52: Default cleaning parameters
audio_array, stats = audio_preprocessor(
    audio_bytes,
    clean_audio=True,  # Set to False to disable by default
    reduction_strength=0.5  # Adjust default strength
)
```

### Adjusting Model Parameters

Edit `myproject/audio_cleaner.py` to fine-tune processing:

```python
# Noise reduction (line 33-40)
reduced_noise = nr.reduce_noise(
    prop_decrease=0.5  # Increase for stronger reduction
)

# High-pass filter cutoff frequency (line 61)
sos = signal.butter(5, 80, 'hp', fs=self.sr)  # Change 80 Hz to desired frequency

# Target loudness for normalization (line 141)
target_loudness=-20.0  # Adjust loudness target in dB
```

## Performance

- **Transcription Speed**: 5-10 seconds audio ≈ 2-5 seconds processing time
- **Memory Usage**: ~2GB for model + ~200MB per transcription
- **CPU Usage**: 1-2 cores at 80-100% during transcription
- **Batch Processing**: Process audio sequentially or implement async queue

## Model Information

- **Model**: `indonesian-nlp/wav2vec2-luganda`
- **Type**: Wav2Vec2 CTC (Connectionist Temporal Classification)
- **Language**: Luganda (lg)
- **Input Sample Rate**: 16kHz
- **Framework**: PyTorch / Hugging Face Transformers

## Architecture

```
Audio Input
    ↓
[Audio Cleaner Module]
  ├─ Noise Reduction
  ├─ Voice Enhancement  
  ├─ Normalization
  └─ Silence Trimming
    ↓
[Resampler: 16kHz]
    ↓
[Wav2Vec2 Processor]
    ↓
[Wav2Vec2 Model]
    ↓
[CTC Decoder]
    ↓
Transcription Output
```

## Links

- [Original Luganda Speech-to-Text API](https://github.com/lucy-kevin/Luganda-Speech-To-Text-API)
- [Luganda Speech-to-Text Mobile App](https://github.com/lucy-kevin/Luganda-Speech-to-Text)
- [Wav2Vec2 Luganda Model](https://huggingface.co/indonesian-nlp/wav2vec2-luganda)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open source and available under the MIT License.

## Support

For issues and questions:
- Check the Troubleshooting section above
- Review existing GitHub issues
- Create a new issue with detailed description and error logs

## Acknowledgments

- Original Luganda STT project by Lucy Kevin
- Wav2Vec2 model by Indonesian-NLP
- Audio processing by SciPy, librosa, and noisereduce
