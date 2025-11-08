# Arabic Dialect Dataset Generator
**Natural, Expressive Speech with Cartesia Sonic 3 TTS**

Generate 2,500 high-quality Arabic speech utterances across 5 authentic regional dialects using Cartesia's state-of-the-art Sonic 3 multilingual TTS model.

---

## 🎯 Features

- **Natural & Expressive**: Cartesia Sonic 3 model for authentic-sounding Arabic speech
- **5 Dialects**: Egyptian, Levantine, Gulf, Moroccan, Tunisian
- **125 Simulated Speakers**: Diverse voice combinations
- **2,500 Utterances**: Complete banking/IVR vocabulary dataset
- **Production-Ready**: 16kHz mono WAV, normalized, validated

---

## 🚀 Quick Start

### 1. Setup

```bash
# Clone repository
git clone <repository-url>
cd Dialect_Dataset

# Run automated setup
chmod +x setup.sh
./setup.sh

# Configure API key
cp .env.template .env
nano .env  # Add your Cartesia API key
```

### 2. Voice Discovery

```bash
# Discover and test Arabic voices
python discover_voices.py

# This will:
# - Query Cartesia API for Arabic-capable voices
# - Generate test samples
# - Create voice-to-dialect mappings
```

### 3. Generate Dataset

```bash
# Full generation (2,500 utterances)
python generate_dataset.py

# Or use quick start script
chmod +x quickstart.sh
./quickstart.sh
```

---

## 📁 Project Structure

```
Dialect_Dataset/
├── config.py                   # Configuration & settings
├── discover_voices.py          # Voice discovery tool
├── generate_dataset.py         # Main dataset generator
├── requirements.txt            # Python dependencies
├── setup.sh                    # Automated setup script
├── quickstart.sh              # Quick start pipeline
├── SETUP.md                   # Detailed setup guide
│
├── examples/                  # Usage examples
│   ├── query_dataset.py      # Query & analyze dataset
│   └── export_subset.py      # Export subset of data
│
└── dataset/                   # Generated output
    ├── EGY/                  # Egyptian Arabic
    ├── LAV/                  # Levantine Arabic
    ├── GLF/                  # Gulf Arabic
    ├── MOR/                  # Moroccan Arabic
    ├── TUN/                  # Tunisian Arabic
    ├── manifest.csv          # Complete metadata
    ├── voice_mapping.json    # Voice mappings
    ├── README.md             # Dataset documentation
    └── validation_report.txt # QA report
```

---

## 🎙️ Dataset Specifications

### Dialects & Coverage

| Dialect | Name | Speakers | Utterances |
|---------|------|----------|------------|
| **EGY** | Egyptian Arabic | 30 | 600 |
| **LAV** | Levantine Arabic | 25 | 500 |
| **GLF** | Gulf Arabic | 20 | 400 |
| **MOR** | Moroccan Arabic (Darija) | 25 | 500 |
| **TUN** | Tunisian Arabic | 25 | 500 |
| **Total** | | **125** | **2,500** |

### Audio Format

| Parameter | Value |
|-----------|-------|
| Format | WAV (PCM) |
| Sample Rate | 16,000 Hz |
| Channels | Mono (1) |
| Bit Depth | 16-bit |
| Encoding | PCM signed 16-bit LE |
| Peak Level | ≤ –3 dBFS |
| Silence Padding | 150ms (head + tail) |

### Vocabulary

**Banking Domain Words (10):**
```
التنشيط (activation), التحويل (transfer), الرصيد (balance),
التسديد (settlement), نعم (yes), لا (no), التمويل (financing),
البيانات (data), الحساب (account), إنتهاء (finished)
```

**Numbers (10):**
```
صفر (0), واحد (1), اثنان (2), ثلاثة (3), أربعة (4),
خمسة (5), ستة (6), سبعة (7), ثمانية (8), تسعة (9)
```

---

## 💻 Usage Examples

### Query Dataset

```python
import pandas as pd

# Load manifest
df = pd.read_csv("dataset/manifest.csv")

# Get Egyptian Arabic utterances
egy = df[df['dialect'] == 'EGY']

# Get male speakers only
male_speakers = df[df['gender'] == 'male']

# Get specific token
yes_utterances = df[df['token'] == 'نعم']

# Combined query
egy_male_words = df[
    (df['dialect'] == 'EGY') &
    (df['gender'] == 'male') &
    (df['class'] == 'word')
]
```

### Load Audio

```python
import soundfile as sf

# Read audio file
audio, sr = sf.read("dataset/EGY/word/نعم/EGY_spk001_word_نعم_take01.wav")

print(f"Duration: {len(audio)/sr:.2f}s")
print(f"Sample rate: {sr} Hz")
```

### Export Subset

```bash
# Export only Egyptian dialect
python examples/export_subset.py \
    --output my_subset \
    --dialect EGY

# Export random sample of 100 files
python examples/export_subset.py \
    --output small_sample \
    --sample 100

# Export specific speakers
python examples/export_subset.py \
    --output speaker_subset \
    --speakers spk001 spk002 spk003
```

---

## 🔧 Advanced Usage

### Generate Specific Dialect Only

```bash
python generate_dataset.py --dialect EGY
```

### Resume Interrupted Generation

```bash
python generate_dataset.py --resume
```

### Validation Only

```bash
python generate_dataset.py --validate-only
```

### Test Specific Voice

```bash
python discover_voices.py --test-voice <voice-id>
```

### List Available Voices

```bash
python discover_voices.py --list-only
```

---

## 📊 Quality Assurance

All generated utterances undergo automated validation:

- ✅ File count verification (per dialect)
- ✅ Audio format compliance (16kHz mono, 16-bit)
- ✅ Manifest completeness check
- ✅ File existence validation
- ✅ Audio quality metrics

See `dataset/validation_report.txt` for detailed QA results.

---

## 🔑 Configuration

### Environment Variables (.env)

```bash
CARTESIA_API_KEY=your_api_key_here
TARGET_SAMPLE_RATE=16000
TARGET_PEAK_DBFS=-3.0
SILENCE_PADDING_MS=150
```

### Customization (config.py)

- Modify `DIALECTS` to add/remove dialects
- Update `WORDS` and `NUMBERS` for different vocabulary
- Adjust audio processing parameters
- Configure rate limiting behavior

---

## ⚙️ Requirements

### System Dependencies

- **Python**: 3.8+
- **ffmpeg**: Required for audio processing

### Python Packages

```
requests>=2.31.0
pandas>=2.0.0
soundfile>=0.12.1
pydub>=0.25.1
numpy>=1.24.0
tqdm>=4.65.0
python-dotenv>=1.0.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 📝 File Naming Convention

Format: `<DIALECT>_<SPKID>_<CLASS>_<TOKEN>_take<NN>.wav`

**Example:** `EGY_spk001_word_التنشيط_take01.wav`

- **DIALECT**: 3-letter code (EGY, LAV, GLF, MOR, TUN)
- **SPKID**: Speaker ID (spk001-spk125)
- **CLASS**: Token class (word, number)
- **TOKEN**: Arabic text
- **NN**: Take number (01-99)

---

## ⏱️ Generation Timeline

| Phase | Duration | Task |
|-------|----------|------|
| Setup | 15 min | Install dependencies, configure API |
| Voice Discovery | 30 min | Test voices, create mappings |
| **Dataset Generation** | **4-5 hours** | Generate 2,500 utterances |
| Validation | 15 min | Run QA checks |

**Total: ~5-6 hours** (mostly automated API calls)

---

## 💰 Cost Estimation

- **Total API calls**: ~2,500 TTS requests
- **Cartesia pricing**: Check [Cartesia Pricing](https://cartesia.ai/pricing)
- **Estimated cost**: Varies by plan (check current rates)

---

## 🚨 Troubleshooting

### "CARTESIA_API_KEY not set"

```bash
cp .env.template .env
nano .env  # Add your API key
```

### "ffmpeg not found"

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

### Rate Limiting (429 errors)

- Normal during generation
- Script includes automatic retry with exponential backoff
- Generation will continue, just slower

### "Voice mapping not found"

```bash
python discover_voices.py
```

---

## 📚 Documentation

- **Setup Guide**: [SETUP.md](SETUP.md)
- **API Documentation**: [Cartesia Docs](https://docs.cartesia.ai/)
- **Dataset README**: `dataset/README.md` (auto-generated)
- **Specification**: [agent.md](agent.md)

---

## 🎯 Use Cases

- **ASR Training**: Train Arabic speech recognition models
- **IVR Systems**: Banking/customer service voice interfaces
- **Dialect Classification**: Build dialect identification models
- **TTS Evaluation**: Benchmark TTS quality across dialects
- **Linguistic Research**: Study Arabic dialect variations

---

## ⚠️ Limitations

1. **Synthetic Speech**: TTS-generated, not human recordings
2. **Dialect Approximation**: Voices mapped to dialects, may not be native speakers
3. **Limited Vocabulary**: 20 tokens (banking domain specific)
4. **Speaker Simulation**: Voice rotation, not unique individuals

---

## 📄 License

Dataset generated using Cartesia AI's Sonic 3 TTS API. Review Cartesia's [Terms of Service](https://cartesia.ai/terms) for usage restrictions.

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Additional dialects (Iraqi, Sudanese, etc.)
- Expanded vocabulary
- Voice quality ratings
- Audio augmentation pipeline
- ML training examples

---

## 📧 Support

- **Cartesia Support**: support@cartesia.ai
- **API Docs**: https://docs.cartesia.ai/
- **Issues**: Check project repository

---

## 📖 Citation

```bibtex
@dataset{arabic_dialect_tts_2025,
  title={Arabic Dialect Speech Dataset: Natural TTS with Cartesia Sonic 3},
  year={2025},
  note={2,500 utterances across 5 dialects using Cartesia Sonic 3 TTS}
}
```

---

**Generate natural, expressive Arabic speech for authentic dialect representation!** 🎙️✨

Built with [Cartesia Sonic 3](https://cartesia.ai/) - State-of-the-art multilingual TTS
