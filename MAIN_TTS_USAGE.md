# Cartesia Sonic 3 TTS - Tunisian Arabic Synthesis

## Professional-Grade Python TTS Script

**File:** `main.py`

**Purpose:** Runtime verification and synthesis of Tunisian Arabic speech using Cartesia Sonic 3 API with comprehensive validation and error handling.

---

## Features

✅ **Runtime Capability Discovery**
- Automatically discovers available models (Sonic 3, sonic-multilingual)
- Verifies Arabic language support
- Detects Tunisian dialect voices or falls back to generic Arabic

✅ **Intelligent Voice Selection**
- Priority 1: Tunisian-specific voices (ar-TN locale)
- Priority 2: Voices tagged with "Tunisian" metadata
- Priority 3: Generic Arabic fallback with warning

✅ **Professional Audio Validation**
- WAV format verification (mono, 24kHz, PCM16)
- Clipping detection (rejects audio exceeding safe thresholds)
- Duration and metadata validation

✅ **Robust Error Handling**
- Clear, actionable error messages
- HTTP error handling with response details
- Graceful fallbacks (SSML → plain text)
- Environment validation

---

## Requirements

**Python:** ≥3.10

**Dependencies:**
```bash
pip install httpx python-dotenv
```

**Environment Variables:**
- `CARTESIA_API_KEY`: Your Cartesia API key (required)

---

## Quick Start

### 1. Set Up Environment

```bash
# Create .env file
echo "CARTESIA_API_KEY=your_api_key_here" > .env

# Or export environment variable
export CARTESIA_API_KEY="your_api_key_here"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Synthesis

```bash
python main.py
```

**Expected Output:**
```
2025-11-08 14:30:00 [INFO] Cartesia Sonic 3 TTS - Tunisian Arabic Synthesis
...
2025-11-08 14:30:05 [INFO] ✓ Audio written to: /path/to/output_tn.wav
...
SUCCESS
Output file: /path/to/output_tn.wav
Model: sonic-multilingual
Voice: voice-id-here
Language: ar (Arabic)
Tunisian-specific: Yes
Text: شنوة أحوالك؟ اليوم باش نمشيو للحانوت نشريو شوية قهوة.
```

---

## Configuration

### Output Settings

Edit constants in `main.py`:

```python
# Output file
OUTPUT_FILE = "output_tn.wav"

# Audio format
TARGET_SAMPLE_RATE = 24000  # Hz
TARGET_CHANNELS = 1         # Mono
TARGET_BIT_DEPTH = 16       # PCM16

# Clipping threshold
CLIPPING_THRESHOLD = 32760  # Allow small margin
```

### API Configuration

```python
API_BASE = "https://api.cartesia.ai"
API_VERSION = "2024-06-10"
TIMEOUT = 30.0  # seconds
```

### Tunisian Sentence

```python
TUNISIAN_SENTENCE = "شنوة أحوالك؟ اليوم باش نمشيو للحانوت نشريو شوية قهوة."
# Translation: "How are you? Today we're going to the shop to buy some coffee."
```

---

## How It Works

### Step-by-Step Execution

1. **Environment Validation**
   - Checks `CARTESIA_API_KEY` is set
   - Validates not empty/placeholder

2. **Model Discovery**
   - Attempts `/models` and `/v1/models` endpoints
   - Verifies Sonic 3 or sonic-multilingual presence
   - Falls back to inferred model if endpoint unavailable

3. **Voice Discovery**
   - Fetches all voices from `/voices` endpoint
   - Filters for Arabic-capable voices
   - Returns count and list

4. **Voice Selection**
   - Searches for Tunisian-specific voices (locale: ar-TN)
   - Checks metadata tags for "Tunisian" keywords
   - Falls back to generic Arabic if no Tunisian voice found
   - Logs warning on fallback

5. **Audio Synthesis**
   - Builds TTS payload with SSML (prosody adjustments)
   - POSTs to `/tts/bytes` endpoint
   - Falls back to plain text if SSML unsupported
   - Returns WAV bytes

6. **Audio Validation**
   - Verifies WAV container format
   - Checks sample rate, channels, bit depth
   - Detects clipping (peak amplitude analysis)
   - Extracts duration metadata

7. **File Output**
   - Writes validated audio to `output_tn.wav`
   - Prints success summary with details

---

## API Integration Details

### Authentication

```python
headers = {
    "Cartesia-Version": "2024-06-10",
    "X-API-Key": api_key,
    "Content-Type": "application/json"
}
```

### TTS Request Payload

```json
{
  "model_id": "sonic-multilingual",
  "transcript": "<speak xml:lang=\"ar\"><prosody rate=\"98%\" pitch=\"+0.5st\">شنوة أحوالك؟...</prosody></speak>",
  "language": "ar",
  "voice": {
    "mode": "id",
    "id": "voice-id-here"
  },
  "output_format": {
    "container": "wav",
    "encoding": "pcm_s16le",
    "sample_rate": 24000
  },
  "inference": {
    "proposer_mode": "last"
  }
}
```

**Note:** `inference.proposer_mode` is an assumption for "last proposer mode" - may not be supported by all models.

---

## Error Handling

### Common Errors

**Missing API Key:**
```
ERROR: Missing required environment variable: CARTESIA_API_KEY
Set CARTESIA_API_KEY in .env file or environment
```
**Solution:** Create `.env` file or export variable

**No Arabic Voices:**
```
ERROR: No Arabic voices available
Verify Arabic support in Cartesia dashboard
```
**Solution:** Check model/voice availability in dashboard

**HTTP 401 Unauthorized:**
```
HTTP 401 error for /voices
Response: {"error": "Invalid API key"}
```
**Solution:** Verify API key is correct and active

**Audio Clipping:**
```
ERROR: Audio clipping detected (peak: 32767/32767)
Reduce gain or request lower volume from API
```
**Solution:** Adjust prosody settings or contact Cartesia support

---

## Assumptions & Notes

### API Assumptions (marked in code)

1. **Model Endpoint:** `/models` or `/v1/models`
   - *May differ in actual API; script tries both*

2. **Last Proposer Mode:** `inference.proposer_mode = "last"`
   - *Experimental feature; may not be supported*

3. **SSML Support:** Prosody tags for expressivity
   - *Falls back to plain text if unsupported*

4. **Voice Response Format:** JSON with `id`, `name`, `language` fields
   - *Handles variations in structure*

### Design Decisions

- **Mono Audio:** Telephony/ASR applications typically use mono
- **24kHz Sample Rate:** Balance between quality and file size
- **PCM16 Encoding:** Standard uncompressed format, widely compatible
- **98% Rate, +0.5st Pitch:** Subtle adjustments for natural, warm speech

---

## Testing & Validation

### Manual Testing

```bash
# Run synthesis
python main.py

# Check output file
file output_tn.wav
# Expected: RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit, mono 24000 Hz

# Play audio (requires audio player)
ffplay -autoexit output_tn.wav
# Or
play output_tn.wav
```

### Automated Validation

The script performs automatic validation:
- ✓ WAV format verification
- ✓ Sample rate/channels/bit-depth check
- ✓ Clipping detection (peak < 32760)
- ✓ Duration extraction
- ✓ File existence

---

## Troubleshooting

### Script Fails with "No models found"

**Cause:** Model discovery endpoint not available or model name changed

**Solution:**
1. Check Cartesia dashboard for available models
2. Update `TARGET_MODELS` list in `main.py`
3. Verify API version is correct

### No Tunisian Voice Warning

**Cause:** No voice with ar-TN locale or Tunisian tags

**Expected:** Script continues with generic Arabic voice

**Action:**
- Check warning message
- Verify fallback voice is acceptable
- Contact Cartesia to request Tunisian voice

### SSML Not Supported Error

**Cause:** Model doesn't support SSML prosody tags

**Expected:** Script automatically falls back to plain text

**Action:** No action needed - fallback is automatic

---

## Advanced Usage

### Custom Text Synthesis

Edit `TUNISIAN_SENTENCE` constant:

```python
TUNISIAN_SENTENCE = "Your Tunisian Arabic text here"
```

### Multiple Outputs

Create a loop in `main()`:

```python
sentences = [
    "شنوة أحوالك؟",
    "اليوم باش نمشيو للحانوت",
    # ... more sentences
]

for i, sentence in enumerate(sentences):
    output_file = f"output_tn_{i+1:03d}.wav"
    # ... synthesis logic
```

### Voice Testing

Use existing `discover_voices.py` for voice exploration:

```bash
# List all Arabic voices
python discover_voices.py --list-only

# Test specific voice
python discover_voices.py --test-voice "voice-id-here"
```

---

## Integration with Dataset Generator

This script is part of the larger **Dialect_Dataset** project.

**Related Files:**
- `config.py` - Shared configuration
- `discover_voices.py` - Voice discovery tool
- `generate_dataset.py` - Batch dataset generation
- `examples/` - Usage examples

**Workflow:**
1. Use `main.py` for single synthesis and testing
2. Use `discover_voices.py` to find suitable voices
3. Use `generate_dataset.py` for batch generation

---

## License & Credits

**Author:** Senior Python TTS Engineer
**Framework:** Cartesia Sonic 3 API
**Language:** Tunisian Arabic (Eastern North African Dialect)

---

## Support

**Issues:** If script fails, check:
1. API key is valid
2. Dependencies are installed
3. Internet connection is active
4. Cartesia API status (check dashboard)

**Logs:** All operations are logged to console with timestamps and severity levels.

**Debug Mode:** Set `LOG_LEVEL` environment variable:
```bash
export LOG_LEVEL=DEBUG
python main.py
```

---

## Version History

**v1.0** (2025-11-08)
- Initial release
- Sonic 3 / sonic-multilingual support
- Tunisian Arabic voice selection
- WAV validation and clipping detection
- SSML with prosody adjustments
- Comprehensive error handling
