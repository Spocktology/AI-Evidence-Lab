# AI Evidence Lab

A web app where you upload a photo or video. It reads the metadata, asks for whatever capture details the
metadata doesn't contain, lets you mark *where* the object or event is, runs forensic preprocessing, and then
has Claude evaluate the media. The result is a detailed report: annotated stills, ranked explanations with
likelihoods, AI-generated/CGI/edited probabilities with ranges, and an overall confidence percentage.

## How it works

1. **Upload (intake).** The original is hashed (SHA-256 and MD5) and stored read-only. Every action is
   written to a chain-of-custody log, and the hash is re-checked before each analysis and in the report.
2. **Metadata.** Uses `exiftool` (EXIF/XMP/IPTC/MakerNotes/QuickTime, including embedded drone and dashcam
   tracks) and `ffprobe` (codec, frame rate, variable frame rate, rotation, HDR, bitrate). It also checks:
   - Content Credentials (C2PA): signer, generator, declared AI source type.
   - Fingerprints left by AI generators and editors.
   - JPEG quantization tables: standard tables suggest a software re-save; custom tables usually come from
     camera firmware.
   - Consistency: editing software, date mismatches, platform re-encodes, low bitrate.
3. **Capture details.** A questionnaire pre-fills what the metadata answered and marks what is missing:
   device, resolution, frame rate, time, location and so on. It always asks:
   - Where the file came from (original, download, screenshot or screen recording).
   - Whether it was shot through glass.
   - How the camera was held or mounted.
   - Conditions, and what the witness saw with their own eyes.
4. **Mark the object.**
   - **Photo:** click or drag over a labelled grid (A1, B2, ...). Grid density can be 4 to 26 columns.
   - **Video:** set a start and end time in the player. Frames from that range are pulled from the
     original at full resolution (every frame if the range is short), and you mark grid cells on each frame.
     Cells where motion was detected are outlined as suggestions.
5. **Preprocessing.** Measurements made to help the AI:
   - **Full resolution where it matters.** Claude reads images natively up to 2,576 px on the long edge
     (about 3.7 MP) and shrinks anything bigger. So the app sends whole frames scaled down for context, and
     sends **your marked cells as 1:1 native-pixel crops** (split into tiles if large). Those crops are
     lossless PNG, so no compression is added.
   - Enhanced view of the region (local contrast and shadow lift), and a nearest-neighbour pixel zoom of
     small objects.
   - Maps and statistics:
     - Error Level Analysis.
     - Noise-residual map.
     - Frequency spectrum, which shows the periodic patterns typical of AI generators and resampling.
     - Per-cell statistics comparing the marked cells with the rest of the frame.
   - Automatic object measurement: size, colour, edge sharpness, and angular size when the focal length is
     known.
   - **Video:**
     - Camera motion is separated from object motion. The marked region is masked out, background features
       are fitted, and when the scene has no features (night sky) the app says the camera motion is unknown
       instead of guessing.
     - Object track: speed in px/s and deg/s, and how straight the path is.
     - **Blink-rate detection.** Aircraft strobes flash about 40 to 100 times a minute.
     - Light-trail stack and motion heat-map.
     - Integrity checks: identical repeated frames, frame-timing gaps, and sudden jumps in noise level
       (possible splices).
6. **AI analysis (two passes, to limit bias).**
   - **Pass 1 (blind):** Claude examines the media, metadata, measurements and *technical* capture facts.
     It does not see the witness account.
   - **Pass 2 (synthesis):** Claude compares the witness account with its blind findings, ranks
     explanations, gives authenticity probabilities with ranges, and writes the report.
7. **Report.**
   - Opens as an HTML page with Print / Save as PDF.
   - Can be downloaded as a single self-contained HTML file.
   - Can redact location data.
   - Includes the raw JSON.
   - Contains: every still Claude saw, with its annotations; observations; ranked explanations with
     supporting and contradicting evidence; metadata findings; measurements; limitations; next steps;
     questions for the witness; the chain of custody; the method; and the AI cost.

## Setup

You need **Python 3.10+**, **ffmpeg** and **exiftool**, and an **Anthropic API key**.

The API key comes from console.anthropic.com → API Keys. Usage is billed per use on the API account. A
Claude.ai subscription can't be used by a web app.

```bash
# macOS
brew install ffmpeg exiftool
# Ubuntu/Debian
sudo apt install ffmpeg libimage-exiftool-perl
# Windows: install ffmpeg and ExifTool and make sure both are on PATH

cp .env.example .env        # then edit .env and paste your ANTHROPIC_API_KEY
./run.sh                    # creates a virtualenv, installs deps, starts the server
# open http://localhost:8000
```

To run it with Docker instead:

```bash
docker build -t ai-evidence-lab .
docker run -p 8000:8000 --env-file .env -v "$PWD/data:/data" ai-evidence-lab
```

### Settings (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | Required. |
| `AEL_MODEL` | `claude-opus-5-5` | Claude model used for the analysis. |
| `AEL_EFFORT` | `high` | `low` / `medium` / `high` / `xhigh` / `max`. Higher is more thorough, slower and costs more. |
| `AEL_USE_FALLBACKS` | `1` | If the model declines a request, it is automatically re-run on a fallback model. |
| `AEL_DATA_DIR` | `./data` | Where cases are stored. Contains original media: keep it private and back it up. |
| `AEL_MAX_UPLOAD_MB` | `4096` | Upload size limit. |

### Cost

Each image the AI reads costs up to about 4,800 input tokens.
- **A photo analysis** sends roughly 8 to 12 images.
- **A video analysis** sends up to 40 images, depending on how many frames you pull.

At Opus 5.5 prices ($4 input / $20 output per million tokens), that is typically about **$0.30 to $1.50 per
case**. The exact token use and cost are printed in each report.

## Tests

```bash
pip install pytest httpx
python -m pytest -q
```

The tests generate a photo and a video of a moving, blinking light. They run the whole pipeline with the
Claude call stubbed out, so no API key is used. They check four things:
- The witness account is withheld from the blind pass.
- Native-resolution crops are sent.
- The object is located and tracked, including its speed and blink rate.
- The report and custody log are produced.

## Security and privacy

- The app has **no login**. Run it on your own machine (it binds to 127.0.0.1) or put it behind
  authentication before exposing it on a network.
- Media, metadata and your answers are sent to the Anthropic API for analysis. The app asks for consent
  before each run.
- GPS coordinates can reveal where someone lives. Use **Redact location data** before sharing a report.

## Limitations

- Percentages are calibrated estimates of belief, not lab measurements.
- AI/CGI detection from pixels alone is not definitive, especially for re-compressed social-media copies.
- RAW files (DNG and similar) are read through their embedded preview.
- The app can't look up astronomy, flight-tracking or weather data. The report says which of these to check.
- This is not a certified forensic examination.
