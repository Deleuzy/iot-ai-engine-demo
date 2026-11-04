# iot-ai-engine-demo

A demo for an IoT product team. It takes **real, public sensor readings** from 55 wireless devices, checks their
quality with explicit rules, looks for faults and forecasts battery voltage per device, and then explains each
flagged event with a model-independent AI engine: retrieval from product and support documents, function calling
to read device history, and guardrails that check the answer in code before it is returned.

The data is real. The faults are **injected** on purpose, so every detector can be scored against a known answer
key. The product documents and the AI backends are written or wired for this demo. Nothing here runs against
any Aurocon system or device. See [What is real and what is simulated](#what-is-real-and-what-is-simulated).

## Key results

Every number below is copied from a file in [`results/`](results/), written by `python -m iot_ai run` and
`python -m iot_ai eval-engine`. Seed 20260401. Test split: the last 30 % of the time window, 2004-03-26 07:10
onwards, 92,675 ten-minute bins across 55 devices, of which 9,350 are inside an injected fault.

| What | Result | File |
|---|---|---|
| Raw readings parsed from the public file | 2,313,156 of 2,313,682 lines (526 malformed lines skipped) | [validation_raw.json](results/validation_raw.json) |
| Clock errors in raw readings | 227,391 readings step back in time against the device sequence; 4 fall outside the study window | [validation_raw.json](results/validation_raw.json) |
| Bins with no reading (10-minute grid, 55 devices) | 102,217 of 308,880 (33 %) | [validation_grid_real.json](results/validation_grid_real.json) |
| Detector: rule-based (all four rules) | precision 0.101, recall 0.990, **F1 0.183**; per-device median F1 0.183 | [detection_metrics.json](results/detection_metrics.json) |
| Detector: rules without the "missing" rule | precision 0.082, recall 0.259, F1 0.1245 | [detection_metrics.json](results/detection_metrics.json) |
| Detector: rolling z-score (6 h, k = 3) | precision 0.214, recall 0.047, F1 0.077 | [detection_metrics.json](results/detection_metrics.json) |
| Detector: isolation forest (unsupervised) | precision 0.079, recall 0.250, F1 0.120 | [detection_metrics.json](results/detection_metrics.json) |
| Detector: gradient boosting (trained on injected faults, threshold 0.8) | precision 0.109, recall 0.341, **F1 0.165**; precision 0.822 when flags on unlabelled real faults are removed | [detection_metrics.json](results/detection_metrics.json) |
| Battery voltage forecast, one hour ahead, MAE | persistence **0.0068 V**, gradient boosting 0.0086 V; **the model does not beat persistence** | [forecast_metrics.json](results/forecast_metrics.json) |
| Engine retrieval, 12 labelled queries | hit@1 **0.50**, hit@3 **0.67** | [engine_eval.json](results/engine_eval.json) |
| Engine, honest mock backend, 40 events | 40 of 40 answers pass all checks; 242 numbers checked against tool output | [engine_eval.json](results/engine_eval.json) |
| Engine guardrails, four broken backends, 160 answers | 160 of 160 caught (wrong number, invented citation, no tool call, unknown tool) | [engine_eval.json](results/engine_eval.json) |
| Tests | 36 pytest tests, ruff and strict mypy clean | CI |

How to read the detector rows. Precision is measured against the injected faults only. The real readings also
contain faults nobody labelled (out-of-range temperatures such as 120 °C, long stuck runs, clock problems), and a
detector that flags them counts as a false positive. The column "precision excluding unlabelled real faults" removes
those flags. It is circular for the rule rows, which flag exactly those bins, so read it for the model rows only.

## What the results say

- **No detector is good on this test.** The best F1 belongs to the rule set, and it is inflated by the "missing"
  rule: a third of all bins have no reading in the real data, so that rule flags almost everything, and its
  recall on dropouts and on the voltage sag is high for that reason alone. Without it, the rules find 35 % of
  spikes and 37 % of stuck runs.
- **The gradient-boosting model does not beat the rules on F1.** Without the missing rule, the rules find no dropouts
  at all, while the model finds 32 % of them, and it finds about as many stuck and spike bins. Its overall F1 is still
  lower than the full rule set, and its raw precision looks low mainly because it also flags real faults that nobody
  labelled.
- **The battery forecast does not beat persistence.** Voltage changes by a few millivolts per hour, so repeating
  the last value is already very good. A model would need more history per device or a different target.
- **Two data findings matter more than the models.** About a third of device-bins are silent, and about one reading
  in ten has a timestamp that disagrees with the device sequence. A production detector has to treat silence and
  clock faults as their own problems, not as noise to smooth over.

## What is real and what is simulated

| Part | Real or simulated | Notes |
|---|---|---|
| Sensor readings | **Real** | Intel Berkeley Research Lab, 54 motes, 28 Feb to 5 Apr 2004: temperature, humidity, light, battery voltage. Downloaded from `db.csail.mit.edu` with its SHA-256 checked against the value pinned in `src/iot_ai/config.py`. The file is not committed. |
| Device identity and placement | Real ids, no product meaning | Motes are called devices. Their locations file is downloaded and checksummed but not used. |
| Faults | **Simulated, injected** | Stuck, spike, dropout and voltage sag, written into the test and training values with a fixed seed. Each event is recorded in `data/processed/injected_events.json`. |
| Validation rules | Real code, fixed thresholds | Written before the results were looked at. Thresholds are in `src/iot_ai/config.py`. |
| Product and support documents (`knowledge/`) | **Written for this demo** | They describe a generic runbook. They are not Aurocon or vendor documentation. |
| Model backends | Mock by default; real providers wired but not run in CI | The mock is deterministic and offline. Anthropic, Mistral and an OpenAI-compatible endpoint (vLLM) are implemented with plain HTTP and tested against a mock transport. They have not been called with a real key in this repository. |
| Embeddings | Hashing embedder by default | Deterministic, no key. It is a bag of words and word pairs, which explains the modest retrieval scores. |
| Vector store | In-memory numpy matrix | Exact cosine search over a few dozen passages. Not a database, and no scale claim. |

## Method

**Validation** ([validate.py](src/iot_ai/validate.py), [config.py](src/iot_ai/config.py)). Rules are plain tests with
fixed thresholds: missing channel, out-of-range value (temperature -10 to 50 °C, humidity 0 to 100 %, voltage 1.8 to
3.3 V), stuck (the same temperature for six 10-minute bins), spike (more than 5 °C away from the centred 70-minute
median), and clock errors (timestamp outside the window, step back, or a gap of more than 60 minutes). The raw
counts and the grid counts are saved separately.

**Grid and split** ([prepare.py](src/iot_ai/prepare.py)). Readings are binned into 10-minute bins per device with the
median of the readings in each bin, so one bad value does not move a bin. The first 70 % of the window is training
data and the rest is test data. One cut-off applies to all devices.

**Injection** ([prepare.py](src/iot_ai/prepare.py)). Each device gets, in each split, three stuck runs of 3 hours,
four 20-minute spikes of +8 °C, three 2-hour dropouts, and two 6-hour voltage sags of -0.3 V. Each fault gets its own
segment of the split, so faults never overlap. The labels are per bin.

**Detectors** ([detect.py](src/iot_ai/detect.py)). The threshold of each learned or tuned detector is chosen on the
last 15 % of the training part, never on the test split. Models are fitted without bins where the rules see a real
fault that is not an injected label, because those bins would teach the model that bad readings are normal.

**Features** ([features.py](src/iot_ai/features.py)). Every feature uses the current and earlier bins only. A unit test
changes future values and checks that no earlier feature moves.

**Forecast** ([forecast.py](src/iot_ai/forecast.py)). Battery voltage one hour ahead, scored only on clean test rows
(no injected fault in the input or target window).

## AI engine

The engine is one class, [engine.py](src/iot_ai/ai/engine.py), that talks to a `Backend` with one method,
`turn(messages, tools)`. Backends:

- [`MockBackend`](src/iot_ai/ai/backends.py): deterministic, offline, used in tests and CI. Its bad modes exist to
  test the guardrails.
- `AnthropicBackend`: Messages API with tool use.
- `OpenAICompatBackend`: any `/chat/completions` endpoint with tools. Mistral's API and a vLLM server (Qwen, Llama)
  use this class.

Flow for one event:

1. **Retrieve** the four most similar passages from [`knowledge/`](knowledge/) (vector store, cosine similarity).
2. **Call tools** the model asks for: `get_device_history` (temperature, humidity, voltage slope, silence, rule
   counts over a window) and `get_device_events`. Arguments are type-checked; unknown tools are refused.
3. **Check the answer in code** ([guardrails.py](src/iot_ai/ai/guardrails.py)):
   - it is JSON with a bounded summary and a suggested action from a fixed list;
   - at least one tool was called and no tool call failed;
   - every cited passage was retrieved for this request;
   - every number in the summary matches a number a tool or the event returned, at the precision written.
     ISO dates and times are skipped; nothing else is.
4. If a check fails, the engine returns a **template explanation** built from the event fields and records the
   findings. It never returns an unchecked model answer.

The engine never sees the injected labels or the fault kinds. It sees the device as a device would show it.

## Run it

Local, with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run python -m iot_ai run          # fetch (checksum), validate, detect, forecast; writes results/ and data/processed/
uv run python -m iot_ai eval-engine  # retrieval, honest backend, guardrail catch rate; writes results/engine_eval.json
uv run python -m iot_ai serve        # API on http://localhost:8000, see /docs
uv run ruff check . && uv run mypy && uv run pytest -q
```

Docker Compose runs the pipeline once, then starts the API when the pipeline has finished:

```bash
mkdir -p data results && chmod -R a+rwX data results   # the image runs as uid 10001
docker compose up --build
curl localhost:8000/events?limit=3
```

To use a real model, set `ENGINE_BACKEND` to `anthropic`, `mistral` or `openai_compat` and the matching key
(`ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`) or `ENGINE_BASE_URL` in the environment. Nothing in this repository needs a
key to pass its tests.

## Limits

- **Injected faults, not real faults.** Precision and recall are against faults the code wrote into the data. Real
  faults that nobody labelled are scored as false positives, and the adjusted column only partly corrects that.
- **One dataset from 2004, indoor lab motes.** It has no product meaning, no gateway, no installation metadata and no
  real battery replacement history. Nothing here says how a field deployment would behave.
- **The spike rule looks ahead.** It uses the centred median, so it cannot run in real time. The detectors that run
  causally are the z-score, the isolation forest and the gradient-boosting model.
- **The model is small and was not tuned for speed.** It does not beat the rules on F1 and does not beat persistence on
  voltage. Those are results, not a bug to hide.
- **The engine checks numbers and citations, not the quality of the wording.** A correct, well-grounded sentence can
  still give a poor recommendation. Human review stays with the technician who acts on it.
- **The broken backends are our own mock modes.** Catching 160 of 160 says the checks work against those four failures,
  not against every failure a real model can make. Real provider calls are implemented and unit-tested through a
  mock transport, but this repository has not run them against a live key.
- **Retrieval is weak on purpose.** The hashing embedder gets hit@1 0.50 on 12 queries. A learned embedding model
  would be the first thing to change.

## Layout

```
src/iot_ai/
  config.py        dataset URLs and checksums, every rule threshold, injection plan
  data.py          download with checksum, raw parsing
  validate.py      rules on raw readings and on the grid
  prepare.py       grid, split, fault injection with labels
  features.py      causal features
  detect.py        baselines, isolation forest, gradient boosting, scoring
  forecast.py      one-hour voltage forecast
  pipeline.py      everything in results/ and data/processed/
  ai/              embeddings, vector store, knowledge loader, tools, backends, guardrails, engine, evaluation
  api.py           FastAPI: /health, /events, /explain, /results/detection
knowledge/         the five runbook documents used for retrieval
results/           saved outputs, the only source of the numbers above
tests/             36 tests, no network access needed
```

CI ([ci.yml](.github/workflows/ci.yml)) runs lint, strict mypy and the tests; then the pipeline on the real file and the
engine evaluation (artifacts uploaded); then Compose with a health check and one explanation request.
