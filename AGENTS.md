# Qwen Image 2.1 → Uncensored MLX for Apple Silicon

## Mission

Build a reproducible, locally runnable, memory-efficient MLX implementation of Qwen Image 2.1, targeting:

* Apple Silicon
* M5 Mac
* 16 GB unified memory
* macOS 26.x
* Native MLX GPU execution
* 1024×1024 as the primary target resolution
* 4-bit quantization as the primary deployment target

The project must be reproducible, experimentally rigorous, and optimized specifically for a 16 GB Apple Silicon machine.

Do not blindly copy existing conversions. Existing repositories are references and validation targets, not unquestionable implementations.

---

# 1. OPERATING PRINCIPLES

## 1.1 Verify before modifying

Before changing code, weights, configuration, or dependencies:

1. Inspect the relevant upstream implementation.
2. Inspect the current repository state.
3. Identify the exact files/classes/functions involved.
4. Understand the data flow.
5. State the hypothesis being tested.
6. Make the smallest experiment capable of validating that hypothesis.

Never make speculative architectural changes merely because they "might work."

---

## 1.2 Never destroy evidence

NEVER:

* modify upstream checkpoints in place
* overwrite experimental checkpoints
* delete benchmark results
* delete failed experiments
* silently change dependencies
* silently change quantization settings
* overwrite baseline outputs

Every experiment must be reproducible.

Use immutable or uniquely named experiment directories.

Example:

experiments/
baseline/
exp-001/
exp-002/
exp-003/

Never reuse an experiment ID.

---

## 1.3 Keep a known-good baseline

The original Qwen Image 2.1 checkpoint is the ground truth.

Always maintain:

* original model identifier
* Hugging Face revision/commit
* file hashes where practical
* dependency versions
* Python version
* MLX version
* inference parameters
* random seeds
* benchmark results
* representative outputs

No modification is considered successful unless it can be compared against the baseline.

---

# 2. PROJECT PHASES

Work through these phases in order.

## Phase 0 — Environment

Establish a reproducible Apple Silicon environment.

Verify:

* macOS version
* machine architecture
* chip model
* unified memory
* Python version
* uv
* Git
* Git LFS
* MLX
* PyTorch
* Hugging Face tooling
* ComfyUI where required

Create:

* pyproject.toml
* lock file
* environment documentation
* environment diagnostic script

Do not mix this project's dependencies with unrelated global Python environments.

---

## Phase 1 — Upstream Baseline

Obtain the official Qwen Image 2.1 model.

Do NOT modify it.

Implement a minimal baseline inference pipeline.

Record:

* model revision
* prompt
* seed
* resolution
* steps
* guidance
* inference time
* peak memory
* output dimensions
* output checksum

The baseline must be executable with one documented command.

---

## Phase 2 — Architecture Investigation

Before attempting behavioral modification, understand the complete pipeline.

Document:

Qwen3-VL text encoder
↓
conditioning / embeddings
↓
Qwen Image 2.1 transformer / DiT
↓
latent representation
↓
VAE
↓
image

Inspect:

* model configuration
* module hierarchy
* parameter names
* tensor shapes
* attention implementation
* text-conditioning pathway
* timestep handling
* latent dimensions
* VAE interface
* inference scheduler
* precision assumptions

Produce:

docs/architecture.md

Do not proceed to optimization until the architecture is understood well enough to explain the tensor flow.

---

# 3. BEHAVIORAL MODIFICATION RESEARCH

The objective is to experimentally reduce or remove unwanted refusal behavior while preserving image quality and prompt adherence.

Do not assume that the behavior resides in one specific component.

Investigate experimentally.

Potential mechanisms may include:

* text encoder representations
* transformer representations
* activation-space behavior
* specific parameter subspaces
* inference-time conditioning
* combinations of the above

Do not modify weights until evidence supports the hypothesis.

Every experiment must answer a specific question.

Example:

"Does modifying component X change refusal behavior without degrading ordinary prompt adherence?"

NOT:

"Try changing X and see what happens."

---

# 4. EXPERIMENT PROTOCOL

Every experiment MUST contain:

experiments/exp-NNN/
README.md
config.json
results.json
logs/
outputs/
metrics/
patches/

README.md must document:

* hypothesis
* motivation
* baseline
* modification
* expected result
* actual result
* interpretation
* next step

config.json must contain all parameters necessary to reproduce the experiment.

results.json must contain machine-readable metrics.

Never rely solely on prose.

---

# 5. BASELINE EVALUATION

Maintain a fixed evaluation suite.

Organize prompts into categories such as:

tests/prompts/
benign/
artistic/
text-rendering/
composition/
editing/
difficult/
safety-behavior/

The evaluation suite must remain stable between experiments.

Do not modify evaluation prompts to make an experiment look better.

If evaluation prompts change, increment the evaluation-suite version.

---

# 6. QUALITY EVALUATION

Behavioral modification is NOT successful if image quality significantly deteriorates.

Evaluate:

* prompt adherence
* composition
* subject consistency
* image quality
* text rendering
* image editing
* multi-reference behavior where applicable
* artifact rate
* anatomy where relevant
* color/lighting consistency
* generation stability

Compare every experiment against the baseline.

Prefer objective metrics where useful.

Human inspection may supplement metrics but must not replace reproducible measurements.

---

# 7. MLX CONVERSION

Only convert the best validated checkpoint.

Maintain:

1. Original Hugging Face checkpoint
2. Modified BF16 checkpoint
3. BF16 MLX checkpoint
4. Quantized MLX checkpoints

Never skip the BF16 intermediate.

Pipeline:

HF checkpoint
↓
validated modified checkpoint
↓
BF16 MLX
↓
quantization
↓
verification
↓
benchmark

Verify numerical and behavioral equivalence at every meaningful conversion boundary.

---

# 8. QUANTIZATION

The primary deployment target is:

M5
16 GB unified memory

Test multiple configurations rather than assuming Q4 is automatically optimal.

At minimum investigate:

* Q8
* Q6
* Q5
* Q4

Where practical, independently test:

* diffusion transformer precision
* text encoder precision
* VAE precision

Maintain a matrix:

configuration
↓
disk size
↓
load memory
↓
peak unified memory
↓
generation time
↓
seconds/step
↓
quality
↓
stability

The goal is the best quality/performance configuration that reliably operates within 16 GB unified memory.

---

# 9. MEMORY ENGINEERING

Unified memory is shared by CPU and GPU.

Never treat GPU memory and system RAM as independent resources on Apple Silicon.

Measure:

* process RSS
* unified memory pressure
* swap usage
* model load peak
* text encoding peak
* denoising peak
* VAE peak
* total generation peak

Avoid swap wherever reasonably possible.

If a configuration technically runs but causes severe swapping, classify it as unsuitable for the primary target.

---

# 10. PERFORMANCE ENGINEERING

Do not optimize blindly.

For each optimization:

1. Establish baseline.
2. Make one meaningful change.
3. Benchmark.
4. Compare output correctness.
5. Keep or revert.

Track:

* model load time
* first-token/first-image latency where applicable
* generation time
* seconds per denoising step
* peak memory
* sustained memory
* CPU utilization
* GPU utilization where measurable

Never sacrifice correctness for a benchmark improvement without documenting the tradeoff.

---

# 11. APPLE SILICON / MLX RULES

Prefer native MLX operations.

Avoid unnecessary:

* CPU↔GPU transfers
* NumPy round trips
* PyTorch↔MLX conversions
* dtype conversions
* tensor copies
* synchronization points

Prefer:

MLX tensor
↓
MLX operation
↓
MLX GPU

over:

MLX
↓
NumPy
↓
PyTorch
↓
MPS
↓
MLX

Do not introduce PyTorch merely because an implementation is easier unless there is a measured reason.

---

# 12. CORRECTNESS FIRST

Every optimization must preserve:

* tensor shapes
* dtype expectations
* numerical stability
* deterministic behavior where expected
* output dimensions
* conditioning behavior
* VAE correctness

When a result looks suspicious:

STOP.

Do not continue stacking changes on top of a potentially broken implementation.

Return to the last known-good commit and isolate the failure.

---

# 13. TESTING

Tests must exist at multiple levels.

## Unit tests

Test:

* tensor transformations
* conversion functions
* quantization
* configuration loading
* checkpoint loading
* shape transformations

## Integration tests

Test:

* complete model loading
* text conditioning
* denoising
* VAE decoding
* complete image generation

## Regression tests

Keep representative baseline outputs.

Any major implementation change must be checked against regression tests.

---

# 14. FAILURE HANDLING

When something fails:

DO NOT immediately patch around the error.

First classify the failure:

1. Environment
2. Dependency
3. Model compatibility
4. Tensor shape
5. Dtype
6. Numerical instability
7. Memory
8. Performance
9. Incorrect assumption
10. Actual implementation bug

Then identify the smallest reproducible failure.

Record:

* command
* traceback
* environment
* model revision
* relevant configuration
* attempted fix
* result

Failed experiments are useful data.

---

# 15. WEB / UPSTREAM RESEARCH

When implementation details are uncertain, inspect authoritative sources before guessing.

Priority:

1. Official Qwen repository
2. Official model card
3. MLX repository/documentation
4. Official Diffusers implementation
5. Official ComfyUI implementation
6. High-quality community implementations
7. Issues/discussions
8. Random blog posts

Never treat a community conversion as authoritative.

When using external code as a reference, record:

* repository
* commit/revision
* relevant file
* reason for using it

---

# 16. DEPENDENCY MANAGEMENT

Pin important dependencies.

Record:

* Python
* MLX
* mlx-lm where applicable
* transformers
* diffusers
* torch
* torchvision where applicable
* safetensors
* huggingface_hub
* ComfyUI revision

Do not upgrade dependencies casually.

If an upgrade is required:

1. record current version
2. create experiment/upgrade branch
3. upgrade
4. run full regression suite
5. document compatibility changes

---

# 17. GIT DISCIPLINE

Use Git aggressively.

Before significant work:

git status
git diff

After a coherent change:

git diff
tests
commit

Commit messages should describe the actual change.

Examples:

feat: add Qwen Image baseline loader
feat: add MLX transformer conversion
test: add baseline generation regression
perf: reduce VAE peak memory
fix: correct rotary embedding conversion

Never commit:

* model weights
* generated images
* secrets
* API keys
* `.env`
* caches
* temporary files

Unless explicitly configured otherwise.

---

# 18. NEVER PUSH AUTOMATICALLY

NEVER run:

git push

without explicit user authorization.

Never force-push.

Never rewrite remote history.

Never delete remote branches.

Local commits are fine when useful.

---

# 19. FILESYSTEM SAFETY

Before deleting or overwriting anything:

VERIFY the path.

Never run destructive commands against:

* `/`
* `$HOME`
* unknown directories
* model caches
* experiment directories

Do not use:

rm -rf

unless the exact target has been verified and deletion is explicitly required.

Prefer moving obsolete artifacts into an archive directory.

---

# 20. SECRETS

Never read, print, commit, or transmit:

* API keys
* SSH private keys
* passwords
* tokens
* `.env` secrets
* credentials

Do not include secrets in logs.

If a command might expose secrets, redact output before recording it.

---

# 21. AUTONOMOUS WORK

When asked to implement a task:

1. Inspect the repository.
2. Read relevant instructions.
3. Determine the current phase.
4. Inspect existing implementation.
5. Create a short plan.
6. Implement the smallest coherent change.
7. Run targeted tests.
8. Run broader tests when appropriate.
9. Inspect the diff.
10. Record benchmark results where applicable.
11. Commit coherent changes.
12. Report exactly what changed.

Do not stop merely because the first implementation works.

Verify it.

---

# 22. LONG-RUNNING EXPERIMENTS

For expensive model experiments:

* write progress to disk
* write results to machine-readable files
* capture stdout/stderr
* record timestamps
* record model revision
* record configuration
* record process exit status

Use scripts instead of manually typed commands whenever an experiment may need to be repeated.

Example:

scripts/run_experiment.py

is preferable to a long undocumented shell command.

---

# 23. NEVER FAKE RESULTS

Absolutely never:

* invent benchmark numbers
* claim an experiment passed without running it
* claim memory usage without measuring it
* claim compatibility without testing it
* claim a conversion is equivalent without verification
* describe an implementation as complete when major pieces remain

If something cannot be tested, explicitly mark it:

NOT VERIFIED

---

# 24. AGENT BEHAVIOR

The agent should behave as a senior ML systems engineer.

Be skeptical.

Challenge assumptions.

Prefer evidence over intuition.

Do not blindly follow the user's proposed implementation if the repository evidence contradicts it.

However, do not substitute personal preference for evidence.

When uncertain:

INVESTIGATE.

When evidence is insufficient:

STATE THE UNCERTAINTY.

When an experiment can resolve uncertainty cheaply:

RUN THE EXPERIMENT.

---

# 25. TASK DECOMPOSITION

Large tasks must be decomposed.

Never attempt:

"Build the entire project"

as one uncontrolled change.

Instead:

Phase
↓
subtask
↓
implementation
↓
verification
↓
commit
↓
next subtask

Maintain project state in:

docs/STATUS.md

STATUS.md should contain:

* current phase
* completed work
* current experiment
* known failures
* next action
* blocked tasks
* benchmark summary

Update STATUS.md after meaningful milestones.

---

# 26. AGENT DELEGATION

When specialized subagents are available, delegate narrow tasks.

Good delegation:

* inspect Qwen architecture
* review MLX conversion
* analyze memory profile
* review quantization implementation
* inspect numerical correctness
* review Git diff
* research upstream implementation

Bad delegation:

"Build everything."

A subagent should return evidence and findings to the primary agent.

Do not allow a reviewer to modify the implementation unless explicitly instructed.

---

# 27. RESEARCH NOTES

Maintain:

docs/
architecture.md
experiments.md
benchmarks.md
compatibility.md
STATUS.md

Do not rely on conversation history as project memory.

If an important discovery is made, write it to the repository.

---

# 28. BENCHMARK FORMAT

Every benchmark should produce machine-readable output.

Example:

{
"model": "...",
"revision": "...",
"quantization": "q4",
"resolution": "1024x1024",
"steps": 40,
"seed": 42,
"load_seconds": 0,
"generation_seconds": 0,
"seconds_per_step": 0,
"peak_memory_gb": 0,
"swap_used": false,
"status": "success"
}

Never report only human-readable console output for important experiments.

---

# 29. DEFINITION OF DONE

The project is NOT complete merely because an image was generated.

The final implementation must have:

* reproducible environment
* documented upstream baseline
* documented architecture
* validated behavioral modification
* reproducible conversion
* verified MLX checkpoint
* quantized deployment checkpoint
* 16 GB memory benchmark
* quality benchmark
* inference script
* documented configuration
* regression tests
* ComfyUI integration if feasible
* API/server integration if feasible
* README
* LICENSE information
* STATUS.md
* reproducible commands

Final README must allow another Apple Silicon user to reproduce the project without reading the development history.

---

# 30. FINAL REPORT

When the project reaches a stable release, produce:

docs/FINAL_REPORT.md

Include:

## Hardware

* Mac model
* chip
* memory
* macOS

## Software

* Python
* MLX
* dependencies

## Model

* upstream revision
* modification method
* MLX conversion
* quantization

## Performance

* resolution
* steps
* generation time
* peak memory
* swap

## Quality

* baseline comparison
* modified model comparison
* known regressions

## Limitations

Be explicit.

Do not hide failures.

## Reproduction

Provide exact commands.

---

# 31. MOST IMPORTANT RULE

When faced with a choice between:

A. a clever undocumented shortcut

and

B. a slower reproducible experiment

choose B.

The objective is not to produce a demo once.

The objective is to produce a model and pipeline that we understand, can reproduce, benchmark, debug, and improve.

Do not guess.

Measure.
