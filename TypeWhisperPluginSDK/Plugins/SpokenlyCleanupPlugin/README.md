# Spokenly Cleanup for TypeWhisper

A local TypeWhisper `PostProcessorPlugin` using the official
[`spokenly/cleanup-1-coreml`](https://huggingface.co/spokenly/cleanup-1-coreml) export.
The distributable plugin bundle includes the compiled Core ML model and tokenizer. There is **no
model download at install time or dictation time**, and transcripts stay on the Mac.

## Fast, no-wait runtime

- Runs the Core ML Neural Engine function (`cpuAndNeuralEngine`) on Apple silicon.
- The plugin asynchronously loads and warms the common 512-token path as soon as it is enabled.
  Core ML's first-use Neural Engine compilation is therefore done before the first cleanup when
  startup has completed. Later short dictations reuse the loaded model.
- The Core ML model card reports about 0.1–0.2 seconds for one- or two-sentence dictations on an
  M4 Pro. That is the upstream benchmark, not a benchmark measured by this plugin.
- The full model asset is about 754 MB. The build script caches the upstream files locally and
  includes them in `SpokenlyCleanupPlugin.bundle`; it is intentionally not checked into git.

## Build

```sh
xcodebuild -project TypeWhisper.xcodeproj -target SpokenlyCleanupPlugin -configuration Release
```

The Xcode target runs `scripts/fetch_spokenly_cleanup_coreml.sh` and bundles the pinned upstream
Core ML model, tokenizer, and model notices. This requires network access during the **build** and
roughly 754 MB of model storage. The first build downloads it; subsequent builds use the local
cache. Install the resulting `.bundle` in **Settings → Integrations → Install from File…**.

## Model behavior

Best support: English. The model card also lists German, Spanish, French, Chinese, Russian, and
Japanese. The prompt follows the model's `text: … / out:` format; decoding is greedy. A failed
inference preserves the original transcript. The Core ML runtime requires macOS 15 or later.

Model license: Apache-2.0. The model is bundled into the plugin, with its LICENSE and NOTICE
files. The artifact is pinned to the upstream revision recorded in the build script.
