import CoreML
import Foundation
import SwiftUI
import Tokenizers
import TypeWhisperPluginSDK

private enum CleanupModelError: LocalizedError {
    case missingBundledModel
    case missingTokenizer
    case invalidOutput

    var errorDescription: String? {
        switch self {
        case .missingBundledModel: "The bundled Spokenly Core ML model is missing. Rebuild the plugin with its model-download build phase."
        case .missingTokenizer: "The bundled Spokenly tokenizer is missing."
        case .invalidOutput: "The Core ML model returned no next token."
        }
    }
}

/// Stateful Core ML decode session for one attention-window function.
/// Access is serialized by the plugin's main-actor post-processing method.
@available(macOS 15.0, *)
private final class SpokenlyCoreMLSession {
    private let model: MLModel
    private let tokenizer: any Tokenizers.Tokenizer
    private let state: MLState
    private let window: Int
    private var cachedTokenCount = 0
    private let endTokenID = 151_643 // Qwen3 <|endoftext|>

    init(model: MLModel, tokenizer: any Tokenizers.Tokenizer, window: Int) {
        self.model = model
        self.tokenizer = tokenizer
        self.window = window
        self.state = model.makeState()
    }

    func clean(_ text: String) throws -> String {
        // Reuse the preallocated Core ML state but logically start a fresh
        // transcript; cacheMask excludes leftover keys until they are replaced.
        cachedTokenCount = 0
        let prompt = SpokenlyCleanupPlugin.prompt(text: text)
        let promptIDs = tokenizer.encode(text: prompt, addSpecialTokens: true)
        guard !promptIDs.isEmpty, promptIDs.count < window else { return text }

        // Core ML's model consumes 16-token blocks. Prefill each block once;
        // generated tokens then use row zero of the same fixed-size function.
        var nextToken = endTokenID
        var offset = 0
        while offset < promptIDs.count {
            let count = min(16, promptIDs.count - offset)
            let block = Array(promptIDs[offset..<(offset + count)])
            nextToken = try predict(block: block, answerRow: count - 1, position: offset)
            offset += count
        }

        let outputLimit = min(256, window - promptIDs.count)
        guard outputLimit > 0 else { return text }
        var generated: [Int] = []
        generated.reserveCapacity(min(outputLimit, 96))

        while generated.count < outputLimit, nextToken != endTokenID {
            generated.append(nextToken)
            guard generated.count < outputLimit else { break }
            // The token just sampled becomes the next one-token input. Only its
            // first cache row is committed; padded rows are never copied into KV.
            nextToken = try predict(
                block: [nextToken],
                answerRow: 0,
                position: promptIDs.count + generated.count - 1
            )
        }

        // A truncated answer is worse than the recognizer's untouched text.
        guard nextToken == endTokenID else { return text }
        let result = tokenizer.decode(tokens: generated, skipSpecialTokens: true)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !result.isEmpty, result.count <= max(text.count * 4, 256) else { return text }
        return result
    }

    private func predict(block: [Int], answerRow: Int, position: Int) throws -> Int {
        let inputIDs = try MLMultiArray(shape: [1, 16], dataType: .int32)
        let positions = try MLMultiArray(shape: [16], dataType: .int32)
        let answer = try MLMultiArray(shape: [1], dataType: .int32)
        let mask = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: window)], dataType: .float16)

        let inputPointer = inputIDs.dataPointer.assumingMemoryBound(to: Int32.self)
        let positionPointer = positions.dataPointer.assumingMemoryBound(to: Int32.self)
        for index in 0..<16 {
            let token = index < block.count ? block[index] : endTokenID
            inputPointer[index] = Int32(token)
            // Positions beyond the live portion are masked by answerRow's
            // causal selection and their returned cache rows are discarded.
            positionPointer[index] = Int32(min(position + index, window + 15))
        }
        answer[0] = NSNumber(value: answerRow)

        let maskPointer = mask.dataPointer.assumingMemoryBound(to: Float16.self)
        for index in 0..<window {
            maskPointer[index] = index < cachedTokenCount ? 0 : -10_000
        }

        let features = try MLDictionaryFeatureProvider(dictionary: [
            "inputIds": MLFeatureValue(multiArray: inputIDs),
            "positions": MLFeatureValue(multiArray: positions),
            "answerRow": MLFeatureValue(multiArray: answer),
            "cacheMask": MLFeatureValue(multiArray: mask),
        ])
        let result = try model.prediction(from: features, using: state)
        guard let nextTokenArray = result.featureValue(for: "nextToken")?.multiArrayValue else {
            throw CleanupModelError.invalidOutput
        }
        guard let newKeys = result.featureValue(for: "newKeys")?.multiArrayValue,
              let newValues = result.featureValue(for: "newValues")?.multiArrayValue else {
            throw CleanupModelError.invalidOutput
        }

        let committedCount = block.count
        appendCache(output: newKeys, stateName: "keyCache", count: committedCount)
        appendCache(output: newValues, stateName: "valueCache", count: committedCount)
        cachedTokenCount = min(window, cachedTokenCount + committedCount)
        return Int(nextTokenArray[0].int32Value)
    }

    /// The model exposes the current block's K/V rows as outputs; it expects
    /// the caller to write those rows into its state buffers. Keep a compact,
    /// ordered sliding window in the first `window` state slots.
    private func appendCache(output: MLMultiArray, stateName: String, count: Int) {
        let retainedCount = min(cachedTokenCount, window - count)
        let dropCount = cachedTokenCount - retainedCount
        let sourceStrides = output.strides.map(\.intValue)

        state.withMultiArray(for: stateName) { cache in
            let destinationStrides = cache.strides.map(\.intValue)
            let source = output.dataPointer.assumingMemoryBound(to: Float16.self)
            let destination = cache.dataPointer.assumingMemoryBound(to: Float16.self)

            if dropCount > 0 {
                for layer in 0..<28 {
                    for head in 0..<8 {
                        for token in 0..<retainedCount {
                            let oldToken = token + dropCount
                            for dimension in 0..<128 {
                                let from = layer * destinationStrides[0]
                                    + head * destinationStrides[1]
                                    + oldToken * destinationStrides[2]
                                    + dimension * destinationStrides[3]
                                let to = layer * destinationStrides[0]
                                    + head * destinationStrides[1]
                                    + token * destinationStrides[2]
                                    + dimension * destinationStrides[3]
                                destination[to] = destination[from]
                            }
                        }
                    }
                }
            }

            for layer in 0..<28 {
                for head in 0..<8 {
                    for token in 0..<count {
                        let targetToken = retainedCount + token
                        for dimension in 0..<128 {
                            let from = layer * sourceStrides[0]
                                + head * sourceStrides[1]
                                + token * sourceStrides[2]
                                + dimension * sourceStrides[3]
                            let to = layer * destinationStrides[0]
                                + head * destinationStrides[1]
                                + targetToken * destinationStrides[2]
                                + dimension * destinationStrides[3]
                            destination[to] = source[from]
                        }
                    }
                }
            }
        }
    }
}

/// Auto-unload duration options matching TypeWhisper preferences.
public enum AutoUnloadTimeoutOption: Int, CaseIterable, Identifiable, Sendable {
    case followApp = -2
    case immediate = -1
    case never = 0
    case after2Minutes = 120
    case after5Minutes = 300
    case after10Minutes = 600
    case after30Minutes = 1800
    case after1Hour = 3600

    public var id: Int { rawValue }

    public var title: String {
        switch self {
        case .followApp: return "Match TypeWhisper setting"
        case .never: return "Never"
        case .immediate: return "Immediate"
        case .after2Minutes: return "After 2 minutes"
        case .after5Minutes: return "After 5 minutes"
        case .after10Minutes: return "After 10 minutes"
        case .after30Minutes: return "After 30 minutes"
        case .after1Hour: return "After 1 hour"
        }
    }
}

@objc(SpokenlyCleanupPlugin)
final class SpokenlyCleanupPlugin: NSObject, ObservableObject, PostProcessorPlugin, @unchecked Sendable {
    static let pluginId = "com.typewhisper.spokenly-cleanup"
    static let pluginName = "Spokenly Cleanup"

    let processorName = "Spokenly Cleanup 1 (Core ML)"
    let priority = 800

    @Published var autoUnloadOption: AutoUnloadTimeoutOption = .after5Minutes {
        didSet {
            host?.setUserDefault(autoUnloadOption.rawValue, forKey: "autoUnloadSeconds")
            Task { @MainActor [weak self] in
                self?.scheduleAutoUnloadIfNeeded()
            }
        }
    }

    @Published private(set) var statusText = "Ready · Model bundled"
    @Published private(set) var isLoading = false
    @Published private(set) var isModelLoaded = false

    private var host: HostServices?
    private var eventSubscriptionId: UUID?

    @MainActor private var sessions: [Int: Any] = [:]
    @MainActor private var tokenizer: (any Tokenizers.Tokenizer)?
    @MainActor private var loadingTask: Task<Void, Never>?
    @MainActor private var autoUnloadTask: Task<Void, Never>?

    required override init() { super.init() }

    func activate(host: HostServices) {
        self.host = host
        if let savedVal = host.userDefault(forKey: "autoUnloadSeconds") as? Int,
           let option = AutoUnloadTimeoutOption(rawValue: savedVal) {
            self.autoUnloadOption = option
        }

        eventSubscriptionId = host.eventBus.subscribe { [weak self] event in
            await self?.handleEvent(event)
        }

        Task { @MainActor [weak self] in
            guard let self else { return }
            if self.effectiveAutoUnloadSeconds == 0 {
                await self.warmUp()
            } else {
                self.statusText = "Ready · Model bundled (prewarms on dictation)"
            }
        }
    }

    func deactivate() {
        if let id = eventSubscriptionId {
            host?.eventBus.unsubscribe(id: id)
            eventSubscriptionId = nil
        }
        Task { @MainActor [weak self] in
            self?.autoUnloadTask?.cancel()
            self?.autoUnloadTask = nil
            self?.loadingTask?.cancel()
            self?.loadingTask = nil
            self?.sessions.removeAll()
            self?.tokenizer = nil
            self?.isLoading = false
            self?.isModelLoaded = false
            self?.statusText = "Model unloaded."
        }
    }

    private func handleEvent(_ event: TypeWhisperEvent) async {
        switch event {
        case .recordingStarted:
            await prewarmForDictation()
        default:
            break
        }
    }

    @MainActor
    private func prewarmForDictation() {
        guard #available(macOS 15.0, *) else { return }
        autoUnloadTask?.cancel()
        autoUnloadTask = nil

        guard !isModelLoaded, loadingTask == nil else { return }

        statusText = "Prewarming model on Neural Engine while you speak…"
        loadingTask = Task { @MainActor [weak self] in
            guard let self else { return }
            self.isLoading = true
            do {
                let tokenizer = try await self.loadTokenizer()
                let session = try await self.loadSession(window: 512, tokenizer: tokenizer)
                _ = try session.clean("Hello there.")
                self.isModelLoaded = true
                self.statusText = "Ready · model bundled, warmed, and running locally"
            } catch {
                self.isModelLoaded = false
                self.statusText = "Could not prepare the bundled model: \(error.localizedDescription)"
            }
            self.isLoading = false
            self.loadingTask = nil
        }
    }

    var effectiveAutoUnloadSeconds: Int {
        switch autoUnloadOption {
        case .followApp:
            let defaults = UserDefaults.standard
            if defaults.object(forKey: "modelAutoUnloadSeconds") != nil {
                return defaults.integer(forKey: "modelAutoUnloadSeconds")
            }
            return 600
        case .immediate:
            return -1
        case .never:
            return 0
        case .after2Minutes:
            return 120
        case .after5Minutes:
            return 300
        case .after10Minutes:
            return 600
        case .after30Minutes:
            return 1800
        case .after1Hour:
            return 3600
        }
    }

    @MainActor
    func scheduleAutoUnloadIfNeeded() {
        autoUnloadTask?.cancel()
        autoUnloadTask = nil

        guard isModelLoaded else { return }

        let seconds = effectiveAutoUnloadSeconds
        guard seconds != 0 else { return }

        if seconds == -1 {
            autoUnloadTask = Task { @MainActor [weak self] in
                try? await Task.sleep(for: .milliseconds(150))
                guard !Task.isCancelled else { return }
                self?.unloadModel()
            }
        } else {
            autoUnloadTask = Task { @MainActor [weak self] in
                try? await Task.sleep(for: .seconds(seconds))
                guard !Task.isCancelled else { return }
                self?.unloadModel()
            }
        }
    }

    @objc func triggerAutoUnload() {
        Task { @MainActor [weak self] in
            self?.unloadModel()
        }
    }

    @MainActor
    func unloadModel() {
        autoUnloadTask?.cancel()
        autoUnloadTask = nil
        loadingTask?.cancel()
        loadingTask = nil
        sessions.removeAll()
        tokenizer = nil
        isLoading = false
        isModelLoaded = false
        statusText = "Model unloaded from memory (will prewarm on dictation)"
    }

    @MainActor
    var settingsView: AnyView? {
        AnyView(SpokenlyCleanupSettingsView(plugin: self))
    }

    @MainActor
    func process(text: String, context: PostProcessingContext) async throws -> String {
        autoUnloadTask?.cancel()
        autoUnloadTask = nil

        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            scheduleAutoUnloadIfNeeded()
            return text
        }
        if let language = context.language, !Self.supportsLanguage(language) {
            scheduleAutoUnloadIfNeeded()
            return text
        }
        guard #available(macOS 15.0, *) else { return text }
        if let loadingTask { await loadingTask.value }

        defer {
            scheduleAutoUnloadIfNeeded()
        }

        do {
            let tokenizer = try await loadTokenizer()
            let estimatedPromptTokens = tokenizer.encode(text: Self.prompt(text: text), addSpecialTokens: true).count
            let window = Self.window(for: estimatedPromptTokens)
            let session = try await loadSession(window: window, tokenizer: tokenizer)
            let cleaned = try session.clean(text)
            isModelLoaded = true
            statusText = "Ready · Core ML Neural Engine · local only"
            return cleaned
        } catch {
            statusText = "Cleanup unavailable; original text kept. \(error.localizedDescription)"
            return text
        }
    }

    @MainActor
    func warmUp() async {
        guard #available(macOS 15.0, *) else {
            statusText = "Spokenly Cleanup requires macOS 15 or later."
            return
        }
        if let loadingTask {
            await loadingTask.value
            return
        }
        isLoading = true
        statusText = "Preparing the bundled model on the Neural Engine…"
        let task = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                let tokenizer = try await self.loadTokenizer()
                let session = try await self.loadSession(window: 512, tokenizer: tokenizer)
                // The first actual prediction triggers Core ML's one-time ANE
                // compilation. Doing it here keeps that work off the first dictation.
                _ = try session.clean("Hello there.")
                self.isModelLoaded = true
                self.statusText = "Ready · model bundled, warmed, and running locally"
                self.scheduleAutoUnloadIfNeeded()
            } catch {
                self.isModelLoaded = false
                self.statusText = "Could not prepare the bundled model: \(error.localizedDescription)"
            }
            self.isLoading = false
            self.loadingTask = nil
        }
        loadingTask = task
        await task.value
    }

    @MainActor
    private func loadTokenizer() async throws -> any Tokenizers.Tokenizer {
        if let tokenizer { return tokenizer }
        let resourceFolder = try Self.resourceFolder()
        guard FileManager.default.fileExists(atPath: resourceFolder.appendingPathComponent("tokenizer.json").path) else {
            throw CleanupModelError.missingTokenizer
        }
        let configURL = resourceFolder.appendingPathComponent("tokenizer_config.json")
        if !FileManager.default.fileExists(atPath: configURL.path) {
            let defaultConfig = """
            {"add_prefix_space":false,"backend":"tokenizers","bos_token":null,"clean_up_tokenization_spaces":false,"eos_token":"<|endoftext|>","errors":"replace","is_local":true,"local_files_only":false,"model_max_length":131072,"pad_token":"<|endoftext|>","split_special_tokens":false,"tokenizer_class":"Qwen2Tokenizer","unk_token":null}
            """
            try? defaultConfig.data(using: .utf8)?.write(to: configURL)
        }
        let loaded = try await AutoTokenizer.from(modelFolder: resourceFolder)
        tokenizer = loaded
        return loaded
    }

    @MainActor
    @available(macOS 15.0, *)
    private func loadSession(window: Int, tokenizer: any Tokenizers.Tokenizer) async throws -> SpokenlyCoreMLSession {
        if let existing = sessions[window] as? SpokenlyCoreMLSession { return existing }
        let modelURL = try Self.resourceFolder().appendingPathComponent("CleanupNeuralEngine.mlmodelc", isDirectory: true)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw CleanupModelError.missingBundledModel
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        configuration.functionName = "window\(window)"
        let model = try MLModel(contentsOf: modelURL, configuration: configuration)
        let session = SpokenlyCoreMLSession(model: model, tokenizer: tokenizer, window: window)
        sessions[window] = session
        return session
    }

    private static func resourceFolder() throws -> URL {
        guard let resources = Bundle(for: SpokenlyCleanupPlugin.self).resourceURL else {
            throw CleanupModelError.missingBundledModel
        }
        return resources.appendingPathComponent("SpokenlyCleanupResources", isDirectory: true)
    }

    static func prompt(text: String) -> String {
        "text: \(text)\nout:"
    }

    static func supportsLanguage(_ language: String) -> Bool {
        let code = language.lowercased().split(whereSeparator: { $0 == "-" || $0 == "_" }).first.map(String.init) ?? ""
        return ["en", "de", "es", "fr", "zh", "ru", "ja"].contains(code)
    }

    static func window(for promptTokens: Int) -> Int {
        if promptTokens + 256 < 512 { return 512 }
        if promptTokens + 256 < 2_048 { return 2_048 }
        return 4_096
    }
}

private struct SpokenlyCleanupSettingsView: View {
    @ObservedObject var plugin: SpokenlyCleanupPlugin

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Spokenly Cleanup 1")
                .font(.headline)
            Text("Fast, local cleanup of dictated text: fillers and repeats removed, spoken corrections applied, and punctuation, capitalization, and numbers formatted. The model is included in this plugin; transcripts are never sent to a server.")
                .fixedSize(horizontal: false, vertical: true)
            Text("The bundled Core ML model runs on Apple’s Neural Engine. It automatically prewarms in the background when you start speaking and unloads from memory when idle.")
                .font(.callout)
                .foregroundStyle(.secondary)

            Picker("Auto-unload model", selection: $plugin.autoUnloadOption) {
                Text(AutoUnloadTimeoutOption.followApp.title).tag(AutoUnloadTimeoutOption.followApp)
                Divider()
                Text(AutoUnloadTimeoutOption.never.title).tag(AutoUnloadTimeoutOption.never)
                Text(AutoUnloadTimeoutOption.immediate.title).tag(AutoUnloadTimeoutOption.immediate)
                Text(AutoUnloadTimeoutOption.after2Minutes.title).tag(AutoUnloadTimeoutOption.after2Minutes)
                Text(AutoUnloadTimeoutOption.after5Minutes.title).tag(AutoUnloadTimeoutOption.after5Minutes)
                Text(AutoUnloadTimeoutOption.after10Minutes.title).tag(AutoUnloadTimeoutOption.after10Minutes)
                Text(AutoUnloadTimeoutOption.after30Minutes.title).tag(AutoUnloadTimeoutOption.after30Minutes)
                Text(AutoUnloadTimeoutOption.after1Hour.title).tag(AutoUnloadTimeoutOption.after1Hour)
            }
            .pickerStyle(.menu)

            HStack(spacing: 10) {
                Button(plugin.isLoading ? "Preparing…" : (plugin.isModelLoaded ? "Model loaded" : "Warm up model")) {
                    Task { await plugin.warmUp() }
                }
                .disabled(plugin.isLoading || plugin.isModelLoaded)

                if plugin.isModelLoaded {
                    Button("Unload model") {
                        plugin.unloadModel()
                    }
                    .disabled(plugin.isLoading)
                }

                ProgressView().controlSize(.small).opacity(plugin.isLoading ? 1 : 0)
            }
            Text(plugin.statusText)
                .font(.caption)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
            Link("Model card and Apache 2.0 license", destination: URL(string: "https://huggingface.co/spokenly/cleanup-1")!)
                .font(.caption)
        }
        .padding(20)
        .frame(maxWidth: 520, alignment: .leading)
    }
}
