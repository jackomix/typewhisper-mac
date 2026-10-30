import XCTest
@testable import SpokenlyCleanupPlugin

final class SpokenlyCleanupPluginTests: XCTestCase {
    func testPromptMatchesSpokenlyFormat() {
        XCTAssertEqual(SpokenlyCleanupPlugin.prompt(text: "um, meet me friday"), "text: um, meet me friday\nout:")
    }

    func testSupportedLanguageCodes() {
        for language in ["en", "en-US", "de-DE", "es", "fr", "zh-CN", "ru", "ja-JP"] {
            XCTAssertTrue(SpokenlyCleanupPlugin.supportsLanguage(language), language)
        }
        XCTAssertFalse(SpokenlyCleanupPlugin.supportsLanguage("pt-BR"))
        XCTAssertFalse(SpokenlyCleanupPlugin.supportsLanguage(""))
    }

    func testShortDictationsUseSmallestNeuralEngineFunction() {
        XCTAssertEqual(SpokenlyCleanupPlugin.window(for: 30), 512)
        XCTAssertEqual(SpokenlyCleanupPlugin.window(for: 256), 2_048)
        XCTAssertEqual(SpokenlyCleanupPlugin.window(for: 1_600), 2_048)
        XCTAssertEqual(SpokenlyCleanupPlugin.window(for: 2_000), 4_096)
    }

    func testAutoUnloadTimeoutOptions() {
        XCTAssertEqual(AutoUnloadTimeoutOption.followApp.rawValue, -2)
        XCTAssertEqual(AutoUnloadTimeoutOption.immediate.rawValue, -1)
        XCTAssertEqual(AutoUnloadTimeoutOption.never.rawValue, 0)
        XCTAssertEqual(AutoUnloadTimeoutOption.after2Minutes.rawValue, 120)
        XCTAssertEqual(AutoUnloadTimeoutOption.after5Minutes.rawValue, 300)
        XCTAssertEqual(AutoUnloadTimeoutOption.after10Minutes.rawValue, 600)
        XCTAssertEqual(AutoUnloadTimeoutOption.after30Minutes.rawValue, 1800)
        XCTAssertEqual(AutoUnloadTimeoutOption.after1Hour.rawValue, 3600)
    }

    @MainActor
    func testPluginUnloadModelClearsLoadedState() {
        let plugin = SpokenlyCleanupPlugin()
        plugin.unloadModel()
        XCTAssertFalse(plugin.isModelLoaded)
        XCTAssertFalse(plugin.isLoading)
        XCTAssertTrue(plugin.statusText.contains("unloaded"))
    }

    func testSplitIntoCleanableChunksKeepsShortTextIntact() {
        let text = "Hello there, meet me on Friday."
        let chunks = SpokenlyCleanupPlugin.splitIntoCleanableChunks(text: text)
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks.first?.text, text)
    }

    func testSplitIntoCleanableChunksSegmentsLongParagraph() {
        let sentence1 = "This is the first long sentence with plenty of detail to test the segmentation logic."
        let sentence2 = "This is the second long sentence that carries another thought and should be separated."
        let sentence3 = "And this is the third sentence completing the paragraph."
        let combined = "\(sentence1) \(sentence2) \(sentence3)"

        let chunks = SpokenlyCleanupPlugin.splitIntoCleanableChunks(text: combined)
        XCTAssertGreaterThan(chunks.count, 1)
        let joined = chunks.map { $0.text + $0.trailingSeparator }.joined().trimmingCharacters(in: .whitespaces)
        XCTAssertEqual(joined, combined)
    }

    func testSplitIntoCleanableChunksSegmentsLongRunOnSentenceWithoutPeriods() {
        let runOn = "I was talking with the team earlier this morning about the roadmap and we agreed that we need to finalize the mockups before the end of this week, because the client is expecting an update on Monday, so please make sure everyone is ready"
        let chunks = SpokenlyCleanupPlugin.splitIntoCleanableChunks(text: runOn)
        XCTAssertGreaterThan(chunks.count, 1)
        for chunk in chunks {
            XCTAssertLessThanOrEqual(chunk.text.count, 220)
        }
        let joined = chunks.map { $0.text + $0.trailingSeparator }.joined().trimmingCharacters(in: .whitespaces)
        XCTAssertEqual(joined, runOn)
    }
}
