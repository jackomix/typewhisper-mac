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
}
