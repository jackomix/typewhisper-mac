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
}
