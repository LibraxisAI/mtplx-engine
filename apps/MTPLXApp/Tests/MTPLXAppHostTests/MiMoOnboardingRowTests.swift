import XCTest
import MTPLXAppCore
@testable import MTPLXAppHost

final class MiMoOnboardingRowTests: XCTestCase {
    func testMiMoHasItsOwnOnboardingCardRightAheadOfTheNineB() throws {
        let row = try XCTUnwrap(RecommendedModelRow.row(for: "mimo-v26-qwen-9b-optimized-speed"))
        XCTAssertEqual(row.choice, .curatedMiMoQwen9BOptimizedSpeed)
        XCTAssertEqual(row.modelID, "mimo-v26-qwen-9b-optimized-speed")
        XCTAssertEqual(row.logo, .qwen)
        XCTAssertEqual(row.title, "MiMo V2.6 Qwen 9B Optimized Speed")
        XCTAssertEqual(row.detailLocalizationKey, "6-bit quantization. Xiaomi's agentic coding distill of Qwen 3.5 9B.")

        let catalog = try XCTUnwrap(MTPLXModelOption.officialCatalog.first { $0.id == row.modelID })
        XCTAssertEqual(row.title, catalog.displayName)
        XCTAssertEqual(row.detailLocalizationKey, catalog.detail)

        // A 16 GB modern Mac: Bonsai first, then MiMo, then the Qwen 3.5 9B.
        let hardware = DetectedHardware(
            chipName: "Apple M4",
            appleSiliconGeneration: "m4",
            unifiedMemoryBytes: 16 * 1_073_741_824
        )
        let choices = RecommendedModelRow.rows(for: MTPLXModelOption.recommendedCatalogIDs(for: hardware)).map(\.choice)
        XCTAssertEqual(Array(choices.prefix(3)), [
            .curatedBonsaiOptimizedSpeed,
            .curatedMiMoQwen9BOptimizedSpeed,
            .curatedQwen35NineBSpeed,
        ])
    }
}
