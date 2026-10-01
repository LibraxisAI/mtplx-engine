import XCTest
import MTPLXAppCore
@testable import MTPLXAppHost

final class Release2114OnboardingTests: XCTestCase {
    func testEveryRecommendationHasAnOnboardingCard() {
        for generation in ["m2", "m5"] {
            for ram in [8, 16, 18, 24, 32, 36, 48, 64, 96, 128, 192, 256, 512] {
                let hardware = DetectedHardware(chipName: "Apple \(generation)", appleSiliconGeneration: generation, unifiedMemoryBytes: Int64(ram) * 1_073_741_824)
                let ids = MTPLXModelOption.recommendedCatalogIDs(for: hardware)
                let rows = RecommendedModelRow.rows(for: ids)
                XCTAssertEqual(rows.count, ids.count, "\(generation) \(ram)")
                for (row, id) in zip(rows, ids) {
                    var state = OnboardingFeatureState()
                    state.hardware = hardware
                    state.select(row.choice)
                    XCTAssertEqual(state.resolvedModel?.id, id)
                }
                if generation == "m5" && ram == 8 {
                    XCTAssertEqual(rows.first?.choice, .curatedQwen35FourBit)
                }
            }
        }
    }
}
