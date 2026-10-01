import XCTest
@testable import MTPLXAppCore

final class ModelFeasibilityTests: XCTestCase {
    private let evaluator = ModelFeasibility()

    private var speed: MTPLXModelOption {
        MTPLXModelOption.officialCatalog.first { $0.id == "optimized-speed" }!
    }
    private var quality: MTPLXModelOption {
        MTPLXModelOption.officialCatalog.first { $0.id == "optimized-quality" }!
    }
    private var fp16: MTPLXModelOption {
        MTPLXModelOption.officialCatalog.first { $0.id == "optimized-speed-fp16" }!
    }

    // Ample disk free for every case below; we exercise the disk gate
    // separately at the bottom of the file.
    private let ampleDiskGiB: Double = 500

    // MARK: - Speed (~17 GiB peak)

    func testSpeedOnLegacy8GBIsInsufficient() {
        let v = evaluator.evaluate(model: speed, chipTier: .legacyApple, ramGiB: 8, diskFreeGiB: ampleDiskGiB)
        XCTAssertEqual(v, .insufficientMemory(needsGiB: 17.0 * 1.5))
    }

    func testSpeedOnLegacy16GBIsInsufficient() {
        // 16 GiB RAM is below the 17 GiB raw peak — insufficient.
        let v = evaluator.evaluate(model: speed, chipTier: .legacyApple, ramGiB: 16, diskFreeGiB: ampleDiskGiB)
        if case .insufficientMemory = v { return }
        XCTFail("Expected insufficientMemory, got \(v)")
    }

    func testSpeedOnModern24GBIsTightFit() {
        // 24 GiB > 17 (peak) but < 25.5 (safe floor 17 * 1.5).
        let v = evaluator.evaluate(model: speed, chipTier: .modernApple, ramGiB: 24, diskFreeGiB: ampleDiskGiB)
        XCTAssertEqual(v, .tightFit)
    }

    func testSpeedOnModern36GBIsRecommended() {
        let v = evaluator.evaluate(model: speed, chipTier: .modernApple, ramGiB: 36, diskFreeGiB: ampleDiskGiB)
        XCTAssertEqual(v, .recommended)
    }

    // MARK: - Quality (~28 GiB peak)

    func testQualityOnLegacy16GBIsInsufficient() {
        let v = evaluator.evaluate(model: quality, chipTier: .legacyApple, ramGiB: 16, diskFreeGiB: ampleDiskGiB)
        if case .insufficientMemory = v { return }
        XCTFail("Expected insufficientMemory, got \(v)")
    }

    func testQualityOnModern36GBIsTightFit() {
        // 36 > 28 (peak) but < 42 (safe floor 28 * 1.5).
        let v = evaluator.evaluate(model: quality, chipTier: .modernApple, ramGiB: 36, diskFreeGiB: ampleDiskGiB)
        XCTAssertEqual(v, .tightFit)
    }

    func testQualityOnModern48GBIsRecommended() {
        let v = evaluator.evaluate(model: quality, chipTier: .modernApple, ramGiB: 48, diskFreeGiB: ampleDiskGiB)
        XCTAssertEqual(v, .recommended)
    }

    func testQualityOnModern128GBIsRecommended() {
        let v = evaluator.evaluate(model: quality, chipTier: .modernApple, ramGiB: 128, diskFreeGiB: ampleDiskGiB)
        XCTAssertEqual(v, .recommended)
    }

    // MARK: - FP16 (~17.5 GiB peak)

    func testFP16OnLegacy16GBIsInsufficient() {
        let v = evaluator.evaluate(model: fp16, chipTier: .legacyApple, ramGiB: 16, diskFreeGiB: ampleDiskGiB)
        if case .insufficientMemory = v { return }
        XCTFail("Expected insufficientMemory, got \(v)")
    }

    func testFP16OnLegacy24GBIsTightFit() {
        // 24 > 17.5 (peak) but < 26.25 (safe floor 17.5 * 1.5).
        let v = evaluator.evaluate(model: fp16, chipTier: .legacyApple, ramGiB: 24, diskFreeGiB: ampleDiskGiB)
        XCTAssertEqual(v, .tightFit)
    }

    func testFP16OnLegacy32GBIsRecommended() {
        let v = evaluator.evaluate(model: fp16, chipTier: .legacyApple, ramGiB: 32, diskFreeGiB: ampleDiskGiB)
        XCTAssertEqual(v, .recommended)
    }

    // MARK: - Intel always insufficient

    func testIntelMacAlwaysInsufficientForCuratedModels() {
        for model in [speed, quality, fp16] {
            let v = evaluator.evaluate(model: model, chipTier: .intel, ramGiB: 64, diskFreeGiB: ampleDiskGiB)
            if case .insufficientMemory = v { continue }
            XCTFail("\(model.id) on Intel should be insufficientMemory, got \(v)")
        }
    }

    // MARK: - Disk gate beats memory gate

    func testInsufficientDiskBlocksEvenWhenMemoryFits() {
        // 128 GiB RAM is plenty for Quality, but 20 GiB free disk is not
        // enough for its 28 GiB download plus the 5 GiB pull headroom.
        let v = evaluator.evaluate(model: quality, chipTier: .modernApple, ramGiB: 128, diskFreeGiB: 20)
        if case .insufficientDisk(let needs) = v {
            XCTAssertEqual(needs, Double(quality.sizeBytes) / 1_073_741_824 + 5, accuracy: 1e-9)
            return
        }
        XCTFail("Expected insufficientDisk, got \(v)")
    }

    func testDiskGateIsTheMTPLXPullRule() throws {
        let flashQuality = try XCTUnwrap(MTPLXModelOption.officialCatalog.first { $0.id == "flash-next-optimized-quality" })
        // 163.3 GiB for the 170 GB pack; the old 2.5x rule asked 395.7.
        let needs = Double(flashQuality.sizeBytes) / 1_073_741_824 + 5
        XCTAssertEqual(evaluator.evaluate(model: flashQuality, chipTier: .modernApple, ramGiB: 256, diskFreeGiB: needs), .recommended)
        XCTAssertEqual(
            evaluator.evaluate(model: flashQuality, chipTier: .modernApple, ramGiB: 256, diskFreeGiB: needs - 0.01),
            .insufficientDisk(needsGiB: needs)
        )
        // A paused download needs only its remaining bytes.
        let half = flashQuality.sizeBytes / 2
        let resumeFree = Double(flashQuality.sizeBytes - half) / 1_073_741_824 + 5
        XCTAssertEqual(
            evaluator.evaluate(model: flashQuality, chipTier: .modernApple, ramGiB: 256, diskFreeGiB: resumeFree, downloadedBytes: half),
            .recommended
        )
        // Unknown size (a Forge probe stub): no disk gate, as before.
        XCTAssertEqual(ModelFeasibility.requiredFreeDiskGiB(sizeBytes: 0), 0)
    }
}
