import XCTest
import MTPLXAppCore

@testable import MTPLXAppHost

final class ModelPickerOverlayTests: XCTestCase {
    func testMissingPersistedCustomModelCanBeRemoved() throws {
        let option = try XCTUnwrap(MTPLXModelOption.customHuggingFaceModel(repoID: "Foo/Bar"))
        let row = ModelPickerPreparedOption(
            option: option,
            currentModel: "Other/Model",
            customModels: [option]
        )

        XCTAssertFalse(row.isInstalled)
        XCTAssertTrue(row.canRemoveFromPicker)
    }

    func testInstalledPersistedCustomModelCannotBeRemoved() throws {
        let directory = temporaryDirectory().appendingPathComponent("installed-model", isDirectory: true)
        try makeCompleteModelFolder(at: directory)
        defer { try? FileManager.default.removeItem(at: directory) }

        let option = MTPLXModelOption(
            id: "custom-installed",
            displayName: "Installed Custom",
            shortName: "Installed Custom",
            detail: "Test model",
            hfModelID: "Foo/Installed",
            localCandidates: [directory.path]
        )
        let row = ModelPickerPreparedOption(
            option: option,
            currentModel: "Other/Model",
            customModels: [option]
        )

        XCTAssertTrue(row.isInstalled)
        XCTAssertFalse(row.canRemoveFromPicker)
    }

    func testOfficialModelCannotBeRemoved() throws {
        let option = try XCTUnwrap(MTPLXModelOption.officialCatalog.first)
        let row = ModelPickerPreparedOption(
            option: option,
            currentModel: "Other/Model",
            customModels: [option]
        )

        XCTAssertFalse(row.canRemoveFromPicker)
    }

    func testCurrentSynthesizedModelCannotBeRemovedAndRemainsRepresentable() throws {
        let option = try XCTUnwrap(MTPLXModelOption.customHuggingFaceModel(repoID: "Foo/Current"))
        let row = ModelPickerPreparedOption(
            option: option,
            currentModel: option.hfModelID,
            customModels: []
        )

        XCTAssertTrue(row.selected)
        XCTAssertFalse(row.canRemoveFromPicker)
        XCTAssertEqual(row.resolvedReference, option.hfModelID)
    }

    // Two builds of one source used to share a row named after that source,
    // and the row launched whichever build was found first. Each build now
    // has its own row, which stores its own folder when selected.
    func testBuildsOfOneSourceEachStoreTheirOwnFolder() throws {
        let root = temporaryDirectory()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let tag = String(UUID().uuidString.prefix(8))
        let source = "\(root.path)/_src-Qwen3.5-9B-V26-Distill-graft"
        let runtime = """
        {"forge_provenance": {"source_repo": "\(source)", "source_format": "bf16_native",
         "forge_recipe": {"body_bits": 4, "body_group_size": 64, "body_mode": "affine", "mtp_policy": "keep_bf16"},
         "forge_inputs": {}, "forged_at": "2026-09-22T15:03:46-07:00", "mtplx_version": "2.12.0",
         "forged_locally": true}}
        """
        let fourBit = root.appendingPathComponent("MiMo-9B-namecheck-\(tag)", isDirectory: true)
        let sixBit = root.appendingPathComponent("MiMo-9B-6bit-\(tag)", isDirectory: true)
        for folder in [fourBit, sixBit] {
            try makeCompleteModelFolder(at: folder)
            try runtime.write(to: folder.appendingPathComponent("mtplx_runtime.json"), atomically: true, encoding: .utf8)
        }
        let library = ModelLibrary(primaryDirectory: root.path)
        XCTAssertEqual(
            MTPLXRuntimeMetadata.read(
                at: fourBit.appendingPathComponent("mtplx_runtime.json").path
            )?.forgeProvenance?.sourceRepo,
            source,
            "the fixture must parse as Forge provenance"
        )

        let rows = MTPLXModelOption.pickerCatalog(
            customModels: [],
            currentModel: sixBit.path,
            modelLibrary: library
        ).map { ModelPickerPreparedOption(option: $0, currentModel: sixBit.path, customModels: []) }

        let fourBitRows = rows.filter { $0.option.localCandidates.contains(fourBit.path) }
        XCTAssertEqual(fourBitRows.map(\.displayName), [fourBit.lastPathComponent])
        XCTAssertEqual(fourBitRows.map(\.resolvedReference), [fourBit.path])
        XCTAssertEqual(fourBitRows.map(\.selected), [false])
        let sixBitRows = rows.filter { $0.option.localCandidates.contains(sixBit.path) }
        XCTAssertEqual(sixBitRows.map(\.displayName), [sixBit.lastPathComponent])
        XCTAssertEqual(sixBitRows.map(\.resolvedReference), [sixBit.path])
        XCTAssertEqual(sixBitRows.map(\.selected), [true])
        XCTAssertEqual(rows.filter(\.selected).count, 1)
    }

    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-model-picker-tests-\(UUID().uuidString)", isDirectory: true)
    }

    private func makeCompleteModelFolder(at folder: URL) throws {
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try "{}".write(to: folder.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try "{}".write(to: folder.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)
        try Data([0]).write(to: folder.appendingPathComponent("model.safetensors"))
    }
}