import Foundation
import XCTest
@testable import MTPLXAppCore

final class ClientControlOwnershipTests: XCTestCase {
    func testAppOwnershipDefaultAndOptOutSurviveSettingsRoundTrip() throws {
        let original = try JSONDecoder().decode(MTPLXAppConfiguration.self, from: Data("{}".utf8))
        XCTAssertTrue(original.controlClientSettings)
        var optedOut = original
        optedOut.controlClientSettings = false
        let restored = try JSONDecoder().decode(MTPLXAppConfiguration.self, from: JSONEncoder().encode(optedOut))
        XCTAssertFalse(restored.controlClientSettings)
        let home = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: home, withIntermediateDirectories: true)
        let executable = home.appendingPathComponent("mtplx")
        try Data("#!/bin/sh\nexit 0\n".utf8).write(to: executable)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: executable.path)
        let builder = MTPLXCommandBuilder(environment: [
            "HOME": home.path, "PATH": home.path, "MTPLX_APP_DISABLE_STANDARD_PATHS": "1"
        ])
        XCTAssertEqual(try builder.buildServeCommand(configuration: original).environment["MTPLX_MANAGED_CLIENT_CONTROLS"], "app")
        XCTAssertEqual(try builder.buildServeCommand(configuration: restored).environment["MTPLX_MANAGED_CLIENT_CONTROLS"], "client")
    }

    @MainActor
    func testOwnershipIsIncludedInTheLiveWirePatch() throws {
        for policy in ["app", "client"] {
            let patch = MTPLXBackendStore.liveSettingsUpdatePatch(from: MutableSettings(managedClientControls: policy))
            let data = try JSONEncoder().encode(patch)
            let wire = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
            XCTAssertEqual(wire["managed_client_controls"] as? String, policy)
            XCTAssertEqual(try JSONDecoder().decode(MutableSettings.self, from: data).managedClientControls, policy)
        }
    }
    func testSettingsMirrorDoesNotReplaceACustomRequestBridge() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let extensions = root.appendingPathComponent("extensions")
        try FileManager.default.createDirectory(at: extensions, withIntermediateDirectories: true)
        let custom = extensions.appendingPathComponent(PiIntegration.requestPolicyExtensionName)
        let text = "export default function customBridge(pi) {}"
        try text.write(to: custom, atomically: true, encoding: .utf8)
        _ = try PiIntegration(configURL: root.appendingPathComponent("models.json"))
            .sync(configuration: MTPLXAppConfiguration())
        XCTAssertEqual(try String(contentsOf: custom, encoding: .utf8), text)
        let mirror = try String(contentsOf: extensions.appendingPathComponent("mtplx-settings-sync.ts"), encoding: .utf8)
        XCTAssertTrue(mirror.contains("pi.setThinkingLevel(effort)"))
    }

}
