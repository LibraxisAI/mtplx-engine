import Foundation
import XCTest
@testable import MTPLXAppCore

/// MLX's macOS 26 wheels need 26.2: a failed import on 26.0 or 26.1 must ask
/// for the macOS update instead of blaming the network.
final class RuntimeImportCheckMessageTests: XCTestCase {
    func testMacOS26BeforePoint2AsksForTheUpdate() {
        for minor in [0, 1] {
            let detail = MTPLXRuntimeBootstrapper.importCheckFailureDetail(
                os: OperatingSystemVersion(majorVersion: 26, minorVersion: minor, patchVersion: 0)
            )
            XCTAssertTrue(detail.contains("macOS 26.2 or later"), detail)
            XCTAssertFalse(detail.contains("network"), detail)
        }
    }

    func testSupportedMacOSKeepsTheNetworkHint() {
        for (major, minor) in [(26, 2), (26, 4), (27, 0), (15, 6)] {
            let detail = MTPLXRuntimeBootstrapper.importCheckFailureDetail(
                os: OperatingSystemVersion(majorVersion: major, minorVersion: minor, patchVersion: 0)
            )
            XCTAssertTrue(detail.contains("network access to PyPI"), detail)
        }
    }
}
