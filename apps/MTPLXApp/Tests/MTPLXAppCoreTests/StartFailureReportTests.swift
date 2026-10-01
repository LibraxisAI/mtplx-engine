import XCTest
@testable import MTPLXAppCore

/// Issue #504: a start that fails before `/health` left nothing on disk, and
/// the banner cut the reason off. These pin the report `mtplx doctor` reads.
/// Pure file I/O in a scratch folder: nothing here launches a process, so the
/// supervisor's own suite (which touches fan state) is not involved.
final class StartFailureReportTests: XCTestCase {
    private var scratch: URL!

    override func setUpWithError() throws {
        scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-start-failure-\(UUID().uuidString)", isDirectory: true)
    }

    override func tearDownWithError() throws {
        // A scratch folder under the system temp directory, created by this
        // test alone.
        try? FileManager.default.removeItem(at: scratch)
    }

    private func entry(_ message: String, _ stream: LogEntry.Stream = .stderr) -> LogEntry {
        LogEntry(date: Date(timeIntervalSince1970: 1_790_000_000), stream: stream, message: message)
    }

    func testDefaultLocationIsUnderTheUsersMTPLXLogs() {
        let home = URL(fileURLWithPath: "/Users/someone", isDirectory: true)
        XCTAssertEqual(
            StartFailureReport.defaultURL(home: home).path,
            "/Users/someone/.mtplx/logs/last-failed-start.log"
        )
    }

    func testTheReportCarriesTheReasonTheVersionsAndTheWholeOutput() throws {
        let url = scratch.appendingPathComponent("logs/last-failed-start.log")
        let entries = [
            entry("launched python -m mtplx.server.openai --api-key ******", .system),
            entry("[5/6] Loading model weights", .stdout),
            entry("libc++abi: terminating due to uncaught exception of type std::runtime_error"),
            entry("[metal::Device] Unable to build metal library from source"),
        ]

        XCTAssertTrue(
            StartFailureReport.write(
                detail: "daemon exited before /health became ready",
                entries: entries,
                to: url,
                appVersion: "2.11.4 (2011060)"
            )
        )

        let text = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(text.hasPrefix("MTPLX failed start\n"))
        XCTAssertTrue(text.contains("reason: daemon exited before /health became ready"))
        XCTAssertTrue(text.contains("app: 2.11.4 (2011060)"))
        XCTAssertTrue(text.contains("macos: "))
        // The line the banner cut off in the report.
        XCTAssertTrue(text.contains("[stderr] [metal::Device] Unable to build metal library from source"))
        XCTAssertTrue(text.contains("[system] launched python -m mtplx.server.openai --api-key ******"))
        XCTAssertTrue(text.contains("last 4 of 4 lines"))
    }

    func testOnlyTheLastLinesAreKept() {
        let entries = (0..<(StartFailureReport.maxLines + 50)).map { entry("line \($0)") }
        let text = StartFailureReport.render(detail: "x", entries: entries)

        XCTAssertFalse(text.contains("] line 49\n"))
        XCTAssertTrue(text.contains("] line 50\n"))
        XCTAssertTrue(text.contains("] line \(StartFailureReport.maxLines + 49)\n"))
        XCTAssertTrue(text.contains("last \(StartFailureReport.maxLines) of \(StartFailureReport.maxLines + 50) lines"))
    }

    func testANewFailureReplacesTheOldReport() throws {
        let url = scratch.appendingPathComponent("last-failed-start.log")
        StartFailureReport.write(detail: "first", entries: [entry("old")], to: url)
        StartFailureReport.write(detail: "second", entries: [entry("new")], to: url)

        let text = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(text.contains("reason: second"))
        XCTAssertFalse(text.contains("old"))
    }

    func testAnUnwritableLocationReportsFalseAndDoesNotThrow() {
        // A path whose parent is a regular file cannot be created.
        let blocker = scratch.appendingPathComponent("blocker")
        try? FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: blocker.path, contents: Data())

        XCTAssertFalse(
            StartFailureReport.write(
                detail: "x",
                entries: [entry("y")],
                to: blocker.appendingPathComponent("logs/last-failed-start.log")
            )
        )
    }
}
