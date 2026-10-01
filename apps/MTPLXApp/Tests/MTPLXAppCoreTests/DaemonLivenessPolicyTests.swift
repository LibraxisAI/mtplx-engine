import XCTest
@testable import MTPLXAppCore

/// Issue #487 truth table: a daemon whose process is alive and whose port
/// still accepts a TCP connection is never reaped on probe timeouts alone.
final class DaemonLivenessPolicyTests: XCTestCase {
    private let alive = DaemonLivenessEvidence(processAlive: true, portAccepting: true)
    private let gone = DaemonLivenessEvidence(processAlive: false, portAccepting: true)
    private let portClosed = DaemonLivenessEvidence(processAlive: true, portAccepting: false)
    private let unknownPidPortOpen = DaemonLivenessEvidence(processAlive: nil, portAccepting: true)
    private let unknownPidPortClosed = DaemonLivenessEvidence(processAlive: nil, portAccepting: false)

    func testASingleMissNeverReapsEvenWhenTheProcessLooksGone() {
        var tracker = DaemonLivenessTracker()
        XCTAssertEqual(tracker.recordMiss(evidence: gone, now: 100), .waiting(misses: 1))
    }

    func testAliveProcessAndOpenPortIsBusyNotDead() {
        // The #487 shape: a 25 s prefix commit, /health silent, two probes
        // missed 3 s apart. Old policy: reap on the second miss.
        var tracker = DaemonLivenessTracker()
        XCTAssertEqual(tracker.recordMiss(evidence: alive, now: 100), .waiting(misses: 1))
        XCTAssertEqual(tracker.recordMiss(evidence: alive, now: 113), .busy(unresponsiveFor: 13))
        XCTAssertEqual(tracker.recordMiss(evidence: alive, now: 126), .busy(unresponsiveFor: 26))
    }

    func testBusyDaemonIsReapedOnlyAfterTheGraceExpires() {
        var tracker = DaemonLivenessTracker(busyGraceSeconds: 90)
        _ = tracker.recordMiss(evidence: alive, now: 100)
        XCTAssertEqual(tracker.recordMiss(evidence: alive, now: 189.9), .busy(unresponsiveFor: 89.9))
        XCTAssertEqual(
            tracker.recordMiss(evidence: alive, now: 190),
            .reap(.unresponsiveGraceExpired(90))
        )
    }

    func testAnAnswerResetsTheSilentStretch() {
        var tracker = DaemonLivenessTracker()
        _ = tracker.recordMiss(evidence: alive, now: 100)
        _ = tracker.recordMiss(evidence: alive, now: 113)
        tracker.recordAnswer()
        XCTAssertEqual(tracker.consecutiveMisses, 0)
        XCTAssertNil(tracker.silentSince)
        // The next silence starts its own 90 s clock.
        XCTAssertEqual(tracker.recordMiss(evidence: alive, now: 500), .waiting(misses: 1))
        XCTAssertEqual(tracker.recordMiss(evidence: alive, now: 513), .busy(unresponsiveFor: 13))
    }

    func testGoneProcessReapsOnTheSecondMissRegardlessOfThePort() {
        var tracker = DaemonLivenessTracker()
        _ = tracker.recordMiss(evidence: gone, now: 100)
        XCTAssertEqual(tracker.recordMiss(evidence: gone, now: 113), .reap(.processGone))
    }

    func testClosedPortReapsOnTheSecondMiss() {
        var tracker = DaemonLivenessTracker()
        _ = tracker.recordMiss(evidence: portClosed, now: 100)
        XCTAssertEqual(tracker.recordMiss(evidence: portClosed, now: 113), .reap(.portClosed))

        var adopted = DaemonLivenessTracker()
        _ = adopted.recordMiss(evidence: unknownPidPortClosed, now: 100)
        XCTAssertEqual(adopted.recordMiss(evidence: unknownPidPortClosed, now: 113), .reap(.portClosed))
    }

    func testUnknownPidWithAnOpenPortIsBusy() {
        // An adopted daemon has no owned Process; the open port is the
        // liveness signal.
        var tracker = DaemonLivenessTracker()
        _ = tracker.recordMiss(evidence: unknownPidPortOpen, now: 100)
        XCTAssertEqual(tracker.recordMiss(evidence: unknownPidPortOpen, now: 113), .busy(unresponsiveFor: 13))
    }

    func testEvidenceLivenessSummary() {
        XCTAssertTrue(alive.indicatesLiveDaemon)
        XCTAssertTrue(unknownPidPortOpen.indicatesLiveDaemon)
        XCTAssertFalse(gone.indicatesLiveDaemon)
        XCTAssertFalse(portClosed.indicatesLiveDaemon)
        XCTAssertFalse(unknownPidPortClosed.indicatesLiveDaemon)
    }
}

final class TCPConnectProbeTests: XCTestCase {
    /// A listening socket that never accepts: the frozen-daemon shape. The
    /// kernel completes the handshake into the backlog, so the probe must
    /// report the port as accepting.
    private func listenOnLoopback() throws -> (fd: Int32, port: Int) {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        XCTAssertGreaterThanOrEqual(fd, 0)
        var reuse: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(0).bigEndian
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
        let bound = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        XCTAssertEqual(bound, 0)
        XCTAssertEqual(listen(fd, 4), 0)
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let named = withUnsafeMutablePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getsockname(fd, $0, &length)
            }
        }
        XCTAssertEqual(named, 0)
        return (fd, Int(UInt16(bigEndian: address.sin_port)))
    }

    func testAListeningPortThatNeverAnswersStillCountsAsAccepting() throws {
        let (fd, port) = try listenOnLoopback()
        defer { close(fd) }
        XCTAssertTrue(TCPConnectProbe.accepts(host: "127.0.0.1", port: port, timeoutSeconds: 1))
        XCTAssertTrue(
            TCPConnectProbe.accepts(url: URL(string: "http://127.0.0.1:\(port)")!, timeoutSeconds: 1)
        )
    }

    func testAClosedPortIsRefusedWithinTheBudget() throws {
        let (fd, port) = try listenOnLoopback()
        close(fd)
        let started = Date()
        XCTAssertFalse(TCPConnectProbe.accepts(host: "127.0.0.1", port: port, timeoutSeconds: 1))
        XCTAssertLessThan(Date().timeIntervalSince(started), 1.5)
    }

    func testBracketedHostAndInvalidPortsAreHandled() {
        XCTAssertFalse(TCPConnectProbe.accepts(host: "127.0.0.1", port: 0))
        XCTAssertFalse(TCPConnectProbe.accepts(host: "127.0.0.1", port: 70_000))
        // A bracketed literal parses; whether ::1 accepts depends on the
        // port, so only the "refused quickly" half is asserted.
        let started = Date()
        _ = TCPConnectProbe.accepts(host: "[::1]", port: 1, timeoutSeconds: 1)
        XCTAssertLessThan(Date().timeIntervalSince(started), 1.5)
    }
}
