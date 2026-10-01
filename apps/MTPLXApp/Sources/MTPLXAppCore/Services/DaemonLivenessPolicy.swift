import Foundation

// MARK: - DaemonLivenessPolicy
//
// Issue #487: the running-daemon watchdog reaped a live daemon four times in
// one day. Each time the daemon was inside a 19-25 s generation-final prefix
// commit, answered nothing on /health for that long, missed two probes in a
// row and was killed while the commit it was doing succeeded.
//
// Product rule: a daemon whose process is alive and whose port still accepts
// a TCP connection is never reaped because of probe timeouts alone. It is
// "busy". Only a gone process, a closed port, or a long unbroken silence
// (`busyGraceSeconds`) may reap. The two older rules stay: a 2xx with a
// payload the app cannot decode is alive (2026-07-06), and a 401/403 is
// alive (issue #109) -- the store maps both to `recordAnswer()`.

/// What the watchdog learned about the daemon besides the probe itself.
public struct DaemonLivenessEvidence: Equatable, Sendable {
    /// `nil` when no pid is known (no startup payload yet, nothing owned).
    public var processAlive: Bool?
    /// The daemon's port completed a TCP handshake within the probe budget.
    public var portAccepting: Bool

    public init(processAlive: Bool?, portAccepting: Bool) {
        self.processAlive = processAlive
        self.portAccepting = portAccepting
    }

    /// Alive by every signal the app can read without HTTP: the process (when
    /// known) exists and the kernel still accepts connections on the port.
    public var indicatesLiveDaemon: Bool {
        processAlive != false && portAccepting
    }
}

public enum DaemonReapReason: Equatable, Sendable {
    /// `kill(pid, 0)` says the daemon process is gone.
    case processGone
    /// Nothing accepts TCP connections on the daemon port any more.
    case portClosed
    /// Alive by process and port, but silent on /health for longer than the
    /// busy grace.
    case unresponsiveGraceExpired(TimeInterval)
}

public enum DaemonWatchdogVerdict: Equatable, Sendable {
    /// Below the consecutive-miss threshold; a single transport blip.
    case waiting(misses: Int)
    /// Missed the threshold, but process and port say the daemon is alive:
    /// treat it as busy and keep probing.
    case busy(unresponsiveFor: TimeInterval)
    case reap(DaemonReapReason)
}

/// Pure decision state for the watchdog loop. The store feeds it one probe at
/// a time with the wall clock it wants to reason in (tests pass numbers).
public struct DaemonLivenessTracker: Equatable, Sendable {
    public var missesBeforeReap: Int
    public var busyGraceSeconds: TimeInterval
    public private(set) var consecutiveMisses = 0
    /// Clock value of the first miss of the current silent stretch.
    public private(set) var silentSince: TimeInterval?

    public init(missesBeforeReap: Int = 2, busyGraceSeconds: TimeInterval = 90) {
        self.missesBeforeReap = max(1, missesBeforeReap)
        self.busyGraceSeconds = max(0, busyGraceSeconds)
    }

    /// The daemon answered (healthy, undecodable-but-2xx, or 401/403).
    public mutating func recordAnswer() {
        consecutiveMisses = 0
        silentSince = nil
    }

    /// The probe timed out, failed at the transport, or came back non-2xx /
    /// self-reported not ok.
    public mutating func recordMiss(
        evidence: DaemonLivenessEvidence,
        now: TimeInterval
    ) -> DaemonWatchdogVerdict {
        consecutiveMisses += 1
        if silentSince == nil {
            silentSince = now
        }
        let unresponsiveFor = max(0, now - (silentSince ?? now))
        guard consecutiveMisses >= missesBeforeReap else {
            return .waiting(misses: consecutiveMisses)
        }
        if evidence.processAlive == false {
            return .reap(.processGone)
        }
        if !evidence.portAccepting {
            return .reap(.portClosed)
        }
        if unresponsiveFor >= busyGraceSeconds {
            return .reap(.unresponsiveGraceExpired(unresponsiveFor))
        }
        return .busy(unresponsiveFor: unresponsiveFor)
    }
}

// MARK: - TCPConnectProbe

/// Kernel-level liveness: does anything still accept a TCP connection on the
/// daemon's port? A frozen-but-alive server (its listener socket open, its
/// accept backlog filling) completes the handshake without ever answering
/// HTTP, which is exactly the shape the watchdog must not mistake for death.
public enum TCPConnectProbe {
    /// `host` may be an IPv4/IPv6 literal (with or without brackets) or a
    /// name; every resolved address is tried in order until one accepts.
    public static func accepts(
        host: String,
        port: Int,
        timeoutSeconds: TimeInterval = 1.0
    ) -> Bool {
        guard port > 0, port <= 65_535 else { return false }
        let bare: String = {
            let trimmed = host.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.hasPrefix("["), trimmed.hasSuffix("]") {
                return String(trimmed.dropFirst().dropLast())
            }
            return trimmed.isEmpty ? "127.0.0.1" : trimmed
        }()
        var hints = addrinfo()
        hints.ai_family = AF_UNSPEC
        hints.ai_socktype = SOCK_STREAM
        hints.ai_protocol = IPPROTO_TCP
        var resolved: UnsafeMutablePointer<addrinfo>?
        guard getaddrinfo(bare, String(port), &hints, &resolved) == 0, let first = resolved else {
            return false
        }
        defer { freeaddrinfo(resolved) }
        var cursor: UnsafeMutablePointer<addrinfo>? = first
        while let entry = cursor {
            if connectCompletes(entry.pointee, timeoutSeconds: timeoutSeconds) {
                return true
            }
            cursor = entry.pointee.ai_next
        }
        return false
    }

    /// Convenience for a daemon base URL (host + port as the app connects).
    public static func accepts(url: URL, timeoutSeconds: TimeInterval = 1.0) -> Bool {
        let port = url.port ?? (url.scheme?.lowercased() == "https" ? 443 : 80)
        return accepts(host: url.host ?? "127.0.0.1", port: port, timeoutSeconds: timeoutSeconds)
    }

    private static func connectCompletes(_ entry: addrinfo, timeoutSeconds: TimeInterval) -> Bool {
        let descriptor = socket(entry.ai_family, entry.ai_socktype, entry.ai_protocol)
        guard descriptor >= 0 else { return false }
        defer { close(descriptor) }
        let flags = fcntl(descriptor, F_GETFL, 0)
        guard flags >= 0, fcntl(descriptor, F_SETFL, flags | O_NONBLOCK) == 0 else { return false }
        let immediate = connect(descriptor, entry.ai_addr, entry.ai_addrlen)
        if immediate == 0 {
            return true
        }
        guard errno == EINPROGRESS else { return false }
        var poller = pollfd(fd: descriptor, events: Int16(POLLOUT), revents: 0)
        let budgetMs = Int32(max(0, timeoutSeconds) * 1000)
        guard poll(&poller, 1, budgetMs) > 0 else { return false }
        var socketError: Int32 = 0
        var length = socklen_t(MemoryLayout<Int32>.size)
        guard getsockopt(descriptor, SOL_SOCKET, SO_ERROR, &socketError, &length) == 0 else {
            return false
        }
        return socketError == 0
    }
}
