import Foundation

/// The daemon's last output, written to disk when a start fails.
///
/// Issue #504: the app could not start on a macOS beta, and the only evidence
/// the reporter could give was a screenshot of a one-line banner that cut the
/// reason off after "daemon exited before /…". The daemon's output lives in a
/// 500-line in-memory ring (`BoundedLogStore`) and nothing reached disk, so a
/// crash that kills the daemon before `/health` left no trace a person could
/// attach. `mtplx doctor`, which every bug report template asks for, reads
/// this file and includes its tail, so a "will not start" report carries its
/// own cause.
///
/// The ring is already safe to paste: the launch line goes in through
/// `redactedCommandLine` with every secret flag value masked, and the daemon
/// does not print prompt or answer text.
public enum StartFailureReport {
    public static let fileName = "last-failed-start.log"
    /// Enough for a Python traceback plus the start-up lines before it.
    public static let maxLines = 200

    public static func defaultURL(
        home: URL = FileManager.default.homeDirectoryForCurrentUser
    ) -> URL {
        home.appendingPathComponent(".mtplx", isDirectory: true)
            .appendingPathComponent("logs", isDirectory: true)
            .appendingPathComponent(fileName, isDirectory: false)
    }

    public static func render(
        detail: String,
        entries: [LogEntry],
        date: Date = Date(),
        appVersion: String? = nil,
        osVersion: String = ProcessInfo.processInfo.operatingSystemVersionString
    ) -> String {
        let stamp = ISO8601DateFormatter()
        let time = DateFormatter()
        time.locale = Locale(identifier: "en_US_POSIX")
        time.dateFormat = "HH:mm:ss.SSS"
        var lines: [String] = [
            "MTPLX failed start",
            "when: \(stamp.string(from: date))",
            "app: \(appVersion ?? "unknown")",
            "macos: \(osVersion)",
            "reason: \(detail)",
            "--- daemon output (last \(min(entries.count, maxLines)) of \(entries.count) lines) ---",
        ]
        for entry in entries.suffix(maxLines) {
            lines.append("\(time.string(from: entry.date)) [\(entry.stream.rawValue)] \(entry.message)")
        }
        return lines.joined(separator: "\n") + "\n"
    }

    /// Best effort by design: a failed start is already being reported to the
    /// user through the thrown error, and a full disk or a read-only home
    /// must not replace that error with a file error. Returns whether the
    /// report landed so a caller (and the test) can tell.
    @discardableResult
    public static func write(
        detail: String,
        entries: [LogEntry],
        to url: URL,
        date: Date = Date(),
        appVersion: String? = nil
    ) -> Bool {
        let text = render(detail: detail, entries: entries, date: date, appVersion: appVersion)
        do {
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try Data(text.utf8).write(to: url, options: .atomic)
            return true
        } catch {
            return false
        }
    }
}
