import Foundation
import XCTest

@testable import MTPLXAppCore

// MARK: - StopKeepsSessionTests
//
// A conversation is one daemon session for its whole life. Until 2.11.4
// the view model gave the conversation a fresh session id after every
// Stop, so the next message reached the daemon as an unknown
// conversation and the whole history was prefilled again. The founder's
// request log, 2026-09-20: the message sent again after a Stop carried a
// new X-MTPLX-Session-Id, arrived as 26,294 prompt tokens with
// cache_miss_reason ssd_prefix_miss and no canonicalization record, and
// took 24.5 s to its first token, while the turn before it, on the
// original id, had been canonicalized and restored. The daemon never
// commits a cancelled generation, so the original session is exactly
// where the next message belongs.

final class StopKeepsSessionTests: XCTestCase {

    @MainActor
    func testTheMessageAfterAStopCarriesTheConversationsSessionId() async throws {
        // Every chat request leaves the session id it named in a file
        // next to the cancel marker; the first turn parks after a
        // fragment (the Stop lands there), the second completes.
        let daemon = try await ChatFakeDaemon.start(chatHandler: """
        with open(os.path.join(os.path.dirname(CANCEL_MARKER), "session-%d" % count), "w") as f:
            f.write(self.headers.get("X-MTPLX-Session-Id") or "")
        self.wfile.write(sse({"id": "chatcmpl-%d" % count,
            "choices": [{"index": 0, "delta": {"role": "assistant"}}]}))
        if count == 1:
            self.wfile.write(sse({"id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {"content": "partial answer"}}]}))
            self.wfile.flush()
            time.sleep(3)
            return
        self.wfile.write(sse({"id": "chatcmpl-2",
            "choices": [{"index": 0, "delta": {"content": "Whole answer."}}]}))
        self.wfile.write(sse({"id": "chatcmpl-2",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
        self.wfile.write(b"data: [DONE]\\n\\n")
        """)
        defer { daemon.terminate() }
        let (viewModel, container) = try makeChatViewModel(daemon: daemon)

        viewModel.send("start a long answer")
        let conversation = try XCTUnwrap(viewModel.current)
        try await pollUntil("the first fragment arrives") { viewModel.hasStreamingContent }

        await viewModel.cancel()
        try await pollUntil("the cancelled turn settles") { !viewModel.isStreaming }
        XCTAssertTrue(daemon.cancelWasCalled)
        XCTAssertNil(viewModel.lastError, "a user stop is not an error")

        viewModel.send("and again")
        try await pollUntil("the second turn settles") {
            !viewModel.isStreaming
                && viewModel.visibleMessages.last?.visibleContent == "Whole answer."
        }

        let markers = daemon.cancelMarkerURL.deletingLastPathComponent()
        let first = try String(
            contentsOf: markers.appendingPathComponent("session-1"), encoding: .utf8
        )
        let second = try String(
            contentsOf: markers.appendingPathComponent("session-2"), encoding: .utf8
        )
        XCTAssertFalse(first.isEmpty, "every request names its session")
        XCTAssertEqual(
            first.lowercased(), conversation.id.uuidString.lowercased(),
            "the session id is the conversation id"
        )
        XCTAssertEqual(
            second, first,
            "a Stop must not move the conversation to a new daemon session"
        )

        let persisted = try persistedAssistantMessages(in: container, conversationID: conversation.id)
        XCTAssertEqual(persisted.map(\.visibleContent), ["partial answer", "Whole answer."])
        XCTAssertEqual(persisted.map { $0.finishReason ?? "" }, ["cancelled", "stop"])
    }
}
