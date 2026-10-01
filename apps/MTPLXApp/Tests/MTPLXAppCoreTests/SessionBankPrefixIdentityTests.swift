import Foundation
import XCTest
@testable import MTPLXAppCore

final class SessionBankPrefixIdentityTests: XCTestCase {
    func testMultipleSnapshotsInOneConversationKeepDistinctStableRows() throws {
        let payload = """
        [
          {"session_id":"coding-session","token_hash":"first","prefix_len":6995,"hits":0,"nbytes":640,"created_at_s":1,"last_access_s":1},
          {"session_id":"coding-session","token_hash":"second","prefix_len":8920,"hits":1,"nbytes":800,"created_at_s":2,"last_access_s":3},
          {"session_id":"coding-session","token_hash":"branched","prefix_len":8920,"hits":0,"nbytes":800,"created_at_s":2,"last_access_s":2}
        ]
        """
        var prefixes = try JSONDecoder().decode([SessionBankPrefix].self, from: Data(payload.utf8))
        XCTAssertEqual(Set(prefixes.map(\.id)).count, 3)

        let identities = prefixes.map(\.id)
        prefixes[0].hits += 1
        prefixes[0].lastAccessS = 4
        XCTAssertEqual(prefixes.map(\.id), identities)
    }

    func testOlderPayloadsWithoutTokenHashesRemainDistinct() throws {
        let payload = """
        [
          {"session_id":"coding-session","prefix_len":512,"hits":0,"nbytes":64,"created_at_s":1,"last_access_s":1},
          {"session_id":"coding-session","prefix_len":1024,"hits":1,"nbytes":128,"created_at_s":2,"last_access_s":3}
        ]
        """
        var prefixes = try JSONDecoder().decode([SessionBankPrefix].self, from: Data(payload.utf8))
        XCTAssertEqual(Set(prefixes.map(\.id)).count, 2)
        XCTAssertNil(prefixes[0].tokenHash)

        let identity = prefixes[1].id
        prefixes[1].hits += 1
        prefixes[1].lastAccessS = 4
        XCTAssertEqual(prefixes[1].id, identity)
    }
}
