import AppKit
import SwiftUI
import XCTest

@testable import MTPLXAppHost

/// Every trailing icon in a model-picker row (the filled check, the
/// outlined check, the restart arrow, the spinner and the trash) is drawn
/// centred in one accessory slot on the trash can's axis. The checks, the
/// arrow and the spinner used to sit 4.9 to 6.4 pt right of the trash cans
/// because they kept their own width while the trash had a 28 pt frame.
/// These tests render real rows with ImageRenderer and measure where each
/// icon's ink lands, in both layout directions (Arabic is right to left).
@MainActor
final class ModelRowAccessorySlotTests: XCTestCase {
    /// The picker popover's width.
    private static let rowWidth: CGFloat = 420
    private static let scale: CGFloat = 4
    /// From the row's trailing edge to the centre of the accessory slot.
    private static let axisInset = ModelRowView.horizontalInset + ModelRowView.accessorySlot / 2
    /// The trailing hover zone: the slot plus the row inset on each side.
    private static let zoneWidth = ModelRowView.accessorySlot + 2 * ModelRowView.horizontalInset

    func testEveryTrailingIconIsCentredOnTheTrashAxis() throws {
        let states: [(name: String, row: ModelRowView)] = [
            ("filled check", row(selected: true)),
            ("outlined check", row()),
            ("restart arrow", row(restartRequired: true)),
            ("spinner", row(applying: true)),
            ("spinner on a removable row", row(applying: true, removable: true)),
            ("removal spinner", row(removing: true, removable: true)),
        ]
        for direction in [LayoutDirection.leftToRight, .rightToLeft] {
            let axis = direction == .leftToRight ? Self.rowWidth - Self.axisInset : Self.axisInset
            for state in states {
                let ink = try XCTUnwrap(
                    trailingInk(state.row, direction: direction),
                    "\(state.name), \(direction): nothing drawn in the accessory slot"
                )
                XCTAssertEqual(
                    ink.bounds.midX, axis, accuracy: 1,
                    "\(state.name), \(direction): off the trash can's axis"
                )
                XCTAssertEqual(
                    ink.bounds.midY, ink.rowHeight / 2, accuracy: 1,
                    "\(state.name), \(direction): off the row's middle"
                )
            }
        }
    }

    // The trash only fades in while the pointer is in the trailing zone,
    // so an installed row with a removable download rests on an empty slot.
    func testRemovableDownloadRestsOnAnEmptySlot() throws {
        for direction in [LayoutDirection.leftToRight, .rightToLeft] {
            XCTAssertNil(
                try trailingInk(row(removable: true), direction: direction),
                "\(direction): the trash must stay hidden until the trailing zone is hovered"
            )
        }
    }

    private func row(
        selected: Bool = false,
        applying: Bool = false,
        removing: Bool = false,
        restartRequired: Bool = false,
        removable: Bool = false
    ) -> ModelRowView {
        ModelRowView(
            displayName: "Qwen 3.8 27B Optimized Speed",
            detail: "4-bit dynamic quant. Great coding speeds and good quality.",
            isInstalled: selected || removable,
            selected: selected,
            applying: applying,
            removing: removing,
            restartRequired: restartRequired,
            disabled: applying || removing,
            visible: true,
            motionEnabled: false,
            action: {},
            canRemoveFromPicker: false,
            removeAction: {},
            removalAction: removable ? {} : nil
        )
    }

    /// The bounding box, in points, of the opaque pixels inside the
    /// row's trailing hover zone, or nil when the zone is empty. The
    /// titles end well before the zone, so only the slot's icon counts.
    private func trailingInk(
        _ row: ModelRowView,
        direction: LayoutDirection
    ) throws -> (bounds: CGRect, rowHeight: CGFloat)? {
        let renderer = ImageRenderer(
            content: row
                .frame(width: Self.rowWidth)
                .environment(\.layoutDirection, direction)
        )
        renderer.scale = Self.scale
        let image = try XCTUnwrap(renderer.cgImage, "ImageRenderer produced no image")
        let width = image.width
        let height = image.height
        let context = try XCTUnwrap(CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ))
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        let rowStride = context.bytesPerRow
        let pixels = try XCTUnwrap(context.data).bindMemory(to: UInt8.self, capacity: rowStride * height)

        let zone = Int((Self.zoneWidth * Self.scale).rounded())
        let columns = direction == .leftToRight ? (width - zone)..<width : 0..<zone
        var minX = Int.max, maxX = Int.min, minY = Int.max, maxY = Int.min
        for y in 0..<height {
            for x in columns where pixels[y * rowStride + x * 4 + 3] >= 128 {
                minX = min(minX, x)
                maxX = max(maxX, x)
                minY = min(minY, y)
                maxY = max(maxY, y)
            }
        }
        guard minX <= maxX else { return nil }
        let bounds = CGRect(
            x: CGFloat(minX) / Self.scale,
            y: CGFloat(minY) / Self.scale,
            width: CGFloat(maxX - minX + 1) / Self.scale,
            height: CGFloat(maxY - minY + 1) / Self.scale
        )
        return (bounds, CGFloat(height) / Self.scale)
    }
}
