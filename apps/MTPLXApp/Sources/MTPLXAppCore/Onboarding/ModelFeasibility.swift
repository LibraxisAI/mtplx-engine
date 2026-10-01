import Foundation

// MARK: - ModelFeasibility
//
// Pure function that decides whether a curated model is a safe pick
// for the detected hardware. Drives the model-pick step's per-card
// badge (Recommended / Tight fit / Insufficient memory / Insufficient
// disk) and the green-amber-red border treatment.
//
// We do this on the Swift side rather than trusting the daemon's
// doctor command because doctor's RAM check is hardcoded to the
// Speed model size (`mtplx/diagnostics.py:29`, `DEFAULT_SPEED_MODEL_SIZE_BYTES`)
// and has no per-variant logic — so on a 24 GB M2 Air it would
// happily let the user select Quality (~28 GiB peak) which would
// OOM at first prompt.
//
// Rules:
//   1. Memory:  needed = `model.peakMemoryGiB * safetyFactor`. Below
//      `model.peakMemoryGiB`: insufficientMemory. Between peak and
//      `peak * 1.5`: tightFit. Above: recommended (modulo disk).
//   2. Disk:    needed = bytes still to download + 5 GiB, the rule
//      `mtplx pull` enforces (`hf_loader._require_download_disk_headroom`).
//      The app downloads through `mtplx pull`, which writes each file as
//      `.incomplete` beside its final name and renames it in place, so no
//      second copy needs room. Below: insufficientDisk.
//   3. Intel:   every model returns insufficientMemory regardless —
//      MTPLX has no first-class Intel support. The user can still
//      proceed via the Other path with a smaller model.
//
// Returns a single, exhaustive verdict so the UI never has to combine
// multiple flags.
//
// SYNC PAIR: mtplx/model_catalog.py (MEMORY_SAFETY_FACTOR,
// DOWNLOAD_HEADROOM_GIB, evaluate_feasibility) mirrors these rules for the
// CLI. Update both sides together.

public enum ModelFeasibilityVerdict: Equatable, Sendable {
    /// Safe to download and run at the daemon's defaults.
    case recommended
    /// Will run but with little headroom for KV growth at long
    /// contexts; the model card shows an amber border + warning.
    case tightFit
    /// Not enough unified memory to load the model safely. Card is
    /// disabled with the required RAM in the message.
    case insufficientMemory(needsGiB: Double)
    /// Not enough free disk to download the weights. Card is disabled
    /// with the required free space in the message.
    case insufficientDisk(needsGiB: Double)
}

public struct ModelFeasibility: Sendable {
    /// `peakMemoryGiB * memorySafetyFactor` is the floor for
    /// `.recommended`. Above peak but below this floor → `.tightFit`.
    public static let memorySafetyFactor: Double = 1.5
    /// Free space `mtplx pull` keeps beyond the bytes it still has to fetch.
    public static let downloadHeadroomGiB: Double = 5

    /// Free GiB a download of `sizeBytes` needs when `downloadedBytes` of it
    /// already sit in the model folder; 0 when the size is unknown.
    public static func requiredFreeDiskGiB(sizeBytes: Int64, downloadedBytes: Int64 = 0) -> Double {
        guard sizeBytes > 0 else { return 0 }
        return Double(max(0, sizeBytes - downloadedBytes)) / 1_073_741_824.0 + downloadHeadroomGiB
    }

    public init() {}

    public func evaluate(
        model: MTPLXModelOption,
        chipTier: ChipTier,
        ramGiB: Double,
        diskFreeGiB: Double,
        downloadedBytes: Int64 = 0
    ) -> ModelFeasibilityVerdict {
        let safeMemoryFloor = model.peakMemoryGiB * Self.memorySafetyFactor
        let diskRequired = Self.requiredFreeDiskGiB(
            sizeBytes: model.sizeBytes,
            downloadedBytes: downloadedBytes
        )

        // Disk pre-flight ahead of memory: a download blocker takes
        // priority over a runtime blocker.
        if diskFreeGiB < diskRequired {
            return .insufficientDisk(needsGiB: diskRequired)
        }

        // Intel: never recommend any catalog model. The user can
        // proceed via Other with a much smaller bespoke model.
        if chipTier == .intel {
            return .insufficientMemory(needsGiB: safeMemoryFloor)
        }

        if ramGiB < model.peakMemoryGiB {
            return .insufficientMemory(needsGiB: safeMemoryFloor)
        }
        if ramGiB < safeMemoryFloor {
            return .tightFit
        }
        return .recommended
    }
}
