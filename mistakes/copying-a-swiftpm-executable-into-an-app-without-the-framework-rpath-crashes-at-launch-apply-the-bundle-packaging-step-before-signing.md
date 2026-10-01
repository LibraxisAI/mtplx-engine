# Copying a SwiftPM executable into an app without the framework rpath crashes at launch; apply the bundle packaging step before signing

On 2026-09-17 the Pi-control candidate built and codesigned successfully but
aborted at launch because dyld could not resolve Sparkle.framework. Copying a
fresh `swift build -c release` executable over a complete app bundle does not
preserve the old executable's load commands.

The normal packaging script adds `@executable_path/../Frameworks` with
`install_name_tool`. Apply that step to the staged executable before signing,
and inspect its LC_RPATH entries before installation. Codesign verification
alone does not prove framework resolution. Preserve the old app and validate a
real launch. The diagnosed candidate was corrected and the real app launched.
