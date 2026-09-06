# Mobile application boundary

Planned: React Native + TypeScript for iOS and Android. No runnable mobile app
is scaffolded yet; do not mistake this directory for a working application.

First slice: load `tests/fixtures/performance.json`, display the notes, and allow
editing string/fret assignments. Use the contract in `packages/music`.

Native capture and inference should emit timestamped image-space observations.
Keep camera buffers out of React state. Display transforms belong to the UI.
