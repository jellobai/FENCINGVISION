# Project Notes

## Working Name

FencingVision

## Project Framing

This project studies whether saber bout video can be converted into useful tactical signals for a saber training assistant.

Useful early signals:

- distance between fencers over time
- closing speed
- movement bursts
- initiative shifts
- simple forward and backward phases

## Good MVP Output

- annotated video
- distance-over-time graph
- tempo-change markers
- likely attack-initiation timestamps
- short textual summary

## 8-Week Prototype Path

1. Collect a small set of consistent side-view bout videos.
2. Set up a person or pose detector.
3. Track both fencers frame to frame.
4. Compute distance, velocity, and acceleration.
5. Add simple rule-based event detection.
6. Evaluate on manually labeled clips.
7. Build a lightweight dashboard.
8. Record a demo and write up findings.
