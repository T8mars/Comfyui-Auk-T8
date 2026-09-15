# Comfyui-Auk-T8 2.0.5

- Fixes stale target-duration validation after an upstream `Load Audio` trim; the node now trusts the audio it actually receives.
- Applies official duration rules to speed, emotion, speech/lyric content, and nonverbal edits.
- Adds task-aware speech edge trimming, official whisper input loudness, and peak protection for lyric and music-separation output.
- Adds quality repair and defaults the loader to the higher-quality AuK Base.
- Ships 17 drag-and-drop ComfyUI workflows covering every UI task entry and all 16 upstream low-level tasks.
- Adds regression coverage for exact official prompts, crop duration, preprocessing, workflow structure, and real Base inference.
