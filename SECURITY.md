# Security Policy

## Supported versions

Only the latest commit on `master` is supported. This is a single-user, local
application; there are no versioned releases.

## Reporting a vulnerability

Please do **not** open a public issue for a security problem. Use GitHub's
private reporting instead: **Security -> Report a vulnerability** on
<https://github.com/Sandeepsrinivasan-14/voice-chatbot/security/advisories/new>.

Include what you found, how to reproduce it, and the impact you see. You can
expect an acknowledgement within a few days; this is a hobby-scale project, so
there is no formal SLA.

## Threat model and hardening notes

The project is designed to run **entirely on your own machine**:

- **No cloud APIs.** Inference (faster-whisper, Ollama, Piper) is local. The
  only network call at runtime is a loopback request to your Ollama server.
  First-time model downloads (Whisper weights, Piper voices, optional Tamil
  TTS) do use the network, once.
- **No authentication.** `chat_web.py` and `record_web.py` have no login and
  no multi-user isolation. They bind to `127.0.0.1` by default (`WEB_HOST`).
  Do not set `WEB_HOST=0.0.0.0` on a machine reachable by untrusted networks
  without putting your own authenticated reverse proxy in front.
- **Docker:** the image binds `0.0.0.0` *inside* the container; always publish
  with `-p 127.0.0.1:5006:5006`, never `-p 5006:5006`.
- **Uploads** to the web endpoints are capped at 25 MB (`Config.max_upload_bytes`).
- **Personal data.** Continuous learning (`LEARNING_ENABLED`, off by default)
  stores corrected transcripts, and optionally voice audio, under
  `data/learning/`. That directory is git-ignored; never commit it.
- **Secrets.** No secrets are required to run the project. `.env` is
  git-ignored; only `.env.example` (placeholders, no credentials) is tracked.
  Automated and manual scans of the working tree and git history found no
  committed credentials as of the hardening pass. Pre-commit runs
  `detect-private-key` to keep it that way.
- **Dependencies** are monitored by Dependabot (pip, GitHub Actions, Docker).
