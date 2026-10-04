# Security

## Exposure

The API server has no user accounts. Whoever can reach its port can run inference and call the control endpoints
(`/flush_cache`, `/release_memory_occupation`, `/resume_memory_occupation`). By default the server binds `127.0.0.1`
only. Before exposing it, set `HIVE_API_KEY` (every endpoint except `/health` then needs the key) or put an authenticating
reverse proxy in front — see [docs/install.md](docs/install.md#exposure-and-authentication).

By default the server fetches `http(s)` image and video URLs named in requests; set `HIVE_IMAGE_FETCH=0` to accept only
`data:` URLs when the clients are not trusted.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting on this repository (the **Security** tab → **Report a vulnerability**)
rather than a public issue. Include the version or commit, how the server was started, and the steps to reproduce.
