"""Limits and path filters used while collecting repository source files.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

MAX_TOTAL_CHARS = 32000
MAX_DOC_CHARS = 8000
MAX_PER_FILE_CHARS = 6000
READ_TIMEOUT_SECONDS = 120
MAX_FILES = 120
MAX_PATH_LIST_CHARS = 1000
# Directory listing bounds for a repository whose recursive tree came back truncated. Each
# listing costs one request against GitHub's hourly quota, so the walk stays small: walking
# directories four levels deep reaches files five components down
# (`src/components/ui/panel.tsx`), and the request budget is spent shallowest-first.
MAX_TREE_REQUESTS = 40
MAX_TREE_DEPTH = 4

IGNORED_DIRS = (
    "node_modules/", ".venv/", "venv/", "env/", "site-packages/", "dist/", "build/",
    "out/", "target/", "vendor/", "__pycache__/", ".git/", ".next/", ".nuxt/",
    ".cache/", "coverage/", ".idea/", ".vscode/", "Pods/", "bin/", "obj/",
)
# Unambiguously binary formats only: extensions that are occasionally readable text are
# left to the content check in ``selection.looks_binary``, which judges the bytes actually
# fetched instead of the name they arrived under.
IGNORED_EXTENSIONS = (
    # Images, media, and fonts.
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp", ".tif", ".tiff", ".bmp",
    ".psd", ".ai", ".eps", ".mp4", ".mov", ".webm", ".avi", ".mkv", ".mp3", ".wav",
    ".ogg", ".flac", ".m4a", ".aac", ".woff", ".woff2", ".ttf", ".otf", ".eot",
    # Archives, installers, and compiled or native binaries.
    ".pdf", ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".whl",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".lib", ".obj", ".class",
    ".jar", ".wasm", ".pyc", ".dmg", ".iso", ".msi", ".deb", ".rpm", ".apk", ".ipa",
    ".lock",
    # Serialized data, databases, and trained model weights. A weight blob tokenizes at
    # roughly one token per character, so one of them spends a whole view's budget and the
    # grader answers with pickled tensor bytes instead of a report.
    ".pt", ".pth", ".onnx", ".tflite", ".task", ".h5", ".hdf5", ".keras",
    ".safetensors", ".ckpt", ".pkl", ".pickle", ".joblib", ".npy", ".npz", ".mat",
    ".parquet", ".feather", ".msgpack", ".sqlite", ".sqlite3", ".mdb", ".db",
    # Minified or generated text that carries no readable evidence.
    ".min.js", ".min.css", ".map",
)
IGNORED_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "cargo.lock",
    "uv.lock", "composer.lock",
}
