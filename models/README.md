# Language-identification model

The prose-language analysis uses fastText's `lid.176.bin` model. The model is
131,266,198 bytes and is therefore not committed to Git.

The analysis script downloads it when `--download-model` is supplied and
checks this SHA-256 digest:

```text
7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e
```

The expected local path is:

```text
models/lid.176.bin
```

