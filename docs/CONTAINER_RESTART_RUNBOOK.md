# Container restart and persistent mounts

The complete research state lives below `/home/dbw`: repository and raw
outputs in `/home/dbw/ANCHOR`, restricted datasets in `/home/dbw/datasets`,
model weights in `/home/dbw/models`, and model-specific environments/code in
the remaining subdirectories. The current filesystem reports `/home/dbw` as a
host ext4 subdirectory mount rather than container overlay storage.

For a replacement container, bind the **same existing persistent host path**
to the **same container path** `/home/dbw`. Do not create or mount a new empty
volume over `/home/dbw`; it would hide the existing files. Mounting only
`/home/dbw/datasets` is insufficient because run fingerprints reference the
repository, model checkpoints, environments, and outputs by their frozen
absolute paths.

The trace-checked Python 3.10.20 runtime now persists under
`/home/dbw/.runtime/miniconda3`; `/opt/miniconda3` is its canonical symlink.
The Hugging Face cache also persists under `/home/dbw/.cache/huggingface`.
`resume_after_container_restart.sh` restores the symlink in a recreated
container. It must not fall back to system Python: changing CUDA/Transformers
packages invalidates frozen backend identities. The mount must be read-write
because the selective VinDr download is incomplete.

Before restart, run:

```bash
cd /home/dbw/ANCHOR
bash scripts/prepare_container_restart.sh
```

The command freezes a live RAG process group at its prefix-safe checkpoint,
checkpoints the manifest hashes and DICOM count, writes a persistent-volume
sentinel, stops the resumable wget and downstream waiters, pauses recovery,
and calls `sync`.

After `/home/dbw` is mounted in the new container at the same path:

```bash
cd /home/dbw/ANCHOR
bash scripts/resume_after_container_restart.sh
```

The resume command fails closed if the sentinel, models, annotations, runtime,
or GPU are missing. Before appending to an unfinished result it reruns frozen
32-case Hulu and LLaVA fixtures and requires 100% normalized text, token F1,
and generated-token-ID identity. VinDr uses `wget -c`, so complete files are
retained and the partial final file resumes. The user must attach once and
enter the PhysioNet password interactively; it is never stored:

```bash
tmux attach -t vindr-selective-download
```
