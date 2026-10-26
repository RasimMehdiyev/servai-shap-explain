# vision-shap

Learning project: produce "highlighted area" images that show *what* a vision pipeline picked in a photo, for the ServAI dashboard (a separate repo at `C:\Workdir\Develop\servai`; do not modify it from here).

The robot's own YOLO model is not available locally, so the plan is zero-shot: **Molmo** (points at things) -> **SAM 2.1** (turns a point into a mask) -> optionally **CLIP** or Molmo to name the item -> overlay PNG. A SAM mask shows *what and where*, not *why* (it is not a SHAP-style attribution).

Test images: `data/wheel_1.jpg`, `data/wheel_2.jpg`, `data/wheel_3.jpg` (prize wheel, 12 hand-drawn red icons, red pointer at the top, a different rotation in each). Open question: is the wheel a stand-in for grocery items (`grocery-shap.ipynb`)? Ask the user before the first real run.

## Setup (decided)
- Work locally in VS Code; run notebooks on a free Colab **T4** via Google's *Colab* VS Code extension (Select Kernel -> Colab). The user has already connected a T4 to `grocery-shap.ipynb`.
- Real code = plain `.py` files (`molmo_points.py`, `sam_masks.py`, currently empty). The notebook stays thin: it imports and calls them.
- The Colab kernel cannot see local files. Plan: private GitHub repo + `git pull` in the first cell; models/outputs on Google Drive (`HF_HOME` on Drive, set before importing HF libraries). The repo does not exist yet: the user creates it before milestone 1.
- Do not use SSH/tunnel workarounds (Colab free-tier terms). Never put tokens or passwords in notebooks.
- You (Claude) probably cannot run cells on the Colab kernel (unconfirmed). Default: the user runs cells and pastes output or saves PNGs to `outputs/` for you to look at.

## How to work with the user (IMPORTANT: they want to learn, not be left clueless)
- Explain the concept before the code. Take the smallest step.
- Predict-then-run: before each cell, ask the user to write down what they expect, then compare.
- Do not write more than ~15 lines at a time unless asked. The user writes the logic (e.g. converting Molmo's coordinates to pixels, working out which wheel segment is under the pointer). You may write boilerplate: installs, imports, plotting helpers.
- At the end of each milestone, quiz them on its "Concepts to learn" list (see `docs/`), then append what they learned to `docs/LEARNING.md`.
- Use subagents for narrow jobs only: a *researcher* (read docs, summarise with sources), a *reviewer* (explain what is wrong in the user's code without fixing it). No agent writes the whole pipeline.
- Do not commit or push unless asked.

## Research already done (read these first)
- `docs/sam-notes.md`: use SAM 2.1 large via transformers (`Sam2Model`); thin line drawings are a risk (try box prompts / several points); snippet not yet run.
- `docs/molmo-notes.md`: Molmo-7B-D in 4-bit, or Molmo2-4B. Points come back as text; Molmo 1 uses 0-100 coordinates, Molmo 2 uses 0-1000. **Biggest risk:** the original Molmo's remote code breaks on newer `transformers` (no confirmed working version).
- `docs/colab-vscode-notes.md`: the Colab extension workflow, file-system limits, an example first cell.
Everything marked "Unverified" in those notes is still unverified.

## Roadmap
0. Environment: connect the T4, run `!nvidia-smi` and a torch check. **Status: the user says the T4 is connected; their output has not been reviewed yet.**
1. SAM alone: one click on `wheel_1.jpg` -> one mask -> overlay.
2. Molmo alone: ask it to point; parse the `<point>` text into pixels.
3. Chain Molmo -> SAM.
4. Name each icon (CLIP or Molmo).
5. Pointer geometry: which segment is under the pointer? In all three photos the pointer seems to sit on a divider line, so a tie rule is needed (e.g. take the clockwise segment).
6. Evaluate on the 3 photos and list the failures (plain circles look alike; icons may be missed).
7. Package as a small service the dashboard can call.
