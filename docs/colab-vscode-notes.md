# Colab + VS Code workflow notes (researcher agent report)

Status: researched from primary sources, **nothing run yet**. Items under "Unverified" are open questions to test.

## The Colab extension for VS Code
- Published by Google: Marketplace `Google.colab`, repo `googlecolab/colab-vscode` (also on Open VSX). Announced 13 Nov 2025. Builds on the Jupyter extension (VS Code installs it if missing).
- **Install:** Extensions view -> search "Google Colab" -> Install.
- **Connect:** open/create a local `.ipynb` -> **Select Kernel** -> **Colab** -> **Auto Connect** (default server) or **New Colab Server** (pick a machine type). Sign in with Google in the browser.
- Adds commands: `Colab: Remove Server`, `Colab: Sign Out`, `Colab: Mount Google Drive to Server...`, plus a Resources view (RAM, disk, GPU).
- No minimum VS Code version is stated for the Colab extension (Claude Code's extension needs VS Code 1.94.0+).
- Free tier: sessions up to 12 h, idle runtimes time out (exact idle/usage limits unpublished). Colab's FAQ says GPU types "vary over time" and does not promise a T4.

## File system: the kernel cannot see your local files
Ways to move things: right-click in the Explorer -> **Upload to Colab**; the `ipywidgets` upload widget; `drive.mount()` (supported since extension v0.2.1); experimental Server Mounting.
Known gaps: `userdata.get()` (Colab Secrets) errors in the extension; `files.upload()/download()` need widget workarounds (see repo wiki); Plotly needs `pio.renderers.default = "vscode"`.
`HF_HOME` sets the Hugging Face cache location (default `$HF_HOME/hub`); set it **before** importing `huggingface_hub`.

## Claude Code and notebooks
- `Read` returns all cells + outputs of an `.ipynb` (size limits). `NotebookEdit` edits one cell by `cell_id`.
- `mcp__ide__executeCode` runs Python in the active notebook's Jupyter kernel, but only after you click Execute in a VS Code prompt, and only with the Jupyter extension, an active notebook and a Python kernel.
- **Not confirmed:** whether `executeCode` works against a Colab kernel (docs don't mention Colab). Plan on *you* running cells.
- Google's `colab-mcp` (Apache-2.0) can connect a local agent to a Colab notebook open in your browser (needs `uv`). Untested here.

## Terms of service: be careful
Colab's FAQ disallows "connecting to remote proxies", and on free usage without a positive compute-unit balance disallows "remote control such as SSH shells, remote desktops". Tunnels are not named, but **do not build SSH/tunnel workarounds on the free tier.** Use the official extension.

## Recommended workflow
1. Install the Colab + Jupyter extensions.
2. Real code = plain `.py` files in a local git repo (pushed to GitHub).
3. Thin `run.ipynb`: first cell = setup, later cells import and call your code.
4. Select Kernel -> Colab -> New Colab Server -> T4 if offered.
5. Run the first cell; check the GPU with `!nvidia-smi`.
6. Edit locally, `git push`; in the notebook `git pull` and re-run (`%load_ext autoreload`, `%autoreload 2`).
7. Save checkpoints/outputs to Drive.

## Example first cell (not run)
```python
from google.colab import drive
drive.mount("/content/drive")          # extension: supported since v0.2.1
import os
os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"   # set BEFORE importing HF libs
REPO = "/content/myrepo"
if not os.path.exists(REPO):
    !git clone https://github.com/YOU/myrepo.git {REPO}
else:
    !git -C {REPO} pull
%cd {REPO}
!pip -q install -r requirements.txt
import sys; sys.path.insert(0, REPO)
```
Private repo: use a fine-grained read-only token entered with `getpass`; never save it in the notebook.

## Pitfalls
- New session = fresh VM: re-run the setup cell every time.
- Anything not on Drive or in git is lost when the session ends.
- Free GPU availability is not guaranteed.
- Don't use `userdata.get()` for secrets.
- Clear notebook outputs before committing (e.g. `nbstripout`).
- Drive can be slow for large model caches.

## Concepts to learn (answer these in your own words)
1. **Kernel:** the process that runs your cells: here on Google's VM, not your PC.
2. **Ephemeral VM:** the disk is wiped at session end, so persistence needs Drive or git.
3. **Cache directory + environment variables:** `HF_HOME` tells libraries where to put downloads.
4. **Editable import + autoreload:** how a notebook picks up code you changed.

## Unverified
- That the free tier gives a T4 (only secondary blogs say so).
- That `mcp__ide__executeCode` works on a Colab kernel.
- That an HF cache on Drive works well with symlinks (`HF_HUB_DISABLE_SYMLINKS=1` exists for network drives, untested).
- The experimental Server Mounting / Terminal features in practice; check the repo's issues.

## Sources
- https://marketplace.visualstudio.com/items?itemName=Google.colab
- https://github.com/googlecolab/colab-vscode (wiki: User-Guide, Known-Issues-and-Workarounds)
- https://developers.googleblog.com/google-colab-is-coming-to-vs-code/
- https://research.google.com/colaboratory/faq.html
- https://research.google.com/colaboratory/local-runtimes.html
- https://code.claude.com/docs/en/vs-code
- https://code.claude.com/docs/en/tools-reference
- https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables
- https://developers.googleblog.com/announcing-the-colab-mcp-server-connect-any-ai-agent-to-google-colab/
- https://github.com/googlecolab/colab-mcp
