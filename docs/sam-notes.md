# SAM notes (researcher agent report)

Status: researched from primary sources, **snippet not run yet**. Items under "Unverified" are open questions to test, not facts.

## Versions and what they accept
| Version | Prompts | Notes |
|---|---|---|
| SAM 1 (`segment-anything`) | points, boxes, mask input | Apache 2.0 |
| SAM 2 / 2.1 (`facebookresearch/sam2`) | points, boxes, masks (images + video) | Apache 2.0. 2.1 sizes: tiny 38.9M, small 46M, base_plus 80.8M, large 224.4M params. `facebook/sam2.1-hiera-large` is not gated |
| SAM 3 (`facebookresearch/sam3`, 848M params) | text and exemplar "concept" prompts, plus points/boxes/masks | Needs Python 3.12+, PyTorch 2.7+, CUDA 12.6+. Gated on Hugging Face (request access). "SAM License" |
| SAM 3.1 (27 Mar 2026) | drop-in update focused on multi-object video tracking | Gated too |

In transformers: SAM 3 text prompts use `Sam3Model`/`Sam3Processor`; point/box prompts use `Sam3TrackerModel`/`Sam3TrackerProcessor` (`facebook/sam3`).

## Recommendation
**SAM 2.1 large via transformers (`Sam2Model`).**
- Molmo gives points, which is exactly SAM 2's prompt type.
- Not gated, Apache 2.0: no access wait.
- `pip install -U transformers accelerate`, no CUDA extension build.
- SAM 3 text prompts are worth an *experiment* later, but SAM 3's own blog says it generalises poorly to fine-grained, out-of-domain concepts (hand-drawn icons are out-of-domain). Switching later is mostly a class-name and model-id change.

## Minimal snippet (from the HF docs, NOT run)
```python
import numpy as np, torch
from PIL import Image
from transformers import Sam2Processor, Sam2Model

model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to("cuda")
processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")

img = Image.open("wheel.jpg").convert("RGB")
x, y = 500, 375                       # pixel point, e.g. from Molmo
inputs = processor(images=img, input_points=[[[[x, y]]]],
                   input_labels=[[[1]]], return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model(**inputs)             # multimask_output=True by default

masks = processor.post_process_masks(out.pred_masks.cpu(),
                                     inputs["original_sizes"])[0]  # (1,3,H,W) bool
best = out.iou_scores.cpu().flatten().argmax().item()   # pick highest-scoring of 3
m = masks[0, best].numpy().astype(bool)

rgba = np.zeros((*m.shape, 4), np.uint8)
rgba[m] = (0, 120, 255, 160)          # semi-transparent blue
overlay = Image.alpha_composite(img.convert("RGBA"), Image.fromarray(rgba))
overlay.save("overlay.png")
```
Nesting of points: `[image][object][point][x, y]`; labels: `[image][object][point]`.
The `iou_scores` indexing is the agent's reading of the documented shapes; check it when you run it.

## Pitfalls
- Coordinates are **(x, y) in original-image pixels**, not (row, col).
- Molmo's points may be normalised (percent of width/height): convert to pixels first.
- bf16 may not be efficient on a T4: prefer fp16 or fp32 (unverified).
- **Thin line drawings:** the SAM 2 paper says the model struggles with very thin or finely detailed objects (stated for video tracking). Nothing found on line drawings specifically, so "mask covers only a stroke instead of the whole icon" is an *expectation to test*.
- Remedies (general practice): box prompt (boxes usually beat single points); several positive points on different strokes plus negative points (label 0) on the white disc; keep `multimask_output=True` for a single click; dilate/close the mask to bridge stroke gaps; crop and upscale each segment first.

## Concepts to learn (answer these in your own words)
1. **Mask:** a per-pixel yes/no image the size of the photo.
2. **Logits vs score:** logits are raw per-pixel numbers (mask = logit > 0); `iou_scores` is the model's own *guess* of mask quality, not ground truth.
3. **`multimask_output`:** one click is ambiguous (stroke? icon? whole disc?), so True returns 3 nested candidates; False returns one, better with a box or several points.
4. **Point labels:** 1 = part of the object, 0 = not part of it.
5. **Prompt hierarchy:** box > several points > a single point, in how much it constrains the answer.

## Unverified
- Real T4 speed / VRAM and checkpoint file sizes.
- That the snippet runs unchanged.
- That SAM 2.1 handles red-on-white line icons well.
- SAM 3 licence text; whether Colab's default Python meets SAM 3's 3.12+ need; SAM 3.1 transformers support.

## Sources
- https://github.com/facebookresearch/sam2
- https://github.com/facebookresearch/sam3
- https://github.com/facebookresearch/segment-anything
- https://huggingface.co/docs/transformers/model_doc/sam2
- https://huggingface.co/docs/transformers/model_doc/sam3
- https://huggingface.co/docs/transformers/model_doc/sam3_tracker
- https://huggingface.co/facebook/sam3
- https://huggingface.co/facebook/sam2.1-hiera-large
- https://ai.meta.com/blog/segment-anything-model-3/
- https://arxiv.org/pdf/2408.00714 (SAM 2 paper)
