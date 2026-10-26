# Molmo notes (researcher agent report)

Status: researched from primary sources, **snippet not run yet**. Items under "Unverified" are open questions to test, not facts.

## Checkpoints and licences
- Original Molmo collection: Molmo-72B-0924, Molmo-7B-D-0924, Molmo-7B-O-0924, MolmoE-1B-0924 (1.5B active / 7.2B total, MoE). The MolmoE card does not mention pointing, so pointing is unconfirmed there.
- **Molmo 2** exists (Ai2 blog, Dec 11 2025): 8B and 4B (Qwen3 base) and Molmo2-O 7B (Olmo base). Ai2 says 4B and 8B handle pointing, and that 8B beats the original 72B on key pointing/grounding benchmarks.
- A later **MolmoPoint** family (MolmoPoint-8B, MolmoPoint-Vid-4B, MolmoPoint-GUI-8B) emits point *tokens* and needs a custom logit processor. More complex: don't start there.
- Apache 2.0, "research and educational use". Molmo 2 also used third-party non-commercial research datasets. MolmoPoint README names no licence.
- **Best fit for a T4:** Molmo-7B-D-0924 (most documented, most community code) or Molmo2-4B. No verified ranking between them for this task.

## Prompting and output format (original Molmo)
- Example prompts from a third-party walkthrough: "Point where the dog is." / "Point to all the people". The paper says ~100 question templates were used, so wording is flexible. "How many" questions can be prefixed with "Answer without points."
- Single point: `<point x="10.0" y="10.0" alt="...">text</point>`
- Several: `<points x1="10.0" y1="10.0" x2="20.0" y2="20.0" alt="...">text</points>` (the paper's appendix writes the second `y` as `y1`: looks like a typo).
- Coordinates scaled **0-100**; the third-party code treats them as percent of image width/height. The paper only says "scaled to 0-100".
- Points are ordered top-down, left-to-right and numbered; the paper says the last number is the count.
- **Molmo2-4B uses a different scale (0-1000)**, converted with `x/1000*image_w`. Do not mix the two.

```python
import re
PT = re.compile(r'x(\d*)="([\d.]+)"\s+y\1="([\d.]+)"')   # matches x="" y="" and x1="" y1=""
def to_pixels(text, w, h):
    return [(float(x)/100*w, float(y)/100*h) for _, x, y in PT.findall(text)]
```

## Loading on a T4 (16 GB, ~15 GB usable)
- bf16 weights are ~16 GB (arithmetic from "8B" params): likely too big.
- Reports found: bf16 ran on a 24 GB RTX 4090; an A100 80 GB OOM'd with the card's default float32 snippet.
- **4-bit bitsandbytes is the likely route.** Pre-quantized 4-bit repo (cyan2k) claims ~7 GB disk / ~12 GB VRAM; a blog author reports ~4.5 GB to load + 1.5-2 GB inference. These disagree: **measure it yourself**.
- `trust_remote_code=True` required; install `einops` and `torchvision`.

## Minimal snippet (model card + 4-bit config added by the agent, NOT run)
```python
# pip install -q einops torchvision bitsandbytes accelerate transformers==4.45.2  # version is a GUESS
import torch, re
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, BitsAndBytesConfig
name = 'allenai/Molmo-7B-D-0924'
processor = AutoProcessor.from_pretrained(name, trust_remote_code=True, device_map='auto')
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type='nf4')
model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, quantization_config=bnb, device_map='auto')
img = Image.open('wheel.jpg').convert('RGB')
inputs = processor.process(images=[img], text='Point to all the hand-drawn icons.')
inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
with torch.no_grad():
    out = model.generate_from_batch(inputs, GenerationConfig(max_new_tokens=300, stop_strings='<|endoftext|>'),
                                    tokenizer=processor.tokenizer)
text = processor.tokenizer.decode(out[0, inputs['input_ids'].size(1):], skip_special_tokens=True)
print(text)   # then parse with to_pixels(text, *img.size)
```

## Pitfalls
- **transformers versions (biggest risk):** card gives no pin; config saved with 4.43.3. With 4.56.2 the remote code fails (`AttributeError: 'NoneType' object has no attribute 'size'` at `past_key_values[0][0].size(-2)`; patch in HF discussion #50, one user says it still failed). On Transformers v5: `'MolmoForCausalLM' object has no attribute 'all_tied_weights_keys'` (#51). 4.45.2 is **unverified**: test it, fall back to a nearby 4.4x.
- vLLM: card says 0.7.2 or earlier. Molmo2-4B pins `transformers==4.57.1`.
- Convert images to RGB (else broadcast error); transparent images can also trouble it.
- T4 has no native bf16 (believed, unverified), hence float16 compute.
- If no points are returned, a naive parser breaks: handle the empty case.

## Concepts to learn (answer these in your own words)
1. **Pointing is text generation:** coordinates come out as a string you parse, not a tensor.
2. **Normalised coordinates:** 0-100 (Molmo 1) vs 0-1000 (Molmo 2): convert with the right width/height.
3. **Quantisation:** 4-bit weights trade a little accuracy for the memory that lets the model fit a T4.
4. **Prompt as search spec:** "Point to all X" vs "Point to X" changes recall.
5. **Points as SAM prompts:** a point says "something is here"; SAM turns it into a mask.

## Unverified
- That 4-bit Molmo-7B-D fits a T4, and its real VRAM use.
- Which transformers version works with the original remote code.
- That 0-100 is relative to full image width/height (only a third-party walkthrough says so).
- That "Point to all..." reliably finds small hand-drawn red icons. The paper says counting data excluded images with >40 targets. Expect misses; try prompts like "Point to every red drawing."
- Whether Molmo2-4B beats Molmo-7B-D here; Molmo 2's image-pointing prompt wording; whether Molmo 2 runs without `trust_remote_code`.

## Sources
- https://huggingface.co/allenai/Molmo-7B-D-0924 (+ discussions /1, /42, /50)
- https://huggingface.co/allenai/MolmoE-1B-0924
- https://huggingface.co/allenai/Molmo2-4B
- https://huggingface.co/cyan2k/molmo-7B-D-bnb-4bit
- https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19
- https://arxiv.org/html/2409.17146
- https://allenai.org/blog/molmo2
- https://github.com/allenai/molmo2/blob/main/MOLMO_POINT_README.md
- https://debuggercafe.com/introduction-to-molmo/ (third party)
