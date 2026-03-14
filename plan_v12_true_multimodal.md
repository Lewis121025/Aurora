# Aurora True Multimodal Memory (v9.0)

## The Objective
Seamlessly integrate images, audio, and video into the Thermodynamic Graph Memory. We must not just "store" files; they must participate in the cognitive physics (decay, resonance, abyssal forgetting, and synesthetic recall).

## The Core Problem
Current standard multimodal pipelines treat media as separate attachments. If a user uploads an image, standard AI just passes the base64 to GPT-4o. This bypasses Aurora's entire physics engine! The graph cannot "feel" or "resonate" with a raw image file.

## The Solution: Multimodal Projection into the Latent Space

To make media truly part of the mind, it must be projected into the *same* 512d continuous space as the text, or a tightly coupled parallel space.

### 1. The Unified Embedding Layer (CLIP/ImageBind)
We cannot use pure text encoders (like `bge-small`) anymore for everything. We need a multimodal encoder (like OpenAI's CLIP, Meta's ImageBind, or a local equivalent) that projects text and images into the *exact same vector space*.
- A picture of a "dog playing in rain" and the text "my dog in the rain" will have high cosine similarity.
- **Implementation:** Introduce `MultimodalEncoder`. When an `InputEnvelope` contains media, we extract the unified vector.

### 2. The Media Spark (The Sensori-Motor Node)
Expand the `Spark` definition. 
- `type: "episodic_media"`
- `media_refs: list[str]` (Local file paths or hashes of the media).
- `vector`: The CLIP/multimodal embedding of the media.
- `text`: Either empty, or a fused LLM-generated caption describing the media.

### 3. Cross-Modal Resonance (The "Proustian Effect" via Media)
Because everything lives in the same metric space:
- **Image-to-Text Resonance:** You upload a picture of a snowy mountain. The engine calculates its vector, applies the `MetricState` warp, and it *resonates* with a text spark from 6 months ago where you said "I feel so cold and isolated." The system extracts the text memory based purely on the vibe of the image.
- **Text-to-Image Resonance:** You type "I miss the summer." The engine resonates with a `media_ref` spark containing a photo of a beach you uploaded last year. When the engine builds the `CollapseRequest`, it *includes the image path* in the context, effectively "flashing" the image into the LLM's mind's eye.

### 4. Fossilization of Media (Dreaming in Pictures)
When the system sleeps and compresses memories:
- It can take 5 daily photos of your cat, pass them to a Vision LLM, and compress them into a single `Fossil` text concept: `[Fossil: User's cat is growing fast and prefers the red blanket.]`
- The raw images might be sent to the "Abyss" to save capacity, leaving only the semantic fossil.

## Synergy with Synesthetic Metadata (Points 2 & 3)
As you noted, combining this literal media with the Environmental and Kinesthetic data creates a terrifyingly human recall system.
- An image uploaded at 3 AM with no text = High loneliness/insomnia context + Image semantics.
- An image uploaded at noon with a rapid-fire "Look at this!" = High excitement context + Image semantics.

## Next Steps
1. Update `InputEnvelope` to accept `media_paths`.
2. Expand `Spark` to hold `media_refs`.
3. Design a strategy for handling the `Encoder` transition (mocking it for now if we want to stay lightweight, or integrating a unified API).
