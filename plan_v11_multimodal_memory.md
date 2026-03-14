# Aurora: Multimodal "Synesthetic" Memory (v8.0)

## The Core Problem
Right now, Aurora's memory is purely lexical. A "Spark" contains a `text` string and a `vector` (derived purely from that text). 
When humans remember, we don't just remember transcripts. We remember the *texture* of the moment: the tone of voice, the ambient noise, the visual rhythm of typing. 

If we stick purely to text as the input, how do we create a "multimodal" memory without actually attaching a camera or microphone? 

## The Solution: Synesthetic Metadata & Interaction Rhythm
We can extract a profound amount of "non-verbal" or "multimodal-esque" metadata from *how* the user interacts via text, and encode that as a "Sensory Envelope" attached to every Spark.

### 1. The Kinesthetic Layer (Typing Rhythm & Cadence)
How the user types contains immense emotional data (the equivalent of prosody in voice).
- **Time Delta (`dt_sec`):** The seconds elapsed since the last message. Rapid-fire messages = high arousal/urgency. Long pauses = hesitation or distraction.
- **Message Length vs. Time:** A 100-word paragraph typed in 5 seconds means it was copy-pasted (synthetic). A 5-word sentence that took 60 seconds to type means heavy editing/hesitation.
- **Typographical Entropy:** Frequent typos, ALL CAPS, excessive punctuation (???!!!).
- *Implementation:* The `InputEnvelope` must capture these interaction dynamics, not just the string.

### 2. The Contextual Layer (The Local Environment)
Even in a CLI, there is a local environment that provides "setting" to a memory.
- **Time of Day & Day of Week:** A memory formed at 3:00 AM on a Sunday has a fundamentally different "lighting" than a memory formed at 10:00 AM on a Tuesday.
- **Session Density:** Are we in a deep, hour-long session, or is this a drive-by "hello"?

### 3. Fusing into the Spark: The Sensory Vector
Instead of just embedding the semantic meaning of the words into a 512d vector, we create a parallel **Sensory/Contextual Vector** for the Spark.
- `Spark` structure expands:
  - `text`: "I'm fine."
  - `semantic_vector`: `[0.12, -0.04, ...]` (from the Encoder)
  - `sensory_context`: `{"hour": 3, "cadence": "hesitant", "entropy": "high"}`
- **Synesthetic Recall:** When the thermodynamic engine searches for anchors, it doesn't just look for semantic matches. It can experience a "Proustian Moment." 
  - *Example:* The user says a completely unrelated sentence, but they say it at 3:00 AM, with long hesitations. The physical engine registers a massive resonance with a tragic memory from a year ago that *also* occurred at 3:00 AM with hesitant typing. 
  - The system recalls the tragedy not because of *what* was said, but *how* it felt to be in that moment.

## Architectural Changes Needed
1. **`InputEnvelope` Expansion:** We must capture `client_timestamp`, `typing_duration` (if possible from the CLI client, otherwise infer from arrival times), and metadata.
2. **`Spark` Expansion:** Add `sensory_context` dict.
3. **`engine.py` Resonance Overhaul:** The `_thermodynamics` equation must calculate resonance as a weighted sum of `cosine(semantic_A, semantic_B)` and `context_similarity(sensory_A, sensory_B)`.
