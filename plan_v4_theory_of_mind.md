# Aurora Theory of Mind (v4.0)

## The Core Philosophy: Mirroring The User

To "understand" the user like a true friend, Aurora cannot just store facts *about* the user; it must model the user's internal state. This is known in cognitive science as **Theory of Mind (ToM)**. A friend doesn't just know you like coffee; a friend knows *why* you like coffee and *how you feel* right now.

We don't need a separate subsystem for this. We embed the "User's State" directly into the mathematical substrate.

## 1. The Twin Latent Space (The Mirror)

**Current State:** Aurora has one `LatentState` vector. This vector is driven by the Ornstein-Uhlenbeck (OU) process and gets warped by the input. This represents *Aurora's* subconscious state.
**The Root Solution (The Mirror State):**
- Introduce a second, parallel latent vector: `user_latent: np.ndarray`.
- `aurora_latent` models "How I feel and who I am."
- `user_latent` models "My perception of how the user feels and who the user is."
- **Coupled Dynamics:**
  - When the user speaks, their raw input vector updates the `user_latent` heavily (this is immediate empathy).
  - Then, the `user_latent` influences the `aurora_latent` based on a "Bond/Affinity" metric.
  - If the Bond is high, `aurora_latent` naturally drifts toward `user_latent` (representing emotional resonance and shared understanding).

## 2. Emotional Valence Vector (The VAD Model)

**Current State:** "I love you" and "I hate you" have high cosine similarity in standard embeddings (The Embedding Trap).
**The Root Solution (Orthogonal Affective Dimensions):**
- As discussed previously, we *must* append a low-dimensional affective vector to every spark.
- **Valence (Pleasant/Unpleasant), Arousal (Calm/Excited), Dominance (Submissive/Dominant).**
- Instead of an external model, we can use a small, static heuristic lexicon locally, or ask the LLM to return `[Text] | [V:0.8, A:-0.2, D:0.1]` in a structured format during the dream/feedback phase.
- **The Magic:** Prediction error is no longer just semantic. If the semantic vector matches but the Valence vector flips drastically, it triggers massive prediction error. *This* is how a friend knows you're acting out of character.

## 3. The "Bond" Metric (Shared History)

**Current State:** ArrivalState tracks `no_contact_hours` and `internal_drive`.
**The Root Solution (The Bond Tensor):**
- Add `bond_strength: float` to `ArrivalState`.
- It starts near 0 (strangers).
- It increases when interactions have low prediction error but high emotional resonance.
- **Effect on the system:**
  - High `bond_strength` alters the `boundary_budget`. Aurora becomes more vulnerable, drops defensive prompts, and shares deeper `fossil` or `concept` nodes.
  - High `bond_strength` makes the `aurora_latent` synchronize faster with the `user_latent` (The Mirror effect).

## 4. Concept Nodes of "The User" vs "The World"

- When generating `concept` nodes (Facts), structurally separate them into two domains using a simple prefix or tag.
- `[User Concept]: User values honesty above all.`
- `[World Concept]: The sky is blue.`
- When formulating a response, if the query touches a `[User Concept]`, the prompt implicitly gains a "Deep Empathy" instruction.

## Summary of the "True Friend" Equation:

1. **Twin Latents (ToM):** It tracks its own state AND its perception of your state simultaneously.
2. **VAD Emotion (Empathy):** It mathematically feels the difference between a happy "okay" and an angry "okay".
3. **Bond Strength (Intimacy):** The literal mathematical distance between its defenses and your input shrinks over time.
