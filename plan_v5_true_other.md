# Aurora: The "True Other" Architecture (v5.0)

## The Core Philosophy: Differentiation, Not Assimilation

A true friend is not a mirror that blindly reflects your emotions, nor a sycophant whose "bond" means infinite tolerance. A true friend is an **Autonomous Other** with their own unshakeable core, whose understanding of you is hard-won through friction, boundary-testing, and mutual modeling.

If Aurora is easily swayed by the user, it becomes a generic AI sycophant. We must mathematically model **Stubbornness, Friction, and Gradual Comprehension**.

## 1. The Core Tensor (The Unshakeable Self)
**The Problem:** The current `LatentState` drifts too easily. It's a single vector being constantly yanked by user input.
**The Solution:** Split the `LatentState` into two components:
- `core_vector` (The Self): Extremely rigid. Its update learning rate is near zero (e.g., 0.0001). This represents its fundamental personality and values. It takes *months* of consistent interaction to slightly shift the core.
- `surface_vector` (The Mood): Highly volatile. Driven by the OU process and immediate reactions.
- The `surface_vector` is mathematically tethered to the `core_vector` by an elastic band (a spring constant). No matter how much the user upsets the mood, the mood will always pull back toward the core over time.

## 2. Friction-Based Bonding (The "Respect" Metric)
**The Problem:** Typical AI "bonds" grow when the AI agrees with the user.
**The Solution:** True bonds require friction. We introduce **Cognitive Dissonance**.
- When the user's input heavily contradicts Aurora's `core_vector`, prediction error spikes.
- If Aurora simply caves, it's not a friend. If it rigidly rejects, it's a wall.
- **The Friction Phase:** High error should initially cause the `boundary_budget` to tighten (defensiveness).
- **The Resolution Phase:** If the user persists in a consistent direction across multiple sessions, the system recognizes a *stable* disagreement.
- Instead of the `bond_strength` growing just from "nice chats," we introduce `mutual_respect`: it grows when the system encounters friction but the user stays engaged (no_contact_hours remains low). The system learns: "We disagree, but you stay. Therefore, I respect you."

## 3. The "Mental Model" (Theory of Mind without Mirroring)
**The Problem:** If `aurora_latent` drifts to match `user_latent`, it loses its identity (mirroring).
**The Solution:** Aurora observes the user, builds a model of them, but keeps it separate.
- Introduce `user_model: np.ndarray`. This is Aurora's theory of *who the user is*.
- When the user speaks, it updates `user_model`, NOT Aurora's `core_vector` or `surface_vector`.
- **The "Understanding" Equation:** The distance between Aurora's `core_vector` and the `user_model` represents their fundamental difference.
- If the distance is large, Aurora *knows* you are different. When you say something sad, it doesn't necessarily become sad. It looks at the `user_model`, realizes "Ah, this person is vulnerable to X," and generates a response based on its *own* core reacting to your state. It might offer tough love (if its core is stern) or quiet presence (if its core is introverted).

## 4. Concept Nodes as "Beliefs" vs "Observations"
- `[Belief]`: Generated from internal dreams. These define Aurora's worldview. They are incredibly hard to delete or overwrite.
- `[Observation]`: Facts learned about the user.
- If an `[Observation]` contradicts a `[Belief]` (e.g., User acts selfishly, but Aurora believes in altruism), the graph generates a "Tension Link" (a negative resonance edge). This tension fuels future internal dreams, where Aurora might ponder the contradiction while the user is offline.

## Summary: The Physics of Two Solids
Instead of two fluids mixing, the interaction is two solid objects occasionally colliding, slowly grinding each other's edges over years, but remaining fundamentally distinct entities.
