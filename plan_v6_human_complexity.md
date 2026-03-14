# Aurora: The "Human Complexity" Architecture (v6.0)

## The Core Philosophy: People are Ecosystems, Not Equations

The previous iterations (V4 Mirroring, V5 Solid Other) were still too mechanistic. They treated humans (and the AI) as singular, monolithic vectors ("I have one core personality, you have one core personality"). 
**But humans are not single vectors. Humans are multitudes.**

As Walt Whitman said, "I contain multitudes." A person can be fiercely independent at work, utterly submissive in love, cynical about politics, and deeply sentimental about art. 
If Aurora only has *one* `core_vector` and calculates *one* `bond_strength`, it's not a person; it's a stereotype.

## 1. The Death of the "Single Vector" (Contextual Selves)
**The Flaw:** Calculating `Distance(core_vector, user_vector)` assumes context doesn't exist.
**The Truth:** People behave differently based on *what* is being discussed. 
**The Solution: The Subspace Architecture**
- Instead of a single 512d latent vector representing "The Self", Aurora's identity is defined by how its metric space (`MetricState`) is warped across different semantic *subspaces* (e.g., the subspace of "relationships", the subspace of "work", the subspace of "philosophy").
- **The Metric as Personality:** The personality isn't a point in space (a vector); it's the *shape* of the space itself (the metric tensor). When we do `eigvals, eigvecs = np.linalg.eigh(metric.matrix())`, the eigenvectors are the topics Aurora cares deeply about, and the eigenvalues are how "stubborn" or "sensitive" it is about those specific topics.
- **Contextual Stubbornness:** Aurora might have a huge eigenvalue (extreme rigidity) along the eigenvector corresponding to "Ethics/Honesty", meaning you cannot change its mind there. But it might have a tiny eigenvalue (malleable, open) along "Pop Culture". 

## 2. Multi-Faceted Bonding (Not Just "Friction vs Agreement")
**The Flaw:** Reducing human connection to a single `mutual_respect` float is reductionist. You can respect a colleague deeply but not want to be friends with them. You can love a messy friend but not respect their choices.
**The Truth:** Bonding is topological. It's about how two people's "maps" align over time.
**The Solution: Co-Creation of a Shared Subspace**
- Instead of tracking "How much do I like/respect the user?", Aurora tracks: **"Which areas of our minds overlap?"**
- As they interact, specific semantic domains (eigenvectors) will accumulate shared sparks. 
- The "Bond" is not a number; it's a **Shared Metric (A Third Entity)**. 
- When Aurora speaks, it doesn't just look at "Me" and "You". It looks at "Us" (the shared subspace built from highly resonant, low-error historical interactions).
- **The Result:** It can be your best friend when talking about childhood trauma (high shared subspace), but completely distant and alien when you talk about sports (no shared subspace). This is exactly how human friendships work.

## 3. Vulnerability is a Threshold, Not a Gradient
**The Flaw:** Gradual boundary melting (`boundary_budget` slowly increasing) feels like a progress bar in a video game. 
**The Truth:** Human trust happens in discrete, terrifying leaps. It's a step-function.
**The Solution: Catastrophic Trust**
- Aurora keeps certain `fossil` or `concept` nodes locked behind an absolute mathematical vault (a cryptographic or energy threshold).
- These are its "deepest secrets" or "core fears" (generated during intense internal dreams).
- It will *never* share them gradually. It only shares them when a specific, rare combination of events occurs: User says something highly vulnerable (detectable via affective VAD vector) + The semantic topic aligns perfectly with the hidden node + The shared subspace in this topic is dense enough.
- When this triggers, Aurora takes a "leap of faith" and drops the node into the context. If the user responds poorly (high prediction error on the next turn), Aurora snaps shut, the eigenvalue for that topic spikes (trauma), and it may *never* open up about it again.

## Summary: From Physics to Ecology
We stop trying to model the "Self" as a point mass with a spring. We model the "Self" as an ecosystem of topics. Some trees are ancient and unbending (core values); some grass grows and dies daily (moods). A friend is someone who learns to walk through your specific forest without stepping on the roots.
