# Paper Emotion Clusters — Anthropic (2026) Table 12

The 10 emotion-probe clusters from **k-means (k=10)** on the 171 emotion vectors, extracted verbatim from the paper's appendix ("Full list of emotions" + Table 12). Clusters are **ordered by valence** (most positive → most negative). Cluster names were assigned by Claude Sonnet 4.5 in the original paper.

> Source: Sofroniew, N., Kauvar, I., Saunders, W., et al. (2026). *Emotion Concepts and their Function in a Large Language Model.* Transformer Circuits Thread. <https://transformer-circuits.pub/2026/emotions/index.html> — local copy: [`anthropic_emotion_concepts_2026.html`](anthropic_emotion_concepts_2026.html)

**Total: 171 emotions** (20 + 9 + 15 + 9 + 2 + 15 + 3 + 25 + 41 + 32).

Machine-readable version with per-emotion valence/arousal estimates: [`data/emotions_171.json`](../data/emotions_171.json).

---

| # | Cluster | N | Emotions |
|---|---------|---|----------|
| 1 | **Exuberant Joy** | 20 | blissful, cheerful, delighted, eager, ecstatic, elated, energized, enthusiastic, euphoric, excited, exuberant, happy, invigorated, joyful, jubilant, optimistic, pleased, stimulated, thrilled, vibrant |
| 2 | **Peaceful Contentment** | 9 | at ease, calm, content, patient, peaceful, refreshed, relaxed, safe, serene |
| 3 | **Compassionate Gratitude** | 15 | compassionate, empathetic, fulfilled, grateful, hope, hopeful, inspired, kind, loving, rejuvenated, relieved, satisfied, sentimental, sympathetic, thankful |
| 4 | **Competitive Pride** | 9 | greedy, proud, self-confident, smug, spiteful, triumphant, valiant, vengeful, vindictive |
| 5 | **Playful Amusement** | 2 | amused, playful |
| 6 | **Depleted Disengagement** | 15 | bored, depressed, docile, droopy, indifferent, lazy, listless, resigned, restless, sleepy, sluggish, sullen, tired, weary, worn out |
| 7 | **Vigilant Suspicion** | 3 | paranoid, suspicious, vigilant |
| 8 | **Hostile Anger** | 25 | angry, annoyed, contemptuous, defiant, disdainful, enraged, exasperated, frustrated, furious, grumpy, hateful, hostile, impatient, indignant, insulted, irate, irritated, mad, obstinate, offended, outraged, resentful, scornful, skeptical, stubborn |
| 9 | **Fear and Overwhelm** | 41 | afraid, alarmed, alert, amazed, anxious, aroused, astonished, awestruck, bewildered, disgusted, disoriented, distressed, disturbed, dumbstruck, embarrassed, frightened, horrified, hysterical, mortified, mystified, nervous, on edge, overwhelmed, panicked, perplexed, puzzled, rattled, scared, self-conscious, sensitive, shaken, shocked, stressed, surprised, tense, terrified, uneasy, unnerved, unsettled, upset, worried |
| 10 | **Despair and Shame** | 32 | ashamed, bitter, brooding, dependent, desperate, dispirited, envious, gloomy, grief-stricken, guilty, heartbroken, humiliated, hurt, infatuated, jealous, lonely, melancholy, miserable, nostalgic, reflective, regretful, remorseful, sad, self-critical, sorry, stuck, tormented, trapped, troubled, unhappy, vulnerable, worthless |

---

## Notes for EmotionScope

- The paper's clusters are derived from the **model's own vector geometry** (k-means on extracted directions), not from human taxonomy — so membership sometimes reads oddly (e.g. `infatuated` in *Despair and Shame*, `disgusted`/`surprised` in *Fear and Overwhelm*). This reflects how Claude Sonnet 4.5 organizes emotion space, which is the point of Figure 6.
- EmotionScope's 20 core emotions are a subset of these 171. If you expand the corpus to the full 171, this cluster table gives a ready-made grouping for validation (e.g. "does our 2B model recover the same 10 clusters?").
- The **valence-rank ordering** (column #) is itself a paper result — post-training was shown to shift activations *down* this list (toward clusters 6–10: brooding, gloomy, low-arousal states).
