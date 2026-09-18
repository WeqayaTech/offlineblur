# Clip 3 — "A Dutchman objects and then converts", Ali Dawah at Speakers Corner (Dailymotion x9hqpp8)

1280x720, 30 fps, first 90 s of 4.2 min, inference at 1920 px. Daytime; Ali Dawah and the Dutchman seated at a table,
a table of men beside them, a busy path behind with pedestrians of both genders crossing constantly (prams, cyclists,
a veiled woman at the gate). The source itself already pixelates one bystander at the left edge.

## Result (run B, code v1.4)

| what | outcome |
|---|---|
| Ali Dawah, the Dutchman, the men at the table and every man in the background | Man, never blurred |
| women crossing behind (blond woman, women with prams, veiled woman at the gate, leopard-coat woman, hijab-wearing women) | 45 identities blurred, 5 of them uncertain; all rows of the contact sheet checked are women |
| frames with blur | 1707 of 2700 (women are in frame 63 % of the time) |
| false blurs, run A | a man's head from behind (0.5 s: one face-weighted "woman" answer outvoted three "man"), a backpack person from behind (4.2 s, uncertain), a man in a black coat joined to the leopard-coat woman by a handoff (0.3 s) |
| false blurs, run B | none of the above remain: the head and the backpack person are Man, the handoff is gone |
| escapes | a blond woman in a blue jacket (track 3, 3.8 s) gets 4 woman / 4 man answers, is uncertain at 0.52 and is therefore not blurred; she is listed first on the review page |
| no gender verdict | 13 of 191 identities (slivers at the frame edge) |
| time | stage 1 1.5 min, stage 2 26 s, stage 3 14.2 min (1951 calls), render 0.9 min |

## What this clip changed (v1.4)

1. **A face-weighted answer cannot outvote plain ones.** The winning share is now the mean of the weighted share and the count share, and a face crop weighs ×2, not ×3.
2. **Handoffs only for segments with no gender lean of their own**, and only on a real box overlap (IoU ≥ 0.3), not containment.
3. **Single-frame hops split a track**: a jump of more than the person's own height between consecutive frames is a tracker hop onto someone else. Note: this does NOT catch the boy → woman case on clip 2 — that track has no jump at all (the boy stood in front of the woman and the mask slid from him to her as he moved), so only appearance or votes could cut it; with 1–2 crops on the boy the vote split cannot fire. Recorded as a known limit.

## Open on this clip

- The blue-jacket blond woman: a judge inconsistency, not a linking problem. A second judge (MiVOLO V2) or a "tie goes to blur when ≥ 4 votes each" policy would decide it; the latter risks false blurs on men in the same situation. Left for review.
- 45 identities for perhaps 25 women in 90 s: every crossing is a fresh track, and body re-id cannot re-join them. Cosmetic.
