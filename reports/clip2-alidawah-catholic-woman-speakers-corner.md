# Clip 2 — Ali Dawah at Speakers Corner, Catholic woman (Dailymotion x95zq22)

640x360, 25 fps, first 240 s of 13.6 min. Daytime, Speakers Corner crowd (~40 distinct people in frame over 4 min),
Ali Dawah (grey jacket) interviewing a woman in a white top; a woman in black with glasses stands at the right edge for
most of the clip; children and many men in the background. Low resolution: most background people are 30–65 px tall.

## Result (run C, final code of the day)

| what | outcome |
|---|---|
| interviewee (woman, white top) | one identity for 227.7 of 240 s (three tracks joined by face links), 22 woman / 2 man votes, blurred in every frame she is visible |
| Ali Dawah | one identity for 229.7 s, Man 8/8, never blurred |
| woman in black at the right edge | blurred, but split into three identities (48 s, 101 s, 25 s) — fragments, not an error |
| hijab-wearing woman with a child (yellow jacket) | blurred (two identities, 1.4 s and 3.0 s) |
| children (boys in red / white) | never blurred (judged Man or no votes) |
| blur frames | 5743 of 6000 |
| false blurs found | none after the fixes below; before them: a man's arm at the frame edge (10.9 s) and an orange blob (2.8 s) were blurred as "uncertain Woman", and a man's two segments (10.9 s) had been absorbed into the woman-in-black's identity |
| escapes found | none confirmed on the six sampled frames; **to check**: a grey-top figure at the far left at 220 s judged Man (may be a woman seen from behind) |
| identities with no gender verdict | 108 of 200 — background people under 80 px tall, whose crops are too small to vote (see below) |

Timing: stage 1 1.5 min (67 fps), stage 2 23 s, stage 3 14.8 min (2153 judge calls), render 0.7 min. ≈ 18 min for 4 min of video.

## What this clip taught (code changes made today because of it)

1. **Box containment is not evidence of a duplicate.** The interviewee stands close to the camera, so background people's boxes fall inside hers; the box rule attached 23 of them (arms, shoulders, far pedestrians) to her identity and blurred them. Now only mask-overlap duplicates attach.
2. **Tiny faces bridge people.** A 0.2 s, 40 px fragment with one detected face linked a man (0.52 face similarity) to the woman in black. Face links now need ≥ 2 face crops on both sides, and a face link across two groups that lean to different genders needs ≥ 0.60 and ≥ 3 faces each. Uncertain leans now count as a class for linking, not only decided ones.
3. **Uncertain blur needs evidence.** A 1:1 vote on an arm-shaped mask was blurred for 10.9 s. Gender-uncertain identities are now blurred only with a winning share ≥ 0.60 and vote weight ≥ 2 (and ≥ 0.5 s); the rest go to review.
4. **The vote-height floor must scale with the frame.** 80 px is a third of the frame height at 360p; it silenced every background person. Changed to max(48 px, 10 % of frame height). Run D (in progress) measures what that does to false blurs.

## Open

- Fragmentation remains the main cosmetic issue (the woman in black = 3 identities). Harmless for the blur, noisy for review.
- 360p sources will always have unjudgeable background people; that is a source-quality limit, not a pipeline one. The editor's review page lists them as "no gender answers".

## Run D — same clip with the relative vote-height floor (48 px at 360p)

- Identities 210: Woman 11, Man 130 (was 78), no verdict 65 (was 108). The floor change let 43 more background people be judged, and none of them became a false woman blur.
- Blurred as Woman: 9 identities, 0 uncertain; all nine rows of the contact sheet are real women (interviewee, the woman in black in five fragments, the headscarf woman, the hijab-wearing woman with a child).
- New defect spotted: track 80 (7.1 s) starts on a boy in a red shirt for a few frames before the tracker moves it onto the woman in black; the vote (6 woman / 2 man) blurs the whole track, so the boy is blurred briefly. The vote split needs 3 votes per side and the boy's part had fewer. Checked the box sequence: there is no jump and no gap — the boy stood in front of the woman and the mask slid from him to her. A geometric split cannot catch this; only per-crop appearance (a re-id embedding discontinuity on the sampled crops) or more crops per segment could. Known limit for now; the boy is blurred for roughly a second.
- Stage 3 took 30.5 min this time for the same 2153 calls (0.85 s/call vs 0.41): the GPU was shared with nothing else, so this is unexplained; watch it.

## Run E — full rerun with the v1.4 code (vote share, handoff, hop split, face size ≥ 24 px)

Identical blur outcome to run D: the same 9 women blurred, 0 uncertain, the same 2 junk identities skipped; 211 identities (Woman 11, Man 130, no verdict 66). No regression from the v1.4 changes on this clip. Face crops dropped from 95 to 75 segments with the 24 px minimum, without losing any woman.
