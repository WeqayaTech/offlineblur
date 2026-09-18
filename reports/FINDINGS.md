# OfflineBlur — running findings log

One entry per test clip, newest last. Each entry links to its own report in this folder. The
question every entry answers: on this footage, what did the pipeline blur wrongly (false blur),
what did it miss (escape), and what changed in the code because of it.

| date | clip | source | result | report |
|---|---|---|---|---|
| 2026-09-17 | ali-dawah-street-interview-source (12 s, 720p, night street, ~25 people) | owner-supplied | v1.0→v1.3 iterations; 12 women blurred, 0 false blurs on men after v1.3, 1 known escape (partial view behind the reporter) | [ali-dawah-street-interview.md](ali-dawah-street-interview.md) |
| 2026-09-17 | dm_alidawah_catholic_woman (240 s of 13.6 min, 360p, Speakers Corner daytime crowd) | Dailymotion x95zq22 | interviewee blurred 227.7/240 s as one identity; Ali Dawah never blurred; 3 false blurs (arm, blob, man merged into a woman) fixed the same day; 108 of 200 identities too small to judge at 360p | [clip2-alidawah-catholic-woman-speakers-corner.md](clip2-alidawah-catholic-woman-speakers-corner.md) |
| 2026-09-17 | dm_dutchman_converts (90 s of 4.2 min, 720p 30 fps, Speakers Corner, busy path) | Dailymotion x9hqpp8 | 45 women blurred, all verified; every man clear; 3 false blurs in run A fixed by v1.4 (vote share, handoff, hop split); 1 escape (blond woman on a 4:4 split, review-flagged) | [clip3-dutchman-converts-speakers-corner.md](clip3-dutchman-converts-speakers-corner.md) |

## Test corpus (downloaded 2026-09-17, first 240 s of each processed unless noted)

YouTube blocks the pod's datacenter address, so the corpus comes from Dailymotion and archive.org mirrors of the same channels.

| clip (stem on the pod) | source | res / fps / length | why it is in the set |
|---|---|---|---|
| ia_why_arent_you_muslim_dawahwise | archive.org, Dawah Wise (Hashim, Mansur, Ijaz), Speakers Corner | 1280x720, 24 fps, 13.7 min | daytime Speakers Corner crowd, several speakers |
| ia_shaykh_uthman_australian | archive.org, One Message Foundation (Shaykh Uthman ibn Farooq) | 1280x720, 60 fps, 23.8 min | man in traditional dress (the known Gulf-dress misread risk), 60 fps |
| dm_alidawah_catholic_woman | Dailymotion x95zq22, Ali Dawah at Speakers Corner | 640x360, 25 fps, 13.6 min | woman interviewee, low resolution |
| dm_young_visitors_speakers_corner | Dailymotion x9i50xa, Speakers Corner | 1280x720, 30 fps, 14 min | young visitors — the adult/child boundary |
| dm_dutchman_converts | Dailymotion x9hqpp8, Ali Dawah | 1280x720, 30 fps, 4.2 min | crowd around a conversion, many men close together |
| dm_munadi_boat_basin | Dailymotion x2hzipv, street dawah, Karachi | 512x288, 24 fps, 4 min | very low resolution, non-UK street, different dress |
