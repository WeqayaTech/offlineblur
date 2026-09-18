"""OfflineBlur v2 — transformer-driven spatio-temporal pipeline (stp).

Phase 1  backbone.py       giant ViT feature extraction on the full frame (DINOv2-g / EVA-02-g), tiled, dense stride
Phase 2  tracker.py        end-to-end transformer tracking with track queries (MeMOTR / MOTRv2)
Phase 3  roi_attention.py  cross-attention ROI demographic head on the shared feature map (gender + age + quality)
         demographics.py   runs phase 1 + 3 per frame for every live track
Phase 4  aggregator.py     Bayesian temporal pooling → one locked profile per track id
         render.py         debug video, MOT text, summary
"""
