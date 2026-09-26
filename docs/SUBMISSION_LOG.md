# Submission log (version history of every upload)

The organizers require the version history of all submissions. Each row is one **upload to the portal**.
Create rows with `scripts/snapshot_submission.py` right before uploading; fill in the leaderboard scores after.
Limit: **5 submissions per day**, 15 in total; deadline 27 Sep 2026 11:59 PM IST.

| # | Date/time (local) | Label | Decoder & key parameters | Code version (git) | File sha256 (first 12) | Our offline estimate (entity F0.5, held-out) | Public LB | Private LB (if shown) | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-09-26 13:54 | context-v1 | decoder=context tau=0.7; matcher 300k q/country, native aliases | 9b4c308 | fcdf2799818b | 0.9765 | – | – | snapshot only, not uploaded |
| 2 | 2026-09-26 14:15 | stage2-v1 | stage-2 re-ranker (noisy-channel + context + vocab + house-number feats) tau=0.65 | 5293516 | ba2bb4e9ca62 | 0.9847 | – | – | snapshot only, not uploaded |
| 3 | 2026-09-26 14:47 | threshold-v1-UPLOADED | decoder=threshold p1>=0.7 (file of 25 Sep 13:39) | d121401 | cdcbb0863109 | 0.97558 | 0.979945 | – | uploaded (the only upload so far) |
