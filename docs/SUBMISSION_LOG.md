# Submission log (version history of every upload)

The organizers require the version history of all submissions. Each row is one **upload to the portal**.
Create rows with `scripts/snapshot_submission.py` right before uploading; fill in the leaderboard scores after.
Limit: **5 submissions per day**, 15 in total; deadline 27 Sep 2026 11:59 PM IST.

| # | Date/time (local) | Label | Decoder & key parameters | Code version (git) | File sha256 (first 12) | Our offline estimate (entity F0.5, held-out) | Public LB | Private LB (if shown) | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-09-26 13:54 | context-v1 | decoder=context tau=0.7; matcher 300k q/country, native aliases | 9b4c308 | fcdf2799818b | 0.9765 | – | – |  |
| 2 | 2026-09-26 14:15 | stage2-v1 | stage-2 re-ranker (noisy-channel + context + vocab + house-number feats) tau=0.65 | 9b4c308+dirty | ba2bb4e9ca62 | 0.9847 | – | – |  |
