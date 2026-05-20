# WP Balance v3.5

Status: heuristic prototype

This is an independent WP Balance indicator. It is not official Win Probability.

## Main Upgrade From v3.0

v3.0 started close to 50% and moved mostly from in-game events.

v3.5 starts from a context-aware initial WP:

- Net Rating difference
- Offensive Rating difference
- Defensive Rating difference
- Home court
- Championship Series / CS expectation
- Recent form
- Matchup edge

Then it updates in real time with:

- score margin
- time remaining
- possession
- foul / FT pressure
- timeout / clutch context

## Version Switching

Use:

```bash
--version v3.0
--version v3.5
```

The code supports both versions without overwriting the old v3.0 behavior.

## Interpretation

`team_A_wp` and `team_B_wp` are WP Balance values from 0 to 100.

- 50 = neutral / even
- above 50 = team_A advantage
- below 50 = team_B advantage

For Mikawa usage, `team_A` defaults to `三河`.
