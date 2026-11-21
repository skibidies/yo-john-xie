# survivor: a Mech Royale bot

A Python bot for the Builders Mech Royale track. It plays for placement: stay inside the zone, fight from good range with good aim, break line of sight to regenerate when losing, and loot when it's safe.

## Files

| File | What it does |
| --- | --- |
| `Dockerfile` | Builds the bot image: Python 3.13.1 and numpy 2.2.6, both pinned. |
| `bot.py` | The brain: modes, movement, aiming and firing, looting, upgrades. |
| `worldmap.py` | Loads `/arena` at startup (about 0.3 s). Handles line of sight, cover checks and A* pathfinding. |
| `mechfight.py` | The kit's protocol wrapper, unchanged. |

## How it plays

Each decision, the bot picks one mode:

- **zone**: outside the safe circle, or the circle is about to close on it. It boosts toward the next circle.
- **fight**: an enemy is in view or was seen recently. It strafes at a preferred range for its weapons, faces the enemy (rear hits do 1.5× damage), leads gun shots, and fires each weapon only when its aim is on target. It ducks behind nearby cover when an enemy's laser is charging at it, and between its own laser shots.
- **retreat**: HP is low and it is losing. It moves to a spot the enemies can't see, so health regenerates.
- **loot**: nothing threatening. It picks up laser or lightning (laser preferred), collects cubes, contests supply drops, and shoots drones and crates for cubes.
- **hold**: nothing to do. It waits quietly next to a building inside the zone, scanning around.

Upgrade picks favour armor, plating, nanites, and then upgrades for weapons it holds.

## Knobs to tune

The `P = {...}` block at the top of `bot.py` controls retreat thresholds, preferred fighting ranges, boosting, and ducking. Change one value at a time, resubmit, and compare scores.
