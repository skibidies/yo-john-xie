"""Mech Royale bot: survive first, fight smart, loot when safe.

Overview of one decision (act):
  1. remember what we see (enemies, loot, crates) and hear
  2. pick a mode: ZONE (get inside the safe circle), RETREAT (break line of sight and regen),
     FIGHT (strafe at a good range), LOOT (weapons, cubes, crates, drones, supply drops), HOLD
  3. move: path-find to the mode's goal, dodge mortar shells, get unstuck
  4. aim and fire every weapon whose shot would actually land
"""
import math
import random
import time

import mechfight as mf
from worldmap import WorldMap

TAU = 2 * math.pi
# Tunable knobs (see README: change these, resubmit, compare scores)
P = {
    "retreat_hp": 0.5,        # retreat below this HP fraction when the enemy is healthier
    "retreat_hp_outnumbered": 0.65,
    "retreat_exit_hp": 0.9,    # come back out once regenerated to this
    "range_gun": 55.0,         # preferred fighting distance with gun only
    "range_laser": 95.0,
    "range_lightning": 28.0,
    "boost_on": 65.0,          # start boost-strafing above this energy
    "duck": True,              # hide from enemy laser charges / between our laser shots
    "seek": True,              # early game: spin-scan and explore for a laser/lightning
    "seek_t": 200.0,
    "hide_multi": True,        # vs 2+ enemies: break line of sight, fight only those who come close
    "hide_close": 45.0,
    "laser_first": True,       # keep hunting for a laser until we have one; grab nearby lasers even mid-fight
    "laser_grab_r": 70.0,
    "two_specials": True,      # with laser or lightning in hand, drop the starter gun for the other one
}
# How much we like each weapon in slot 1 (we always keep the gun in slot 0)
WEAPON_RANK = {"laser": 4.0, "lightning": 3.0, "mortar": -5.0, "gun": 0.0}
# Upgrade preference (higher first). Weapon upgrades only apply to weapons we hold.
UPGRADE_RANK = {
    "armor": 100, "plating": 95, "nanites": 80, "laser_charge": 78, "laser_damage": 76,
    "gun_damage": 70, "gun_rate": 68, "gyros": 62, "lightning_efficiency": 60,
    "lightning_range": 58, "gun_velocity": 55, "servos": 52, "gun_mag": 50, "sensors": 48,
    "reactor": 45, "capacitor": 42, "jets": 40, "lightning_chain": 38, "mortar_reload": 30,
    "mortar_velocity": 28, "mortar_splash": 26, "laser_pierce": 20,
}


def wrap(a):
    return (a + math.pi) % TAU - math.pi


def dist2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def norm2(x, y):
    d = math.hypot(x, y)
    return (x / d, y / d) if d > 1e-6 else (0.0, 0.0)


def ang_between(y1, p1, y2, p2):
    """Angle between two (yaw, pitch) directions."""
    a = (math.cos(p1) * math.cos(y1), math.cos(p1) * math.sin(y1), math.sin(p1))
    b = (math.cos(p2) * math.cos(y2), math.cos(p2) * math.sin(y2), math.sin(p2))
    d = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))
    return math.acos(d)


class Bot:
    name = "survivor"

    def __init__(self, hello):
        self.cfg = hello["config"]
        self.rng = random.Random(hello.get("seed", 1))
        t0 = time.time()
        self.map = WorldMap(hello["map"]["dir"])
        mf.log(f"map loaded in {time.time() - t0:.2f}s")
        c = self.cfg
        self.reach = c["loot"]["pickup_radius"]
        self.cube_reach = c["cubes"].get("pickup_radius", 3.5)
        self.enemies = {}   # id -> info dict (mechs)
        self.npcs = {}
        self.crates = {}    # id -> [pos, hp, t_seen]
        self.cubes = {}     # id -> [pos, t_seen]
        self.pickups = {}   # id -> [kind, pos, t_seen]
        self.bad_items = {}  # id -> time until which we ignore it
        self.threats = []   # (t, bearing, weight) from sounds / damage
        self.last_dmg_t = -99.0
        self.last_deal_t = -99.0
        self.mode = "loot"
        self.retreat_until = -1
        self.strafe_sign = 1
        self.strafe_switch_t = 0
        self.boosting = False
        self.path = []
        self.path_goal = None
        self.path_t = -99
        self.stuck_t = 0.0
        self.unstick_until = -1
        self.unstick_dir = (1.0, 0.0)
        self.last_pos = None
        self.target_id = None
        self.cover_pt = None
        self.cover_t = -99
        self.cover_key = None
        self.enemy_laser_cd = {}
        self.idle_goal = None
        self.idle_t = -99
        self.loot_goal_id = None
        self.loot_fail = {}
        self.scan_phase = self.rng.random() * TAU
        self.laser_hold_t = 0.0
        self.fight_label = ""

    # ------------------------------------------------------------------ helpers
    def zone_now(self, z, t):
        cx, cy = z["center"]
        r = z["radius"]
        nx_, ny_ = z["next_center"]
        nr = z["next_radius"]
        s, e = z["shrink_start"], z["shrink_end"]
        if t <= s:
            return (cx, cy), r
        if t >= e or e <= s:
            return (nx_, ny_), nr
        f = (t - s) / (e - s)
        return (cx + (nx_ - cx) * f, cy + (ny_ - cy) * f), r + (nr - r) * f

    def safe_circle(self, obs, pos):
        """Circle we should be inside soon, and whether getting there is urgent."""
        z, t = obs["zone"], obs["t"]
        (ccx, ccy), cr = self.zone_now(z, t)
        nc, nr = z["next_center"], z["next_radius"]
        speed = self.me["stats"].get("walk_speed", 11.7)
        d_cur = dist2(pos, (ccx, ccy))
        outside_now = d_cur > cr - 3
        margin_n = min(15.0, nr * 0.25)
        need = max(0.0, dist2(pos, nc) - (nr - margin_n))
        travel = need * 1.35 / speed + 6
        time_left = z["shrink_end"] - t
        soon = travel > time_left * 0.85 or t > z["shrink_start"] - 15
        if soon or outside_now:
            return (nc, max(nr - margin_n, 0.0)), (outside_now or travel > time_left * 0.85)
        return ((ccx, ccy), max(cr - 20, 0.0)), False

    def in_circle(self, p, circ, slack=0.0):
        return dist2(p, circ[0]) <= circ[1] + slack

    def eye(self, pos):
        return (pos[0], pos[1], pos[2] + self.me["stats"].get("eye_height", 8))

    def muzzle(self):
        p = self.me["pos"]
        return (p[0], p[1], p[2] + self.me["stats"].get("muzzle_height", 7))

    @staticmethod
    def center_of(a):
        p = a["pos"]
        if a.get("kind") == "npc":
            return (p[0], p[1], p[2] + 1.75)
        s = 1 + 0.05 * (a.get("cubes") or 0)
        return (p[0], p[1], p[2] + 4.5 * s)

    def weapon(self, kind):
        for i, w in enumerate(self.me["weapons"]):
            if w and w["kind"] == kind:
                return i, w
        return None, None

    # ------------------------------------------------------------------ memory
    def remember(self, obs):
        t = obs["t"]
        pos = self.me["pos"]
        for a in obs["mechs"]:
            rec = dict(a)
            rec["t"] = t
            if a["kind"] == "mech":
                old = self.enemies.get(a["id"])
                if old and "hp" not in a:
                    rec["hp"] = old.get("hp")
                self.enemies[a["id"]] = rec
            else:
                self.npcs[a["id"]] = rec
        seen = {a["id"] for a in obs["mechs"]}
        for d in (self.enemies, self.npcs):
            for k in list(d):
                if k not in seen and t - d[k]["t"] > 6:
                    del d[k]
        for c in obs["crates"]:
            self.crates[c["id"]] = [c["pos"], c.get("hp", 300), t]
        for c in obs["cubes"]:
            self.cubes[c["id"]] = [c["pos"], t]
        for p in obs["pickups"]:
            self.pickups[p["id"]] = [p["kind"], p["pos"], t]
        vis_c = {c["id"] for c in obs["crates"]}
        vis_q = {c["id"] for c in obs["cubes"]}
        vis_p = {p["id"] for p in obs["pickups"]}
        eye = self.eye(pos)
        for store, vis, pi in ((self.crates, vis_c, 0), (self.cubes, vis_q, 0), (self.pickups, vis_p, 1)):
            for k in list(store):
                if k in vis:
                    continue
                ip = store[k][pi]
                age = t - store[k][-1]
                if age > 90:
                    del store[k]
                    continue
                if dist2(ip, pos) < 30 and self.map.los(eye, (ip[0], ip[1], ip[2] + 1.0)):
                    del store[k]  # we'd see it from here, so it's gone
        for e in obs["events"]:
            ty = e["type"]
            if ty == "damaged":
                if e.get("weapon") not in ("zone", "npc"):
                    self.last_dmg_t = t
                    if e.get("bearing") is not None:
                        self.threats.append((t, e["bearing"], 3.0, e.get("by")))
            elif ty == "hit":
                self.last_deal_t = t
            elif ty == "take_failed":
                self.bad_items[e["id"]] = t + (60 if e["reason"] in ("have", "gone") else 4)
            elif ty == "took":
                self.pickups.pop(e["id"], None)
        for bm in obs.get("beams", []):
            if bm.get("kind") == "laser" and bm.get("by") is not None:
                self.enemy_laser_cd[bm["by"]] = t - bm.get("age", 0) + self.cfg["weapons"]["laser"].get("cooldown", 1.65)
        for s in obs["sounds"]:
            k = s["kind"]
            if k in ("npc_gun", "crate", "drop", "death", "explosion"):
                continue
            w = {"near": 2.0, "mid": 1.0, "far": 0.4}.get(s["dist"], 0.5)
            self.threats.append((t - s.get("age", 0), s["bearing"], w, None))
        self.threats = [th for th in self.threats if t - th[0] < 4][-30:]

    def threat_bearing(self, t):
        """Best guess of the direction danger is coming from, or None."""
        sx = sy = 0.0
        for (tt, b, w, _) in self.threats:
            k = w * max(0.0, 1 - (t - tt) / 4)
            sx += k * math.cos(b)
            sy += k * math.sin(b)
        if math.hypot(sx, sy) < 0.5:
            return None
        return math.atan2(sy, sx)

    # ------------------------------------------------------------------ targeting
    def visible_enemies(self, obs):
        return [a for a in obs["mechs"] if a["kind"] == "mech"]

    def pick_target(self, obs):
        pos = self.me["pos"]
        mz = self.muzzle()
        best, bs = None, 1e18
        for e in self.enemies.values():
            if obs["t"] - e["t"] > 5:
                continue
            c = self.center_of(e)
            d = math.dist(pos, e["pos"])
            if d > 320:
                continue
            los = self.map.los(mz, c)
            hpf = (e.get("hp") or 1400) / max(1, e.get("max_hp") or 1400)
            s = d + 120 * hpf + (0 if los else 90) + (0 if e["t"] == obs["t"] else 40)
            if e["id"] == self.target_id:
                s -= 35
            if s < bs:
                bs, best = s, e
        self.target_id = best["id"] if best else None
        return best

    def lead_point(self, src, tgt, vel, speed):
        tx, ty, tz = tgt
        p = tgt
        for _ in range(3):
            tt = math.dist(src, p) / speed
            p = (tx + vel[0] * tt, ty + vel[1] * tt, tz + vel[2] * tt * 0.5)
        return p

    @staticmethod
    def aim_to(src, p):
        dx, dy, dz = p[0] - src[0], p[1] - src[1], p[2] - src[2]
        return [math.atan2(dy, dx), math.atan2(dz, math.hypot(dx, dy))]

    def mortar_solution(self, src, tgt, speed, g):
        dx, dy = tgt[0] - src[0], tgt[1] - src[1]
        D = math.hypot(dx, dy)
        h = tgt[2] - src[2]
        if D < 1:
            return None
        k = g * D * D / (2 * speed * speed)
        disc = D * D - 4 * k * (h + k)
        if disc < 0:
            return None
        yaw = math.atan2(dy, dx)
        sols = []
        for sgn in (-1, 1):
            u = (D + sgn * math.sqrt(disc)) / (2 * k)
            sols.append(math.atan(u))
        mnp, mxp = self.me["stats"].get("min_pitch", -0.7), self.me["stats"].get("max_pitch", 1.3)
        out = []
        for pitch in sols:
            if not (mnp <= pitch <= mxp):
                continue
            # check arc clearance
            vh = speed * math.cos(pitch)
            T = D / max(vh, 1e-3)
            ok = True
            prev = src
            for i in range(1, 9):
                tt = T * i / 9
                p = (src[0] + math.cos(yaw) * vh * tt, src[1] + math.sin(yaw) * vh * tt,
                     src[2] + speed * math.sin(pitch) * tt - 0.5 * g * tt * tt)
                if not self.map.los(prev, p, spacing=3.0):
                    ok = False
                    break
                prev = p
            if ok:
                out.append((pitch, T))
        if not out:
            return None
        pitch, T = out[0]
        return yaw, pitch, T

    # ------------------------------------------------------------------ movement
    def plan_move(self, pos, goal, t, urgent=False):
        """Return a unit-ish 2D move vector toward goal using A* + smoothing."""
        if goal is None:
            return (0.0, 0.0)
        d = dist2(pos, goal)
        if d < 2.0:
            return (0.0, 0.0)
        if self.map.walk_clear(pos, goal):
            self.path = []
            return norm2(goal[0] - pos[0], goal[1] - pos[1])
        replan = (self.path_goal is None or dist2(self.path_goal, goal) > 8
                  or t - self.path_t > 3.0 or (not self.path and t - self.path_t > 1.0))
        if replan and time.perf_counter() - self.t_start < 0.015:
            p = self.map.astar(pos, goal, max_expand=2500)
            self.path = p or []
            self.path_goal = goal
            self.path_t = t
        while self.path and dist2(pos, self.path[0]) < 4.0:
            self.path.pop(0)
        if len(self.path) > 1 and self.map.walk_clear(pos, self.path[1]):
            self.path.pop(0)
        if not self.path:
            return norm2(goal[0] - pos[0], goal[1] - pos[1])
        wp = self.path[0]
        return norm2(wp[0] - pos[0], wp[1] - pos[1])

    def safe_dir(self, pos, mv, look=6.0):
        """If the move heads into a wall, try nearby directions."""
        if mv == (0.0, 0.0):
            return mv
        base = math.atan2(mv[1], mv[0])
        mag = math.hypot(*mv)
        for off in (0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6):
            a = base + off
            p = (pos[0] + math.cos(a) * look, pos[1] + math.sin(a) * look)
            if self.map.free_at(*p):
                return (math.cos(a) * mag, math.sin(a) * mag)
        return mv

    def find_cover(self, pos, threats_pos, circ, t, radii=(6, 10, 17, 27, 40)):
        key = radii
        if self.cover_pt is not None and t - self.cover_t < 0.6 and self.cover_key == key:
            return self.cover_pt
        self.cover_key = key
        cands = []
        for r in radii:
            for k in range(16):
                a = k * TAU / 16
                p = (pos[0] + math.cos(a) * r, pos[1] + math.sin(a) * r)
                if not self.map.free_at(*p):
                    continue
                if circ and not self.in_circle(p, circ, slack=5):
                    continue
                cands.append(p)
        best = None
        if cands and threats_pos:
            import numpy as np
            hid = np.ones(len(cands), dtype=bool)
            for tp in threats_pos:
                g = [self.map.surface_at(*c) for c in cands]
                pts_mid = [(c[0], c[1], gz + 4.5) for c, gz in zip(cands, g)]
                pts_top = [(c[0], c[1], gz + 8.5) for c, gz in zip(cands, g)]
                hid &= ~self.map.los_many(tp, pts_mid)
                hid &= ~self.map.los_many(tp, pts_top)
            bs = 1e18
            for i, c in enumerate(cands):
                if not hid[i]:
                    continue
                if not self.map.walk_clear(pos, c):
                    continue
                d = dist2(pos, c)
                # avoid running toward the enemy
                tw = 0.0
                for tp in threats_pos:
                    ex, ey = norm2(tp[0] - pos[0], tp[1] - pos[1])
                    cx, cy = norm2(c[0] - pos[0], c[1] - pos[1])
                    tw = max(tw, ex * cx + ey * cy)
                s = d + 25 * max(0.0, tw)
                if s < bs:
                    bs, best = s, c
        self.cover_pt = best
        self.cover_t = t
        return best

    def explore_goal(self, pos, circ, t):
        g = getattr(self, "exp_goal", None)
        if g is not None and t - self.exp_t < 8 and dist2(pos, g) > 6 and self.in_circle(g, circ, 0):
            return g
        c, r = circ
        best, bs = None, 1e18
        for _ in range(30):
            a = self.rng.random() * TAU
            rr = self.rng.uniform(50, 110)
            p = (pos[0] + math.cos(a) * rr, pos[1] + math.sin(a) * rr)
            if not self.map.free_at(*p) or not self.in_circle(p, circ, -10):
                continue
            s = dist2(p, c) * 0.4 + self.rng.random() * 40
            if s < bs:
                bs, best = s, p
        self.exp_goal, self.exp_t = best, t
        return best

    def idle_spot(self, pos, circ, t):
        """A quiet spot near a building inside the safe circle."""
        if self.idle_goal and t - self.idle_t < 12 and self.in_circle(self.idle_goal, circ, 0):
            return self.idle_goal
        if self.idle_goal and t - self.idle_t < 1.5:
            return self.idle_goal
        c, r = circ
        best, bs = None, 1e18
        for _ in range(40):
            a = self.rng.random() * TAU
            rr = r * 0.6 * math.sqrt(self.rng.random())
            p = (c[0] + math.cos(a) * rr, c[1] + math.sin(a) * rr)
            if not self.map.free_at(*p):
                continue
            near_b = any(not self.map.free_at(p[0] + math.cos(b) * 9, p[1] + math.sin(b) * 9)
                         for b in (0, 1.57, 3.14, 4.71))
            s = dist2(p, pos) * 0.5 + dist2(p, c) * 0.3 + (0 if near_b else 40)
            if s < bs:
                bs, best = s, p
        if best is None:
            best = tuple(c)
        self.idle_goal = best
        self.idle_t = t
        return best

    # ------------------------------------------------------------------ main
    def act(self, obs):
        self.t_start = time.perf_counter()
        self.me = me = obs["self"]
        t = obs["t"]
        pos = me["pos"]
        act = {"tick": obs["tick"]}
        self.remember(obs)

        hp_frac = me["hp"] / max(1.0, me["max_hp"])
        circ, zone_urgent = self.safe_circle(obs, pos)
        vis = self.visible_enemies(obs)
        target = self.pick_target(obs)
        recent_enemy = [e for e in self.enemies.values() if t - e["t"] < 5 and math.dist(e["pos"], pos) < 260]
        hurt_recently = t - self.last_dmg_t < 2.5

        # -------- upgrade offer
        if obs.get("offer"):
            held = {w["kind"] for w in me["weapons"] if w}
            defs = {u["id"]: u for u in self.cfg.get("upgrades", [])}
            best_i, best_s = 0, -1e9
            for i, ch in enumerate(obs["offer"]["choices"]):
                req = defs.get(ch["id"], {}).get("requires")
                s = UPGRADE_RANK.get(ch["id"], 10)
                if req and req not in held:
                    s -= 100
                if s > best_s:
                    best_s, best_i = s, i
            act["pick"] = best_i

        # -------- choose mode
        n_threat = len(recent_enemy)
        losing = False
        if target is not None:
            ehp = (target.get("hp") or 1400) / max(1, target.get("max_hp") or 1400)
            losing = hp_frac < ehp - 0.1
        if self.mode == "retreat":
            if hp_frac > P["retreat_exit_hp"] or not recent_enemy or t > self.retreat_until:
                self.mode = "fight"
        else:
            if recent_enemy and ((hp_frac < P["retreat_hp"] and losing) or (hp_frac < P["retreat_hp_outnumbered"] and n_threat >= 2) or hp_frac < 0.2):
                self.mode = "retreat"
                self.retreat_until = t + 14
        mode = self.mode
        if mode != "retreat":
            mode = "fight" if (recent_enemy or hurt_recently) else "loot"
        if zone_urgent:
            mode = "zone" if mode != "retreat" else "retreat"
        self.mode = mode if mode == "retreat" else self.mode if self.mode == "retreat" else mode

        move = (0.0, 0.0)
        boost = False
        label = mode
        goal = None

        # -------- movement by mode
        if mode == "zone" or (zone_urgent and mode == "retreat"):
            goal = circ[0]
            move = self.plan_move(pos, goal, t, urgent=True)
            outside = not me["in_zone"]
            boost = outside or me["energy"] > 50
            label = "zone"
        elif mode == "retreat":
            tps = [(e["pos"][0], e["pos"][1], e["pos"][2] + 8) for e in recent_enemy]
            cov = self.find_cover(pos, tps, circ, t)
            if cov is not None:
                goal = cov
                move = self.plan_move(pos, goal, t)
                label = "retreat->cover"
            else:
                # run directly away from the enemies' mean, toward the safe circle center
                ax = sum(tp[0] for tp in tps) / len(tps)
                ay = sum(tp[1] for tp in tps) / len(tps)
                aw = norm2(pos[0] - ax, pos[1] - ay)
                tc = norm2(circ[0][0] - pos[0], circ[0][1] - pos[1])
                move = norm2(aw[0] + 0.5 * tc[0], aw[1] + 0.5 * tc[1])
                label = "retreat->away"
            boost = me["energy"] > 15
        elif mode == "fight" and target is not None:
            lg = self.laser_grab(pos, circ, t) if P["laser_first"] and not self.has_laser() and hp_frac > 0.35 else None
            if lg is not None:
                # a laser is close: get it now, shooting as we go
                move = self.plan_move(pos, lg, t)
                boost = me["energy"] > 20
                label = f"fight {target['id']} grab-laser"
            else:
                move, boost = self.fight_move(obs, target, circ, t)
                label = f"fight {target['id']} {self.fight_label}"
        elif mode == "fight":
            # hurt by something we can't see: move to cover, face the bearing
            tb = self.threat_bearing(t)
            if tb is not None:
                tp = (pos[0] + math.cos(tb) * 80, pos[1] + math.sin(tb) * 80, pos[2] + 8)
                cov = self.find_cover(pos, [tp], circ, t)
                if cov:
                    goal = cov
                    move = self.plan_move(pos, goal, t)
            label = "under fire"
        else:
            goal, why = self.loot_goal(obs, circ, hp_frac)
            if goal is None and P["seek"] and not self.has_laser() and t < P["seek_t"]:
                goal = self.explore_goal(pos, circ, t)
                why = "explore"
            if goal is None:
                goal = self.idle_spot(pos, circ, t)
                why = "hold"
            move = self.plan_move(pos, goal, t)
            if why == "hold" and dist2(pos, goal) < 12:
                move = (move[0] * 0.35, move[1] * 0.35)  # quiet: below step-sound speed
            if why.startswith("roof"):
                gz = goal[2] if len(goal) > 2 else pos[2]
                if gz > pos[2] + 1.0 and dist2(pos, goal) < 14:
                    act["jump"] = me["grounded"]
                    boost = True
            label = why

        # -------- dodge mortar shells
        dv = self.dodge_shells(obs, pos)
        if dv is not None:
            move = norm2(move[0] * 0.3 + dv[0], move[1] * 0.3 + dv[1])
            boost = True

        # -------- keep out of zone damage when strafing/looting
        if mode in ("fight", "loot") and not self.in_circle(pos, circ, -2):
            tc = norm2(circ[0][0] - pos[0], circ[0][1] - pos[1])
            move = norm2(move[0] + tc[0] * 0.8, move[1] + tc[1] * 0.8)

        move = self.safe_dir(pos, move)

        # -------- stuck handling: jump and jet
        spd = math.hypot(me["vel"][0], me["vel"][1])
        if math.hypot(*move) > 0.5 and spd < 2.0 and me["grounded"]:
            self.stuck_t += 0.1
        else:
            self.stuck_t = max(0.0, self.stuck_t - 0.05)
        if self.stuck_t > 0.7:
            self.stuck_t = 0
            self.unstick_until = t + 1.3
            a = math.atan2(move[1], move[0]) + self.rng.uniform(-0.8, 0.8)
            self.unstick_dir = (math.cos(a), math.sin(a))
            self.path = []
        if t < self.unstick_until:
            move = self.unstick_dir
            if me["grounded"]:
                act["jump"] = True
            boost = me["energy"] > 10

        if me["energy"] < 8:
            boost = False
        act["move"] = [round(move[0], 3), round(move[1], 3)]
        act["boost"] = bool(boost)

        # -------- aim & fire
        self.aim_and_fire(obs, act, target, mode, move)

        # -------- take weapons / cubes handled by walking; take explicitly
        self.try_take(obs, act)

        mf.draw.label(f"{label} hp{me['hp']:.0f} e{me['energy']:.0f}")
        if goal is not None:
            mf.draw.line([pos[0], pos[1], pos[2] + 2], [goal[0], goal[1], pos[2] + 2], "#40c0ff")
        return act

    # ------------------------------------------------------------------ fight movement
    def pref_range(self, target):
        held = {w["kind"] for w in self.me["weapons"] if w}
        ew = set(target.get("weapons") or [])
        if "lightning" in held:
            r = P["range_lightning"]
        elif "laser" in held:
            r = P["range_laser"]
        else:
            r = P["range_gun"]
        if "lightning" in ew and "lightning" not in held:
            r = max(r, 60.0)
        if "mortar" in ew and "lightning" not in held:
            r = min(r, 40.0)
        return r

    def fight_move(self, obs, target, circ, t):
        me = self.me
        pos = me["pos"]
        tp = target["pos"]
        d = dist2(pos, tp)
        to = norm2(tp[0] - pos[0], tp[1] - pos[1])
        perp = (-to[1] * self.strafe_sign, to[0] * self.strafe_sign)
        if t > self.strafe_switch_t:
            self.strafe_sign = -self.strafe_sign
            self.strafe_switch_t = t + self.rng.uniform(0.45, 1.2)
        pref = self.pref_range(target)
        los = self.map.los(self.muzzle(), self.center_of(target))
        held = {w["kind"] for w in me["weapons"] if w}
        ew = set(target.get("weapons") or [])
        teye = (tp[0], tp[1], tp[2] + 8)
        my_laser_i, my_laser = self.weapon("laser")
        my_charging = my_laser is not None and my_laser.get("charge", 0) > 0
        bearing_to_me = math.atan2(pos[1] - tp[1], pos[0] - tp[0])
        aimed = abs(wrap((target.get("aim") or [9, 0])[0] - bearing_to_me)) < 0.15
        duck = los and target.get("charging") and aimed and not my_charging and d > 45
        # peek-shooting: after our laser fires, hide during its cooldown if they have a laser too
        if my_laser is not None and not my_charging and my_laser.get("cooldown", 0) > 0.6 \
                and "laser" in ew and los and d > 45:
            duck = True
        # gun-only vs a laser at range: don't trade, wait at a corner for them to come close
        ambush = ("laser" in ew and not (held & {"laser", "lightning"}) and d > 60)
        others = [e for e in self.enemies.values() if e["id"] != target["id"] and t - e["t"] < 3
                  and dist2(pos, e["pos"]) < 230]
        if P["hide_multi"] and others and d > P["hide_close"] and not my_charging:
            tps = [(e["pos"][0], e["pos"][1], e["pos"][2] + 8) for e in others + [target]]
            cov = self.find_cover(pos, tps, circ, t, radii=(5, 9, 14, 20, 28, 38, 50, 65))
            if cov is not None:
                self.fight_label = "hide"
                return self.plan_move(pos, cov, t), me["energy"] > 20
        if (duck or ambush) and P["duck"] and t - target["t"] < 1.0:
            cov = self.find_cover(pos, [teye], circ, t, radii=(5, 8, 12) if duck else (6, 10, 17, 27))
            if cov is not None:
                self.fight_label = "duck" if duck else "ambush"
                return self.plan_move(pos, cov, t), me["energy"] > 20
        self.fight_label = ""
        if "lightning" in held and "laser" in ew and d > 40:
            pref = 25.0  # rush: lightning shorts their laser
        if not los:
            # find them: go toward their last position (but not too eagerly if far)
            mv = self.plan_move(pos, (tp[0], tp[1]), t)
            return mv, me["energy"] > 60
        radial = max(-1.0, min(1.0, (d - pref) / 25.0))
        mv = (to[0] * radial + perp[0], to[1] * radial + perp[1])
        mv = norm2(*mv)
        # check strafe path; flip if it runs into a wall
        ahead = (pos[0] + mv[0] * 7, pos[1] + mv[1] * 7)
        if not self.map.free_at(*ahead):
            self.strafe_sign = -self.strafe_sign
            self.strafe_switch_t = t + self.rng.uniform(0.5, 1.0)
            perp = (-perp[0], -perp[1])
            mv = norm2(to[0] * radial + perp[0], to[1] * radial + perp[1])
        # boost in bursts to be hard to hit, but keep energy for lightning
        li, _ = self.weapon("lightning")
        floor = 35 if li is not None else 15
        if self.boosting and me["energy"] < floor:
            self.boosting = False
        elif not self.boosting and me["energy"] > P["boost_on"]:
            self.boosting = True
        _, laser = self.weapon("laser")
        if laser and laser.get("charge", 0) > 0:
            self.boosting = False
        return mv, self.boosting

    def dodge_shells(self, obs, pos):
        g = self.cfg["weapons"]["mortar"].get("gravity", 22)
        sr = self.cfg["weapons"]["mortar"].get("splash_radius", 7) + 4
        for pr in obs["projectiles"]:
            if pr["kind"] != "mortar":
                continue
            x, y, z = pr["pos"]
            vx, vy, vz = pr["vel"]
            for i in range(1, 80):
                tt = i * 0.05
                px, py, pz = x + vx * tt, y + vy * tt, z + vz * tt - 0.5 * g * tt * tt
                if pz <= self.map.surface_at(px, py):
                    if dist2((px, py), pos) < sr:
                        return norm2(pos[0] - px, pos[1] - py) if dist2((px, py), pos) > 0.5 else (1.0, 0.0)
                    break
        return None

    # ------------------------------------------------------------------ loot
    def loot_goal(self, obs, circ, hp_frac):
        me = self.me
        pos = me["pos"]
        t = obs["t"]
        best, bs, why = None, 1e18, ""
        slot1 = me["weapons"][1]
        cur_rank = WEAPON_RANK.get(slot1["kind"], 0) if slot1 else -1
        held = {w["kind"] for w in me["weapons"] if w}
        energy = me["energy"]

        def reachable(p):
            dz = p[2] - pos[2]
            if dz > 3.0:
                return dz < 40 and energy > 70, True
            return True, False

        for pid, (kind, p, ts) in self.pickups.items():
            if self.bad_items.get(pid, -1) > t or kind in held:
                continue
            gain = WEAPON_RANK.get(kind, 0) - cur_rank
            if self.pair_slot(kind) is not None:
                gain = 1.5
            if gain <= 0.4:
                continue
            if not self.in_circle(p, circ, -5):
                continue
            ok, roof = reachable(p)
            if not ok:
                continue
            s = dist2(pos, p) - 60 * gain - (60 if slot1 is None else 0) + (25 if roof else 0)
            if s < bs:
                bs, best, why = s, p, ("roof weapon" if roof else f"get {kind}")
        for cid, (p, ts) in self.cubes.items():
            if self.bad_items.get(cid, -1) > t or not self.in_circle(p, circ, -3):
                continue
            ok, roof = reachable(p)
            if not ok:
                continue
            s = dist2(pos, p) - 70 + (25 if roof else 0)
            if s < bs:
                bs, best, why = s, p, ("roof cube" if roof else "cube")
        # supply drop
        for dr in obs.get("drops", []):
            p = dr["pos"]
            if not self.in_circle(p, circ, -2):
                continue
            d = dist2(pos, p)
            if d > 220 or hp_frac < 0.3:
                continue
            if dr.get("holder") is not None and dr.get("holder") != me["id"]:
                continue
            travel = d / 12
            if dr["lands_in"] > travel + 6:
                continue
            s = d - 90
            if s < bs:
                bs, best, why = s, (p[0], p[1], p[2]), "drop"
        if best is not None:
            return best, why
        # shootable things: crates / drones -> stand within range and let aim_and_fire do it
        return None, ""

    def pair_slot(self, kind):
        """Slot of the starter gun if taking `kind` would give us both laser and lightning."""
        if not P["two_specials"] or kind not in ("laser", "lightning"):
            return None
        held = {w["kind"] for w in self.me["weapons"] if w}
        other = "lightning" if kind == "laser" else "laser"
        if other in held and kind not in held:
            gi, _ = self.weapon("gun")
            return gi
        return None

    def has_laser(self):
        return any(w and w["kind"] == "laser" for w in self.me["weapons"])

    def laser_grab(self, pos, circ, t):
        best, bd = None, P["laser_grab_r"]
        for pid, (kind, p, ts) in self.pickups.items():
            if kind != "laser" or self.bad_items.get(pid, -1) > t or abs(p[2] - pos[2]) > 3.0:
                continue
            if not self.in_circle(p, circ, -3):
                continue
            d = dist2(pos, p)
            if d < bd:
                bd, best = d, p
        return best

    def try_take(self, obs, act):
        me = self.me
        pos = me["pos"]
        t = obs["t"]
        slot1 = me["weapons"][1]
        held = {w["kind"] for w in me["weapons"] if w}
        cur_rank = WEAPON_RANK.get(slot1["kind"], 0) if slot1 else -1
        best = None
        for p in obs["pickups"]:
            if p["kind"] in held or self.bad_items.get(p["id"], -1) > t:
                continue
            if dist2(p["pos"], pos) > self.reach - 0.3 or abs(p["pos"][2] - pos[2]) > 3.8:
                continue
            g = WEAPON_RANK.get(p["kind"], 0) - cur_rank
            if g > 0.4 and (best is None or g > best[0]):
                best = (g, p)
        pair = None
        for p in obs["pickups"]:
            if self.bad_items.get(p["id"], -1) > t:
                continue
            if dist2(p["pos"], pos) > self.reach - 0.3 or abs(p["pos"][2] - pos[2]) > 3.8:
                continue
            gs = self.pair_slot(p["kind"])
            if gs is not None:
                pair = (gs, p)
                break
        if pair is not None:
            act["drop"] = pair[0]  # drop the starter gun, keep both specials
            act["take"] = pair[1]["id"]
            return
        if best:
            if slot1 is not None:
                if slot1["kind"] == "laser" and slot1.get("charge", 0) > 0:
                    return
                act["drop"] = 1
            act["take"] = best[1]["id"]

    # ------------------------------------------------------------------ aim & fire
    def aim_and_fire(self, obs, act, target, mode, move):
        me = self.me
        pos = me["pos"]
        t = obs["t"]
        mz = self.muzzle()
        fire = [False, False]
        aim = None
        ay, ap = me["aim"]

        shoot_obj = None   # (center, vel, radius, kind)
        if target is not None and t - target["t"] < 0.15:
            shoot_obj = (self.center_of(target), target.get("vel", [0, 0, 0]),
                         2.5 * (1 + 0.05 * (target.get("cubes") or 0)), "mech", target)
        elif mode != "retreat":
            # drones and crates when nothing better to do
            shoot_obj = self.side_target(obs)

        if shoot_obj is not None:
            c, vel, rad, kind, obj = shoot_obj
            d = math.dist(mz, c)
            los = self.map.los(mz, c)
            gi, gun = self.weapon("gun")
            li, laser = self.weapon("laser")
            ki, light = self.weapon("lightning")
            oi, mort = self.weapon("mortar")
            tol = math.atan2(rad * 0.75, max(d, 1)) + 0.004

            laser_busy = laser is not None and laser.get("charge", 0) > 0
            # 1. laser: aim straight at the target (hitscan)
            if laser and los and d < laser["stats"].get("range", 280) * 0.95 and kind == "mech":
                p = (c[0] + vel[0] * 0.1, c[1] + vel[1] * 0.1, c[2])
                aim = self.aim_to(mz, p)
                full = laser["stats"].get("charge", 0.75)
                ch = laser.get("charge", 0)
                if laser["cooldown"] <= 0.02 or ch > 0:
                    if ch >= full * 0.97 and ang_between(ay, ap, aim[0], aim[1]) < tol:
                        fire[li] = False  # release -> fires
                    else:
                        fire[li] = True
            elif laser_busy:
                # charging but lost the shot: keep holding, aim at last known
                fire[li] = True
                aim = self.aim_to(mz, c)
            # 2. gun (lead the target) unless the laser owns the aim
            if gun and los and d < gun["stats"].get("range", 230):
                spd = gun["stats"].get("speed", 252)
                p = self.lead_point(mz, c, vel, spd)
                gaim = self.aim_to(mz, p)
                if aim is None:
                    aim = gaim
                if gun.get("reload", 0) <= 0 and gun.get("ammo", 1) > 0:
                    if ang_between(ay, ap, gaim[0], gaim[1]) < tol + 0.006:
                        fire[gi] = True
            # 3. lightning (cone, auto-target)
            if light:
                rng_ = light["stats"].get("range", 45)
                cone = math.radians(light["stats"].get("cone_deg", 30)) / 2
                if aim is None:
                    aim = self.aim_to(mz, c)
                yaw_to = math.atan2(c[1] - pos[1], c[0] - pos[0])
                if d < rng_ * 0.95 and los and abs(wrap(yaw_to - ay)) < cone * 0.8 \
                        and me["energy"] > light["stats"].get("energy", 3) + 2:
                    fire[ki] = True
            # 4. mortar: lob at targets out of sight or far away
            if mort and kind in ("mech", "npc") and mort["cooldown"] <= 0.05:
                if d > 22 and (not los or aim is None or (gun and d > 120)):
                    st = mort["stats"]
                    src = (pos[0], pos[1], pos[2] + me["stats"].get("height", 9) + 0.5)
                    tgt = (c[0], c[1], c[2] - (4.5 if kind == "mech" else 1.75))
                    sol = self.mortar_solution(src, tgt, st.get("speed", 72), st.get("gravity", 22))
                    if sol:
                        yaw, pitch, T = sol
                        tgt2 = (tgt[0] + vel[0] * T, tgt[1] + vel[1] * T, tgt[2])
                        sol2 = self.mortar_solution(src, tgt2, st.get("speed", 72), st.get("gravity", 22))
                        if sol2:
                            yaw, pitch, _ = sol2
                        if not (aim and (fire[gi if gi is not None else 0] or (laser and fire[li]))):
                            aim = [yaw, pitch]
                            if ang_between(ay, ap, yaw, pitch) < 0.03:
                                fire[oi] = True

        if aim is None:
            # no shot: face the threat or scan around
            tb = self.threat_bearing(t)
            if target is not None:
                aim = self.aim_to(mz, self.center_of(target))
            elif tb is not None:
                aim = [tb, 0.0]
            elif P["seek"] and not self.has_laser() and t < P["seek_t"]:
                aim = [wrap(self.scan_phase + t * 3.2), -0.05]  # spin to spot weapons
            else:
                mv = math.atan2(move[1], move[0]) if math.hypot(*move) > 0.1 else ay
                aim = [wrap(mv + 1.9 * math.sin(t * 1.3 + self.scan_phase)), 0.0]
            # reload while nothing to shoot
            gi, gun = self.weapon("gun")
            if gun and gun.get("reload", 0) <= 0 and gun.get("ammo", 12) < gun["stats"].get("mag", 12) * 0.6:
                act["reload"] = True
            _, laser = self.weapon("laser")
            li, _ = self.weapon("laser")
            if laser and laser.get("charge", 0) > 0:
                fire[li] = True  # keep holding until a target shows (auto-fires after hold_s)
        mnp, mxp = me["stats"].get("min_pitch", -0.7), me["stats"].get("max_pitch", 1.3)
        act["aim"] = [round(wrap(aim[0]), 4), round(max(mnp, min(mxp, aim[1])), 4)]
        act["fire"] = fire

    def side_target(self, obs):
        """Shoot drones and crates for cubes when calm."""
        me = self.me
        pos = me["pos"]
        mz = self.muzzle()
        hp_frac = me["hp"] / max(1, me["max_hp"])
        best, bs = None, 1e18
        for a in obs["mechs"]:
            if a["kind"] != "npc" or hp_frac < 0.45:
                continue
            c = self.center_of(a)
            d = math.dist(mz, c)
            if d > 140 or not self.map.los(mz, c):
                continue
            s = d - 20
            if s < bs:
                bs, best = s, (c, a.get("vel", [0, 0, 0]), 1.8, "npc", a)
        for cr in obs["crates"]:
            p = cr["pos"]
            c = (p[0], p[1], p[2] + 1.2)
            d = math.dist(mz, c)
            if d > 110 or not self.map.los(mz, c):
                continue
            if bs > d:
                bs, best = d, (c, [0, 0, 0], 1.4, "crate", cr)
        return best


if __name__ == "__main__":
    mf.run(Bot)
