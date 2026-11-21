"""Static map model: terrain + building heights on a 1 m grid, line-of-sight, and a coarse nav grid."""
import json
import os
import heapq
import math
import numpy as np


class WorldMap:
    def __init__(self, map_dir):
        w = json.load(open(os.path.join(map_dir, "v2", "world.json")))
        t = w["terrain"]
        self.x0, self.y0, self.step = float(t["minX"]), float(t["minY"]), float(t["step"])
        self.nx, self.ny = int(t["nx"]), int(t["ny"])
        raw = np.fromfile(os.path.join(map_dir, "v2", t.get("file", "terrain.i16")), dtype="<i2")
        self.ground = (raw.reshape(self.ny, self.nx).astype(np.float32) * float(t.get("scale", 0.01)))
        b = w["bounds"]
        self.bx0, self.bx1, self.by0, self.by1 = b["minX"], b["maxX"], b["minY"], b["maxY"]
        # solid_top: top of ground-anchored solids per cell (-inf if none)
        # slab_base/slab_top: elevated solids (bridges, overhangs) per cell
        top = np.full((self.ny, self.nx), -1e9, dtype=np.float32)
        sbase = np.full((self.ny, self.nx), 1e9, dtype=np.float32)
        stop = np.full((self.ny, self.nx), -1e9, dtype=np.float32)
        for p in w["prisms"]:
            mask, (j0, i0) = self._raster(p["ring"], p.get("holes") or [])
            if mask is None:
                continue
            h, wd = mask.shape
            g = self.ground[j0:j0 + h, i0:i0 + wd]
            base, ptop = float(p["base"]), float(p["top"])
            anchored = mask & (base <= g + 1.5)
            elev = mask & ~anchored
            sub = top[j0:j0 + h, i0:i0 + wd]
            np.maximum(sub, np.where(anchored, ptop, -1e9), out=sub)
            sb = sbase[j0:j0 + h, i0:i0 + wd]
            np.minimum(sb, np.where(elev, base, 1e9), out=sb)
            st = stop[j0:j0 + h, i0:i0 + wd]
            np.maximum(st, np.where(elev, ptop, -1e9), out=st)
        self.solid_top = top
        self.slab_base = sbase
        self.slab_top = stop
        # surface height you stand on (ignores slabs: we walk under bridges)
        self.surface = np.maximum(self.ground, top)
        self.building = top > self.ground + 1.2
        self._build_nav()

    # ---------- rasterization ----------
    def _raster(self, ring, holes):
        polys = [ring] + list(holes)
        xs = np.array(ring[0::2]); ys = np.array(ring[1::2])
        i0 = max(0, int(math.floor((xs.min() - self.x0) / self.step)) - 1)
        i1 = min(self.nx - 1, int(math.ceil((xs.max() - self.x0) / self.step)) + 1)
        j0 = max(0, int(math.floor((ys.min() - self.y0) / self.step)) - 1)
        j1 = min(self.ny - 1, int(math.ceil((ys.max() - self.y0) / self.step)) + 1)
        if i1 <= i0 or j1 <= j0:
            return None, (0, 0)
        W, H = i1 - i0 + 2, j1 - j0 + 1
        acc = np.zeros((H, W), dtype=np.int32)
        cy = self.y0 + (np.arange(H) + j0) * self.step
        for poly in polys:
            px = np.array(poly[0::2], dtype=np.float64); py = np.array(poly[1::2], dtype=np.float64)
            qx = np.roll(px, -1); qy = np.roll(py, -1)
            for ax, ay, bx, by in zip(px, py, qx, qy):
                if ay == by:
                    continue
                lo, hi = (ay, by) if ay < by else (by, ay)
                rows = np.nonzero((cy >= lo) & (cy < hi))[0]
                if rows.size == 0:
                    continue
                xc = ax + (cy[rows] - ay) * (bx - ax) / (by - ay)
                col = np.ceil((xc - self.x0) / self.step).astype(np.int64) - i0
                col = np.clip(col, 0, W - 1)
                np.add.at(acc, (rows, col), 1)
        inside = (np.cumsum(acc, axis=1) & 1).astype(bool)[:, :W - 1]
        return inside, (j0, i0)

    # ---------- queries ----------
    def idx(self, x, y):
        i = int((x - self.x0) / self.step + 0.5)
        j = int((y - self.y0) / self.step + 0.5)
        return min(max(j, 0), self.ny - 1), min(max(i, 0), self.nx - 1)

    def ground_at(self, x, y):
        j, i = self.idx(x, y)
        return float(self.ground[j, i])

    def surface_at(self, x, y):
        j, i = self.idx(x, y)
        return float(self.surface[j, i])

    def los(self, a, b, spacing=1.5):
        """True if the straight segment a->b (3D) is not blocked by terrain or solids."""
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        n = max(2, int(d / spacing))
        tt = np.linspace(0.04, 0.96, n)
        return bool(self._clear(a[0] + dx * tt, a[1] + dy * tt, a[2] + dz * tt).all())

    def los_many(self, a, pts, spacing=2.0):
        """Vectorized: a is one 3D point, pts is (k,3). Returns bool array (k,)."""
        pts = np.asarray(pts, dtype=np.float64)
        d = pts - np.asarray(a, dtype=np.float64)
        L = np.sqrt((d * d).sum(1)).max() if len(pts) else 0
        n = max(2, int(L / spacing))
        tt = np.linspace(0.04, 0.96, n)[None, :]
        X = a[0] + d[:, 0:1] * tt
        Y = a[1] + d[:, 1:2] * tt
        Z = a[2] + d[:, 2:3] * tt
        return self._clear(X, Y, Z).all(axis=1)

    def _clear(self, X, Y, Z):
        I = np.clip(((X - self.x0) / self.step + 0.5).astype(np.int64), 0, self.nx - 1)
        J = np.clip(((Y - self.y0) / self.step + 0.5).astype(np.int64), 0, self.ny - 1)
        top = self.solid_top[J, I]
        g = self.ground[J, I]
        ok = (Z > top) & (Z > g - 0.3)
        sb = self.slab_base[J, I]
        st = self.slab_top[J, I]
        ok &= ~((Z > sb) & (Z < st))
        return ok

    # ---------- navigation (coarse grid) ----------
    def _build_nav(self):
        c = 4  # meters per nav cell
        self.nc = c
        H, W = self.ny // c, self.nx // c
        b = self.building[:H * c, :W * c]
        # dilate buildings by ~3 m (mech radius 2.5)
        d = b.copy()
        for s in range(1, 4):
            d[s:, :] |= b[:-s, :]; d[:-s, :] |= b[s:, :]
            d[:, s:] |= b[:, :-s]; d[:, :-s] |= b[:, s:]
        blocked = d.reshape(H, c, W, c).mean(axis=(1, 3)) > 0.3
        g = self.ground[:H * c, :W * c].reshape(H, c, W, c)
        grange = g.max(axis=(1, 3)) - g.min(axis=(1, 3))
        blocked |= grange > 3.2  # very steep
        gm = g.mean(axis=(1, 3))
        # map-edge margin
        cx = self.x0 + (np.arange(W) + 0.5) * c
        cy = self.y0 + (np.arange(H) + 0.5) * c
        blocked |= (cx[None, :] < self.bx0 + 6) | (cx[None, :] > self.bx1 - 6)
        blocked |= (cy[:, None] < self.by0 + 6) | (cy[:, None] > self.by1 - 6)
        self.nav_blocked = blocked
        self.nav_g = gm
        self.nW, self.nH = W, H
        self._nb = blocked.tolist()
        self._ng = gm.tolist()

    def nav_cell(self, x, y):
        i = int((x - self.x0) / self.nc); j = int((y - self.y0) / self.nc)
        return min(max(j, 0), self.nH - 1), min(max(i, 0), self.nW - 1)

    def nav_center(self, j, i):
        return (self.x0 + (i + 0.5) * self.nc, self.y0 + (j + 0.5) * self.nc)

    def nearest_free(self, j, i, rad=6):
        nb = self._nb
        if not nb[j][i]:
            return (j, i)
        best = None
        for r in range(1, rad + 1):
            for dj in range(-r, r + 1):
                for di in (-r, r) if abs(dj) != r else range(-r, r + 1):
                    jj, ii = j + dj, i + di
                    if 0 <= jj < self.nH and 0 <= ii < self.nW and not nb[jj][ii]:
                        dd = dj * dj + di * di
                        if best is None or dd < best[0]:
                            best = (dd, (jj, ii))
            if best:
                return best[1]
        return None

    def astar(self, start, goal, max_expand=4000):
        """start/goal world (x,y). Returns list of world waypoints or None."""
        start = (start[0], start[1]); goal = (goal[0], goal[1])
        s = self.nearest_free(*self.nav_cell(*start))
        g = self.nearest_free(*self.nav_cell(*goal))
        if s is None or g is None:
            return None
        if s == g:
            return [goal]
        nb, ng = self._nb, self._ng
        H, W = self.nH, self.nW
        gj, gi = g
        openh = [(0.0, 0.0, s)]
        came = {s: None}
        cost = {s: 0.0}
        best = (1e18, s)
        nbrs = ((0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),
                (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414))
        n = 0
        found = False
        while openh:
            f, cc, cur = heapq.heappop(openh)
            if cur == g:
                found = True
                break
            n += 1
            if n > max_expand:
                break
            j, i = cur
            hcur = math.hypot(j - gj, i - gi)
            if hcur < best[0]:
                best = (hcur, cur)
            gc = cost[cur]
            if cc > gc + 1e-9:
                continue
            g0 = ng[j][i]
            for dj, di, w in nbrs:
                jj, ii = j + dj, i + di
                if jj < 0 or ii < 0 or jj >= H or ii >= W or nb[jj][ii]:
                    continue
                if dj and di and (nb[j][ii] or nb[jj][i]):
                    continue
                dz = ng[jj][ii] - g0
                if dz > 2.5 * w:
                    continue
                nc_ = gc + w + (0.3 * dz if dz > 0 else 0)
                nxt = (jj, ii)
                if nc_ < cost.get(nxt, 1e18):
                    cost[nxt] = nc_
                    came[nxt] = cur
                    hh = math.hypot(jj - gj, ii - gi)
                    heapq.heappush(openh, (nc_ + 1.3 * hh, nc_, nxt))
        end = g if found else best[1]
        path = []
        c = end
        while c is not None:
            path.append(c)
            c = came[c]
        path.reverse()
        pts = [self.nav_center(j, i) for j, i in path]
        if found:
            pts[-1] = tuple(goal)
        return self._smooth(start, pts)

    def walk_clear(self, a, b):
        """Straight walk a->b (2D) through free nav cells."""
        dx, dy = b[0] - a[0], b[1] - a[1]
        d = math.hypot(dx, dy)
        n = max(1, int(d / 2.0))
        nb = self._nb
        for k in range(1, n + 1):
            t = k / n
            j, i = self.nav_cell(a[0] + dx * t, a[1] + dy * t)
            if nb[j][i]:
                return False
        return True

    def _smooth(self, start, pts):
        out = []
        cur = start
        k = 0
        while k < len(pts):
            far = k
            for m in range(min(len(pts) - 1, k + 12), k, -1):
                if self.walk_clear(cur, pts[m]):
                    far = m
                    break
            out.append(pts[far])
            cur = pts[far]
            k = far + 1
        return out

    def free_at(self, x, y, *_):
        j, i = self.nav_cell(x, y)
        return not self._nb[j][i]
