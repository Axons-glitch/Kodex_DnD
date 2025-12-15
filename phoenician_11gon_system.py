# phoenician_11gon_system.py
# 11‑gon geometry with 22 Phoenician symbols (Aleph..Taw).
# INPUT sequence = vertices 1..11 (top is 1, then clockwise 2..11).
#
# Shifts:
#   • S (vertex shift, mod 11)       : which vertex gets the FIRST symbol.
#   • start_shift (alphabet, mod 22) : which alphabet symbol is FIRST (0=Aleph, 1=Beth, ...).
#   => FIRST symbol sits at vertex V0 = (S mod 11) + 1; next symbols proceed clockwise.
#
# Drawing:
#   • draw_cryptic : no labels
#   • draw_named   : Latin names
#   • draw_glyphs  : Unicode Phoenician glyphs (needs explicit font path)
#   • All edges gray; realized path black; start vertex red; center prints start_shift
#
# Costs:
#   • Per-glyph base cost (non-flat) via set_costs({1:.., .., 22:..})
#   • STACKABLE connection multipliers by #lines already drawn:
#       set_connection_multipliers({2:1.5, 4:1.3})
#         -> for the 3rd glyph (2 lines) use ×1.5; for the 5th glyph (4 lines) use ×1.5×1.3=×1.95
#       The k-th glyph uses product of all multipliers whose thresholds t satisfy (k-1) >= t.
#
# Printing:
#   • print_spell_meanings_and_cost(vertices, S, start_shift) uses the SAME mapping as drawing.
#
# Minimal:
#   from phoenician_11gon_system import Phoenician11System
#   font = r'.\\fonts\\NotoSansPhoenician-Regular.ttf'
#   g = Phoenician11System(glyph_font_path=font)
#   seq = [1,7,4,11,9,9,2]  # vertices (1..11), duplicates removed
#   g.set_connection_multipliers({2:1.5, 4:1.5})  # stack after 2 and after 4 lines
#   g.draw_named(seq, S=3, start_shift=2, savepath='named.png')
#   g.draw_glyphs(seq, S=3, start_shift=2, savepath='glyphs.png')
#   g.print_spell_meanings_and_cost(seq, S=3, start_shift=2)
#


TOP_ANGLE_DEG = 90.0
N_VERT = 11
N_ALPHA = 22

PHOENICIAN_NAMES_22 = [
    'Aleph','Beth','Gimel','Daleth','He','Waw','Zayin','Heth','Teth','Yod',
    'Kaph','Lamed','Mem','Nun','Samekh','Ayin','Pe','Tsade','Qoph','Resh','Shin','Taw'
]

def phoenician_glyph(j1: int) -> str:
    if not (1 <= j1 <= N_ALPHA):
        raise ValueError('Phoenician glyph index must be 1..22.')
    return chr(0x10900 + (j1 - 1))

def wrap1(x: int, m: int) -> int:
    return ((int(x) - 1) % m) + 1

class Phoenician11System:
    def __init__(self, clockwise: bool = True, radius: float = 1.0,
                 glyph_font_path: str | None = None):
        self.clockwise = bool(clockwise)
        self.R = float(radius)

        # Optional font for glyphs
        self._glyph_fp = None
        if glyph_font_path is not None:
            fm.fontManager.addfont(glyph_font_path)
            self._glyph_fp = fm.FontProperties(fname=glyph_font_path)

        # meanings and per-glyph base costs (non-flat default example values)
        self.meanings = {i+1: '—' for i in range(N_ALPHA)}
        self.costs = {i+1: float(i) for i in range(N_ALPHA)}  # example non-flat: 1..22

        # STACKABLE connection multipliers: dict {threshold_lines: multiplier}
        # Default behavior: after 2 lines, ×1.5
        self.connection_thresholds: dict[int, float] = {2: 1.5}

        self.last_info = {}

    # -------- geometry --------
    def vertices_xy(self):
        step = 360.0 / N_VERT
        angs = [(TOP_ANGLE_DEG - i*step if self.clockwise else TOP_ANGLE_DEG + i*step)
                for i in range(N_VERT)]
        angs = np.deg2rad(angs)
        x = self.R * np.cos(angs); y = self.R * np.sin(angs)
        return np.column_stack([x, y])

    # -------- ring labeling with two shifts --------
    def ring_letter_at_vertex(self, v: int, S: int = 0, start_shift: int = 0) -> int:
        """Letter index (1..22) at vertex v (1..11).
           FIRST symbol index a0 = wrap1(start_shift+1, 22).
           FIRST vertex index v0 = wrap1(S+1, 11).
           Vertex v gets the (v - v0)th letter after a0 (mod 22).
        """
        a0 = wrap1(int(start_shift) + 1, N_ALPHA)
        v0 = wrap1(int(S) + 1, N_VERT)
        delta = (int(v) - v0) % N_VERT
        return wrap1(a0 + delta, N_ALPHA)

    def ring_letters(self, S: int = 0, start_shift: int = 0):
        return [self.ring_letter_at_vertex(v, S=S, start_shift=start_shift) for v in range(1, N_VERT+1)]

    # -------- meanings & costs --------
    def set_meanings(self, meanings):
        if isinstance(meanings, (list, tuple)):
            if len(meanings) != N_ALPHA:
                raise ValueError('Provide 22 meanings.')
            for i, txt in enumerate(meanings, start=1):
                self.meanings[i] = str(txt)
        elif isinstance(meanings, dict):
            for k, v in meanings.items():
                k = int(k)
                if not (1 <= k <= N_ALPHA):
                    raise ValueError('Meaning key must be 1..22')
                self.meanings[k] = str(v)
        else:
            raise TypeError('Use list/tuple of 22 or {index:text}')

    def set_costs(self, costs):
        if isinstance(costs, (list, tuple)):
            if len(costs) != N_ALPHA:
                raise ValueError('Provide 22 costs.')
            for i, c in enumerate(costs, start=1):
                self.costs[i] = float(c)
        elif isinstance(costs, dict):
            for k, v in costs.items():
                k = int(k)
                if not (1 <= k <= N_ALPHA):
                    raise ValueError('Cost key must be 1..22')
                self.costs[k] = float(v)
        else:
            raise TypeError('Use list/tuple of 22 or {index:float}')

    # ---- multipliers (stackable) ----
    def set_connection_multipliers(self, thresholds: dict[int, float]):
        """Set stackable multipliers: product of multipliers for all thresholds t with lines>=t.
        Example: {2:1.5, 4:1.3} -> at 5th glyph (4 lines) multiply by 1.5*1.3.
        """
        cleaned = {}
        for t, m in thresholds.items():
            t = int(t); m = float(m)
            if t < 0: 
                continue
            cleaned[t] = m
        self.connection_thresholds = dict(sorted(cleaned.items()))

    # Back-compat with earlier single-threshold API
    def set_connection_multiplier(self, after_lines: int, value: float):
        self.set_connection_multipliers({int(after_lines): float(value)})

    # -------- input sanitation (vertices 1..11) --------
    def _sanitize_vertices(self, vertices):
        clean, seen, invalid, dupes = [], set(), [], []
        for x in vertices:
            try:
                v = int(x)
            except Exception:
                invalid.append(x); continue
            if not (1 <= v <= N_VERT):
                invalid.append(x); continue
            if v in seen:
                dupes.append(v); continue
            seen.add(v); clean.append(v)
        self.last_info = {'invalid_removed': invalid, 'duplicates_removed': dupes}
        return clean

    # -------- cost model (stacked multipliers) --------
    def _accum_multiplier(self, lines_so_far: int) -> float:
        mult = 1.0
        for t, m in self.connection_thresholds.items():
            if lines_so_far >= t:
                mult *= m
        return mult

    def spell_cost_breakdown(self, vertices, S: int = 0, start_shift: int = 0) -> pd.DataFrame:
        """Vertices in -> meanings & costs out.
           k-th glyph uses product of all thresholds satisfied by lines_so_far=(k-1).
        """
        seq = self._sanitize_vertices(vertices)
        rows = []
        total = 0.0
        for idx, v in enumerate(seq, start=1):
            j = self.ring_letter_at_vertex(v, S=S, start_shift=start_shift)
            name = PHOENICIAN_NAMES_22[j-1]
            glyph = phoenician_glyph(j)
            meaning = self.meanings.get(j, '—')
            base = float(self.costs.get(j, 1.0))

            lines_so_far = idx - 1
            mult = self._accum_multiplier(lines_so_far)

            step_cost = base * mult
            total += step_cost

            rows.append({
                'Step': idx,
                'Vertex': v,
                'LetterIndex': j,
                'Name': name,
                'Glyph': glyph,
                'Meaning': meaning,
                'BaseCost': base,
                'LinesSoFar': lines_so_far,
                'Multiplier': mult,
                'StepCost': step_cost,
                'Cumulative': total
            })
        # footer
        rows.append({
            'Step': '',
            'Vertex': '',
            'LetterIndex': '',
            'Name': f"Top={PHOENICIAN_NAMES_22[self.ring_letter_at_vertex(1,S,start_shift)-1]}",
            'Glyph': phoenician_glyph(self.ring_letter_at_vertex(1,S,start_shift)),
            'Meaning': f"S={S}, start_shift={start_shift}; dupes={self.last_info.get('duplicates_removed', [])}; invalid={self.last_info.get('invalid_removed', [])}",
            'BaseCost': '',
            'LinesSoFar': '',
            'Multiplier': '',
            'StepCost': '',
            'Cumulative': total
        })
        return pd.DataFrame(rows)

    def print_spell_meanings_and_cost(self, vertices, S: int = 0, start_shift: int = 0):
        df = self.spell_cost_breakdown(vertices, S=S, start_shift=start_shift)
        lines = []
        for _, row in df.iloc[:-1].iterrows():
            lines.append(f"{int(row['Step']):02d}. V{int(row['Vertex'])} — {row['Name']} : {row['Meaning']}  "
                         f"(base {row['BaseCost']}, x{row['Multiplier']} -> {row['StepCost']})")
        total = df.iloc[-1]['Cumulative']
        text = "\n".join(lines) + f"\nTOTAL COST: {total}"
        print(text)
        return text

    # -------- drawing --------
    def _draw(self, vertices, S: int, start_shift: int, labels: str, arrows: bool, show_circle: bool,
              full_gray: bool, node_size: int, figsize, savepath: str,
              center_number_color: str = 'black', start_dot_size: int = 40):
        V = self.vertices_xy()
        fig, ax = plt.subplots(figsize=figsize)
        if show_circle:
            ax.add_artist(plt.Circle((0,0), self.R, fill=False))

        if full_gray:
            for i in range(N_VERT):
                for j in range(i+1, N_VERT):
                    ax.plot([V[i,0], V[j,0]], [V[i,1], V[j,1]], linewidth=0.6, color='0.85')

        ax.scatter(V[:,0], V[:,1], s=node_size, color='0.0')

        if labels in ('names','glyphs'):
            for i in range(N_VERT):
                v = i + 1
                j = self.ring_letter_at_vertex(v, S=S, start_shift=start_shift)
                vx, vy = V[i]; r = (vx*vx + vy*vy) ** 0.5
                ox = vx * (1 + 0.08*self.R/(r if r else 1)); oy = vy * (1 + 0.08*self.R/(r if r else 1))
                if labels == 'names':
                    ax.text(ox, oy, PHOENICIAN_NAMES_22[j-1], ha='center', va='center', fontsize=11)
                else:
                    if self._glyph_fp is None:
                        raise RuntimeError('glyphs requested but no glyph_font_path was provided at init.')
                    ax.text(ox, oy, phoenician_glyph(j), ha='center', va='center', fontsize=11,
                            fontproperties=self._glyph_fp)

        # Path
        seq = self._sanitize_vertices(vertices)
        if len(seq) >= 1:
            idxs = [v-1 for v in seq]
            pts = V[idxs, :]
            for a, b in zip(pts[:-1], pts[1:]):
                if arrows:
                    ax.annotate('', xy=(b[0], b[1]), xytext=(a[0], a[1]),
                                arrowprops=dict(arrowstyle='-|>', linewidth=2.8, color='0.0'))
                else:
                    ax.plot([a[0], b[0]], [a[1], b[1]], linewidth=2.8, color='0.0')
            ax.scatter([pts[0,0]], [pts[0,1]], s=start_dot_size, color='red', zorder=5)

        # Center number = start_shift
        ax.text(0, 0, f'{int(start_shift)}', ha='center', va='center', fontsize=14, color=center_number_color)

        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-self.R*1.22, self.R*1.22); ax.set_ylim(-self.R*1.22, self.R*1.22)
        if savepath: fig.savefig(savepath, dpi=220, bbox_inches='tight')
        return fig, ax

    # Public API
    def draw_cryptic(self, vertices, S: int = 0, start_shift: int = 0, arrows=True, show_circle=True,
                     full_gray=True, node_size=18, figsize=(6,6), savepath=None):
        return self._draw(vertices, S, start_shift, 'none', arrows, show_circle, full_gray,
                          node_size, figsize, savepath)

    def draw_named(self, vertices, S: int = 0, start_shift: int = 0, arrows=True, show_circle=True,
                   full_gray=True, node_size=18, figsize=(6,6), savepath=None):
        return self._draw(vertices, S, start_shift, 'names', arrows, show_circle, full_gray,
                          node_size, figsize, savepath)

    def draw_glyphs(self, vertices, S: int = 0, start_shift: int = 0, arrows=True, show_circle=True,
                    full_gray=True, node_size=18, figsize=(6,6), savepath=None):
        return self._draw(vertices, S, start_shift, 'glyphs', arrows, show_circle, full_gray,
                          node_size, figsize, savepath)

# ------- demo when run directly -------
if __name__ == '__main__':
    font = r'.\fonts\NotoSansPhoenician-Regular.ttf'  # change if needed
    seq  = [1,7,4,11,9,9,2]  # VERTICES (1..11). Duplicates removed automatically.
    S = 3           # which vertex gets the FIRST symbol
    start_shift = 2 # which alphabet symbol is FIRST (Aleph+2 => Gimel)
    g = Phoenician11System(glyph_font_path=font)

    # Example meanings + non-flat costs
    g.set_meanings({
        1:'Origin', 2:'House', 3:'Path', 4:'Door', 5:'Window', 6:'Link', 7:'Weapon', 8:'Wall', 9:'Coil', 10:'Hand', 11:'Palm',
        12:'Goad', 13:'Water', 14:'Fish', 15:'Prop', 16:'Eye', 17:'Mouth', 18:'Hook', 19:'Back of head', 20:'Head', 21:'Tooth', 22:'Mark'
    })
    g.set_costs({1:2.0,2:1.3,3:1.7,4:2.2,5:1.1,6:1.9,7:2.8,8:1.6,9:1.4,10:1.2,11:1.5,
                 12:2.5,13:1.8,14:2.1,15:1.3,16:2.7,17:1.6,18:1.9,19:2.2,20:1.4,21:2.0,22:1.1})

    # Stackable multipliers: after 2 lines ×1.5, after 4 lines another ×1.5
    g.set_connection_multipliers({2:1.5, 4:1.5})

    # Draw variants
    g.draw_cryptic(seq, S=S, start_shift=start_shift, savepath='spell_cryptic.png')
    g.draw_named(seq,   S=S, start_shift=start_shift, savepath='spell_named.png')
    g.draw_glyphs(seq,  S=S, start_shift=start_shift, savepath='spell_glyphs.png')

    # Meanings + cost with the SAME mapping (S & start_shift)
    g.print_spell_meanings_and_cost(seq, S=S, start_shift=start_shift)
