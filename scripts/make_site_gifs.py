"""Animated GIFs for the project page (docs/assets/*.gif).

Every number drawn here is taken from the paper; nothing is read from transcripts.
The replicate-row animation is a schematic of the readout in Fig. 1c: blocks stand
for messages, not for real message text.

    python scripts/make_site_gifs.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets"
SS = 2  # supersampling factor

# Validated categorical slots 1-3 (light mode) + recessive neutrals.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID, EMPTY = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#efeee9"

SANS = "/System/Library/Fonts/Supplemental/Arial.ttf"
SANS_B = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(SANS_B if bold else SANS, size * SS)


def canvas(w: int, h: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (w * SS, h * SS), SURFACE)
    return im, ImageDraw.Draw(im)


def rr(d, x, y, w, h, fill, r=4, outline=None, width=0):
    d.rounded_rectangle([x * SS, y * SS, (x + w) * SS, (y + h) * SS], radius=r * SS,
                        fill=fill, outline=outline, width=width * SS)


def text(d, x, y, s, f, fill=INK, anchor="la"):
    d.text((x * SS, y * SS), s, font=f, fill=fill, anchor=anchor)


def finish(frames: list[Image.Image], durations: list[int], name: str) -> None:
    small = [fr.resize((fr.width // SS, fr.height // SS), Image.LANCZOS) for fr in frames]
    # One shared palette, built from a strip of every frame, so a colour never shifts between frames.
    strip = Image.new("RGB", (small[0].width, small[0].height * len(small)))
    for k, fr in enumerate(small):
        strip.paste(fr, (0, k * fr.height))
    ref = strip.quantize(colors=128, method=Image.MEDIANCUT, dither=Image.NONE)
    pal = [fr.quantize(palette=ref, dither=Image.NONE) for fr in small]
    OUT.mkdir(parents=True, exist_ok=True)
    pal[0].save(OUT / name, save_all=True, append_images=pal[1:], duration=durations,
                loop=0, optimize=True, disposal=2)
    print(name, len(frames), "frames", (OUT / name).stat().st_size // 1024, "KB")


# ---------------------------------------------------------------- 1. replicates
def gif_replicates() -> None:
    """Five greedy replicates, read out message by message (Fig. 1c, schematic).

    Cache on: rows split at message index 1, the simulated user's opening utterance.
    Cache off: every row identical to the end.
    """
    W, H, N, R = 720, 330, 12, 5
    x0, y0, cw, ch, gap = 96, 92, 42, 26, 6
    roles = ["agent", "user"] * (N // 2)  # index 0 is the agent greeting
    role_col = {"agent": AQUA, "user": BLUE}
    # per replicate: shade offsets after divergence so rows visibly differ
    variants = [0, 1, 2, 3, 4]
    frames, durs = [], []

    def draw(phase: str, upto: int, banner: str | None):
        im, d = canvas(W, H)
        on = phase == "on"
        text(d, 24, 22, "Five greedy replicates of one airline task", font(17, True))
        tag = "prefix cache ON (default)" if on else "prefix cache OFF"
        tw = d.textlength(tag, font=font(12, True)) / SS
        rr(d, 24, 50, tw + 24, 24, ORANGE if on else BLUE, r=12)
        text(d, 36, 55, tag, font(12, True), fill="#ffffff")
        text(d, W - 24, 56, "T = 0 for both models", font(12), fill=INK2, anchor="ra")
        for i in range(N):
            text(d, x0 + i * (cw + gap) + cw / 2, y0 - 6, str(i), font(10), fill=MUTED, anchor="ms")
        for r in range(R):
            y = y0 + r * (ch + 10)
            text(d, x0 - 16, y + ch / 2, f"run {r + 1}", font(12), fill=INK2, anchor="rm")
            for i in range(N):
                x = x0 + i * (cw + gap)
                if i >= upto:
                    rr(d, x, y, cw, ch, EMPTY)
                    continue
                base = role_col[roles[i]]
                diverged = on and i >= 1
                if diverged:
                    rr(d, x, y, cw, ch, "#fbe3d9", outline=ORANGE, width=2)
                    # a different "content" mark per replicate
                    k = variants[r]
                    for j in range(3):
                        ln = [26, 18, 30, 22, 14][(k + j) % 5]
                        rr(d, x + 6, y + 5 + j * 6, ln, 3, ORANGE, r=1)
                else:
                    rr(d, x, y, cw, ch, base)
                    for j in range(3):
                        ln = [30, 22, 26][j]
                        rr(d, x + 6, y + 5 + j * 6, ln, 3, "#ffffff", r=1)
        if on and upto >= 2:
            xm = x0 + 1 * (cw + gap) - gap / 2
            d.line([xm * SS, (y0 - 2) * SS, xm * SS, (y0 + R * (ch + 10)) * SS], fill=ORANGE, width=2 * SS)
        yl = H - 34
        for k, (lab, col) in enumerate([("agent message", AQUA), ("simulated user", BLUE)]):
            rr(d, 24 + k * 140, yl, 14, 14, col, r=3)
            text(d, 44 + k * 140, yl + 1, lab, font(12), fill=INK2)
        if on:
            rr(d, 304, yl, 14, 14, "#fbe3d9", r=3, outline=ORANGE, width=2)
            text(d, 324, yl + 1, "runs disagree", font(12), fill=INK2)
        if banner:
            text(d, W - 24, yl + 1, banner, font(13, True), fill=ORANGE if on else BLUE, anchor="ra")
        return im

    for phase in ("on", "off"):
        for u in range(0, N + 1):
            frames.append(draw(phase, u, None)); durs.append(130 if u else 500)
        msg = ("diverges at message 1: the user's opening line" if phase == "on"
               else "all five runs identical, message for message")
        frames.append(draw(phase, N, msg)); durs.append(2600)
    finish(frames, durs, "replicates.gif")


# ---------------------------------------------------------------- 2. 2x2 matrix
def gif_matrix() -> None:
    """Engine x cache, 20 matched airline tasks (Fig. 3a): 0/20 on, 20/20 off, both engines."""
    W, H = 720, 440
    cx, cy, cw, chh = 150, 104, 260, 128
    cells = {("vLLM", "on"): 0, ("vLLM", "off"): 20, ("SGLang", "on"): 0, ("SGLang", "off"): 20}
    order = [("vLLM", "on"), ("SGLang", "on"), ("vLLM", "off"), ("SGLang", "off")]
    frames, durs = [], []

    def draw(progress: dict):
        im, d = canvas(W, H)
        text(d, 24, 22, "Engine × cache: tasks whose 3 replicates are identical", font(17, True))
        text(d, 24, 48, "20 matched airline tasks · Qwen3-30B-A3B agent, Mistral-7B simulator · T = 0",
             font(12), fill=INK2)
        for ci, cache in enumerate(["on", "off"]):
            lab = "prefix cache ON (default)" if cache == "on" else "prefix cache OFF"
            text(d, cx + ci * (cw + 16) + cw / 2, cy - 10, lab, font(13, True),
                 fill=ORANGE if cache == "on" else BLUE, anchor="ms")
        for ri, eng in enumerate(["vLLM", "SGLang"]):
            text(d, cx - 18, cy + ri * (chh + 14) + chh / 2, eng, font(15, True), anchor="rm")
            for ci, cache in enumerate(["on", "off"]):
                x, y = cx + ci * (cw + 16), cy + ri * (chh + 14)
                col = ORANGE if cache == "on" else BLUE
                rr(d, x, y, cw, chh, "#ffffff", r=8, outline=GRID, width=1)
                n = progress.get((eng, cache), 0)
                for t in range(20):
                    dx, dy = x + 18 + (t % 10) * 23, y + 18 + (t // 10) * 23
                    if t < n:
                        if cache == "on":
                            rr(d, dx, dy, 16, 16, "#fbe3d9", r=8, outline=ORANGE, width=2)
                        else:
                            rr(d, dx, dy, 16, 16, BLUE, r=8)
                    else:
                        rr(d, dx, dy, 16, 16, EMPTY, r=8)
                if n == 20:
                    k = cells[(eng, cache)]
                    text(d, x + 18, y + 72, f"{k}/20", font(26, True), fill=col)
                    text(d, x + 98, y + 82, "reproduce",
                         font(13), fill=INK2)
        rr(d, 24, H - 30, 14, 14, BLUE, r=7); text(d, 44, H - 29, "identical", font(12), fill=INK2)
        rr(d, 124, H - 30, 14, 14, "#fbe3d9", r=7, outline=ORANGE, width=2)
        text(d, 144, H - 29, "diverged", font(12), fill=INK2)
        return im

    prog: dict = {}
    frames.append(draw(prog)); durs.append(500)
    for key in order:
        for t in range(1, 21, 2):
            prog[key] = t + 1
            frames.append(draw(dict(prog))); durs.append(45)
        frames.append(draw(dict(prog))); durs.append(700)
    frames.append(draw(dict(prog))); durs.append(3000)
    finish(frames, durs, "matrix.gif")


# ---------------------------------------------------------------- 3. domains
def gif_domains() -> None:
    """The same flag in four domains (Fig. 3b), matched tasks."""
    W, H = 720, 380
    data = [("airline", 0, 20, 20), ("retail", 0, 20, 20), ("telecom", 1, 11, 11), ("banking", 4, 7, 19)]
    px, py, pw, ph = 150, 70, 330, 250
    frames, durs = [], []

    def ypos(frac):
        return py + ph - frac * ph

    def draw(t: float, labels: bool):
        im, d = canvas(W, H)
        text(d, 24, 22, "The same flag in each of four domains", font(17, True))
        for g in (0, 25, 50, 75, 100):
            y = ypos(g / 100)
            d.line([px * SS, y * SS, (px + pw) * SS, y * SS], fill=GRID, width=1 * SS)
            text(d, px - 10, y, f"{g}%", font(11), fill=MUTED, anchor="rm")
        text(d, px, py + ph + 14, "prefix cache ON", font(12, True), fill=ORANGE, anchor="ma")
        text(d, px + pw, py + ph + 14, "prefix cache OFF", font(12, True), fill=BLUE, anchor="ma")
        for name, on, off, n in data:
            col = ORANGE if name == "banking" else BLUE
            y1, y2 = ypos(on / n), ypos(off / n)
            xe, ye = px + pw * t, y1 + (y2 - y1) * t
            d.line([px * SS, y1 * SS, xe * SS, ye * SS], fill=col, width=3 * SS)
            r = 6
            d.ellipse([(px - r) * SS, (y1 - r) * SS, (px + r) * SS, (y1 + r) * SS], fill=col,
                      outline=SURFACE, width=2 * SS)
            if t >= 1:
                d.ellipse([(xe - r) * SS, (ye - r) * SS, (xe + r) * SS, (ye + r) * SS], fill=col,
                          outline=SURFACE, width=2 * SS)
        if labels:
            text(d, px - 46, ypos(4 / 19), "4/19", font(11, True), fill=ORANGE, anchor="rm")
            text(d, px - 46, ypos(1 / 11), "1/11", font(11, True), fill=BLUE, anchor="rm")
            text(d, px - 46, ypos(0), "0/20", font(11, True), fill=BLUE, anchor="rm")
            ys = {"airline": 0, "retail": 1, "telecom": 2}
            for name, on, off, n in data:
                if name == "banking":
                    text(d, px + pw + 14, ypos(off / n), f"banking  {off}/{n}", font(13, True),
                         fill=INK, anchor="lm")
                else:
                    text(d, px + pw + 14, ypos(1) + ys[name] * 18 - 2, f"{name}  {off}/{n}",
                         font(13, True), fill=INK, anchor="lm")
            text(d, px + pw + 14, ypos(7 / 19) + 22, "residual is at tool", font(11), fill=INK2, anchor="lm")
            text(d, px + pw + 14, ypos(7 / 19) + 36, "results, never a model", font(11), fill=INK2, anchor="lm")
        return im

    frames.append(draw(0, False)); durs.append(700)
    for k in range(1, 21):
        frames.append(draw(k / 20, False)); durs.append(50)
    frames.append(draw(1, True)); durs.append(3200)
    finish(frames, durs, "domains.gif")


# ---------------------------------------------------------------- 4. concealment
def gif_conceal() -> None:
    """Reward hides most divergence (Fig. 4): airline 50 diverge / 12 change reward."""
    W, H = 720, 300
    gx, gy, s, g = 40, 96, 20, 5
    frames, durs = [], []
    changed = {0, 4, 9, 13, 17, 22, 27, 31, 36, 40, 44, 48}  # 12 cells, spread for legibility (schematic positions)

    def draw(step: int, nlit: int):
        im, d = canvas(W, H)
        text(d, 24, 22, "What the score sees", font(17, True))
        sub = {0: "50 airline tasks, both models greedy, default deployment",
               1: "transcripts: all 50 tasks diverge across replicates",
               2: "reward: only 12 of 50 tasks change score"}[step]
        text(d, 24, 50, sub, font(13), fill=INK2)
        for t in range(50):
            x, y = gx + (t % 25) * (s + g), gy + (t // 25) * (s + g)
            if step == 0 or t >= nlit:
                fill, out = EMPTY, None
            elif step == 1:
                fill, out = "#fbe3d9", ORANGE
            else:
                fill, out = (AQUA, None) if t in changed else ("#fbe3d9", ORANGE)
            rr(d, x, y, s, s, fill, r=4, outline=out, width=2 if out else 0)
        if step == 2:
            text(d, gx, gy + 2 * (s + g) + 28, "12", font(30, True), fill=AQUA)
            text(d, gx + 46, gy + 2 * (s + g) + 40, "change reward", font(13), fill=INK2)
            text(d, gx + 210, gy + 2 * (s + g) + 28, "38", font(30, True), fill=ORANGE)
            text(d, gx + 256, gy + 2 * (s + g) + 40, "diverge invisibly (76%)", font(13), fill=INK2)
            text(d, gx, gy + 2 * (s + g) + 86, "Retail: 8 change reward, 42 invisible (84%).", font(12), fill=MUTED)
        return im

    frames.append(draw(0, 0)); durs.append(900)
    for n in range(2, 51, 2):
        frames.append(draw(1, n)); durs.append(40)
    frames.append(draw(1, 50)); durs.append(1400)
    frames.append(draw(2, 50)); durs.append(3600)
    finish(frames, durs, "conceal.gif")


if __name__ == "__main__":
    gif_replicates()
    gif_matrix()
    gif_domains()
    gif_conceal()
