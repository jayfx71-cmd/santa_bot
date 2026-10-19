"""Signal card, 'Profits Santa Ai' style: dark glass card, logo top left, big BUY/SELL panel, gold bars, candlesticks.
BUY = green, SELL = red. Needs friend_logo.png, friend_gold.png and the five Inter-*.otf font files next to this file."""
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

A = Path(__file__).parent if (Path(__file__).parent / "friend_logo.png").exists() else Path(__file__).parent / "assets"
W, H = 1774, 887
BRAND_A, BRAND_B = "Profits Santa", "Ai"          # the name on the card: first part white, second part red
TAGLINE = ("TRADE SMART", "STAY PROFITABLE")
THEMES = {
    "BUY":  dict(main=(34, 214, 118), light=(120, 250, 175), dark=(10, 70, 40)),
    "SELL": dict(main=(255, 59, 85), light=(255, 125, 140), dark=(95, 14, 30)),
}
WHITE, GREY, RED, GREEN = (255, 255, 255), (150, 158, 172), (255, 59, 85), (34, 214, 118)
SL_COL, TP_COL = (255, 59, 85), (40, 220, 130)


def font(name, size):
    return ImageFont.truetype(str(A / f"Inter-{name}.otf"), size)


def rgba(c, a=255):
    return (c[0], c[1], c[2], int(a))


def glow(base, draw_fn, color, radius, strength=1.0):
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(layer), color)
    layer = layer.filter(ImageFilter.GaussianBlur(radius))
    if strength != 1.0:
        layer.putalpha(layer.getchannel("A").point(lambda v: min(255, int(v * strength))))
    base.alpha_composite(layer)


def spaced_width(text, f, sp):
    return sum(f.getlength(ch) for ch in text) + sp * (len(text) - 1)


def spaced(d, xy, text, f, fill, sp=0, anchor="l"):
    x, y = xy
    w = spaced_width(text, f, sp)
    x = x - (w / 2 if anchor == "m" else (w if anchor == "r" else 0))
    for ch in text:
        d.text((x, y), ch, font=f, fill=fill, anchor="lm")
        x += f.getlength(ch) + sp


def vgrad(size, top, bottom):
    w, h = size
    t = np.linspace(0, 1, h)[:, None, None]
    arr = np.array(top, np.float32) * (1 - t) + np.array(bottom, np.float32) * t
    return Image.fromarray(np.repeat(arr, w, axis=1).astype(np.uint8), "RGBA")


def hgrad(size, left, right):
    w, h = size
    t = np.linspace(0, 1, w)[None, :, None]
    arr = np.array(left, np.float32) * (1 - t) + np.array(right, np.float32) * t
    return Image.fromarray(np.repeat(arr, h, axis=0).astype(np.uint8), "RGBA")


def rrect_mask(size, box, r):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle(box, r, fill=255)
    return m


def base_card(side_main):
    """Background + glass card + logo + brand name + tagline. Used by both the signal card and the recap."""
    img = Image.new("RGBA", (W, H), (7, 8, 11, 255))
    glow(img, lambda d, c: d.polygon([(0, 0), (520, 0), (300, 330), (0, 330)], fill=(120, 140, 190, 40)), None, 90)
    glow(img, lambda d, c: (d.ellipse((1450, 600, 2000, 1100), fill=rgba(c, 70)), d.ellipse((-300, -200, 200, 250), fill=rgba(c, 40))), side_main, 110)
    card = (68, 90, 1708, 828)
    img.paste(vgrad((W, H), (14, 16, 21, 250), (8, 9, 12, 252)), (0, 0), rrect_mask((W, H), card, 42))
    ImageDraw.Draw(img).rounded_rectangle(card, 42, outline=(70, 76, 90, 255), width=2)
    # logo, divider, brand
    logo = Image.open(A / "friend_logo.png").convert("RGBA")
    img.alpha_composite(logo, (125, 112))
    d = ImageDraw.Draw(img)
    d.line([(328, 155), (328, 258)], fill=(70, 75, 88, 255), width=2)
    fb = font("Black", 76)
    d.text((370, 192), BRAND_A + " ", font=fb, fill=WHITE, anchor="lm")
    d.text((370 + fb.getlength(BRAND_A + " "), 192), BRAND_B, font=fb, fill=RED, anchor="lm")
    spaced(d, (373, 251), "TRADING SIGNALS", font("Medium", 25), (170, 176, 190), 9)
    return img


def candles(img, rising):
    """Faded candlesticks at the right edge: rising (mostly green) for BUY, falling (mostly red) for SELL."""
    rnd = random.Random(7)
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    n, x0, step = 8, 1500, 24
    for i in range(n):
        t = i / (n - 1)
        centre = (500 - t * 250) if rising else (265 + t * 235)
        h = rnd.randint(36, 78)
        up = rnd.random() > 0.3 if rising else rnd.random() < 0.3
        col = GREEN if up else RED
        a = int(50 + 120 * (t if rising else 1 - t))
        x = x0 + i * step
        top, bot = centre - h / 2, centre + h / 2
        d.line([(x + 9, top - 20), (x + 9, bot + 20)], fill=rgba(col, a), width=2)
        d.rounded_rectangle((x, top, x + 18, bot), 3, fill=rgba(col, a))
    img.alpha_composite(layer)


def render(tr, path="scalp.png"):
    side = tr["side"]
    th = THEMES[side]
    main, light, dark = th["main"], th["light"], th["dark"]
    img = base_card(main)
    # top right tagline
    d = ImageDraw.Draw(img)
    spaced(d, (1342, 140), TAGLINE[0], font("Medium", 22), (140, 146, 160), 6)
    spaced(d, (1342, 169), TAGLINE[1], font("Medium", 22), (140, 146, 160), 6)
    d.line([(1342, 199), (1527, 199)], fill=(120, 126, 140, 255), width=2)
    candles(img, side == "BUY")

    # BUY / SELL panel
    box = (138, 323, 964, 520)
    glow(img, lambda dd, c: dd.rounded_rectangle(box, 40, outline=rgba(c, 255), width=6), main, 12, 0.9)
    img.paste(hgrad((W, H), rgba(dark, 190), (10, 12, 16, 235)), (0, 0), rrect_mask((W, H), box, 40))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(box, 40, outline=rgba(main, 235), width=3)
    cx, cy = 268, 425
    d.ellipse((cx - 68, cy - 68, cx + 68, cy + 68), fill=rgba(dark, 150))
    tri = [(cx - 38, cy + 24), (cx + 38, cy + 24), (cx, cy - 36)] if side == "BUY" else [(cx - 38, cy - 24), (cx + 38, cy - 24), (cx, cy + 36)]
    glow(img, lambda dd, c: dd.polygon(tri, fill=rgba(c)), light, 8, 1.0)
    d = ImageDraw.Draw(img)
    d.polygon(tri, fill=rgba(light))
    d.line([(376, 368), (376, 485)], fill=(80, 90, 100, 255), width=2)
    d.text((422, 400), side, font=font("Black", 128), fill=WHITE, anchor="lm")
    spaced(d, (424, 478), f"XAU/USD  ·  RR 1:{tr['ratio']:g}", font("Medium", 24), (175, 182, 195), 6)
    if tr.get("n"):                                   # signal counter, top right of the panel
        d.text((934, 370), f"#{tr['n']:02d}", font=font("Black", 44), fill=rgba(light), anchor="rm")
        d.text((934, 416), "/ 60", font=font("Bold", 26), fill=(150, 156, 170), anchor="rm")

    # gold bars + instrument
    gold = Image.open(A / "friend_gold.png").convert("RGBA")
    img.alpha_composite(gold, (985, 322))
    d = ImageDraw.Draw(img)
    d.text((1242, 373), "XAU/USD", font=font("Bold", 56), fill=WHITE, anchor="lm")
    spaced(d, (1243, 428), "GOLD", font("Medium", 26), (150, 156, 170), 8)

    # three boxes
    risk = tr["risk"]
    spec = [((138, 553, 623, 763), "ENTRY", f"{tr['entry']:.2f}", WHITE, None, (200, 206, 220), "cross"),
            ((649, 553, 1127, 763), "STOP LOSS", f"{tr['sl']:.2f}", SL_COL, f"-${risk:.2f}", SL_COL, "shield"),
            ((1152, 553, 1637, 763), "TAKE PROFIT", f"{tr['tp']:.2f}", TP_COL, f"+${risk * tr['ratio']:.2f}", TP_COL, "trend")]
    for bx, label, value, vcol, sub, ecol, icon in spec:
        img.paste(vgrad((W, H), (15, 17, 23, 245), (8, 9, 12, 250)), (0, 0), rrect_mask((W, H), bx, 30))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle(bx, 30, outline=(66, 72, 86, 255), width=2)
        x0, y0 = bx[0], bx[1]
        ix, iy, ic = x0 + 70, y0 + 54, rgba(ecol, 240)
        if icon == "cross":
            d.ellipse((ix - 22, iy - 22, ix + 22, iy + 22), outline=ic, width=3)
            d.ellipse((ix - 5, iy - 5, ix + 5, iy + 5), outline=ic, width=3)
            for (a_, b_, c_, e_) in ((ix - 32, iy, ix - 14, iy), (ix + 14, iy, ix + 32, iy), (ix, iy - 32, ix, iy - 14), (ix, iy + 14, ix, iy + 32)):
                d.line([(a_, b_), (c_, e_)], fill=ic, width=3)
        elif icon == "shield":
            d.polygon([(ix, iy - 28), (ix + 24, iy - 17), (ix + 21, iy + 12), (ix, iy + 28), (ix - 21, iy + 12), (ix - 24, iy - 17)], outline=ic, width=3)
            d.line([(ix - 9, iy + 1), (ix - 2, iy + 9), (ix + 11, iy - 8)], fill=ic, width=3)
        else:
            d.line([(ix - 28, iy + 16), (ix - 8, iy - 4), (ix + 4, iy + 8), (ix + 26, iy - 18)], fill=ic, width=4, joint="curve")
            d.polygon([(ix + 28, iy - 24), (ix + 28, iy - 6), (ix + 10, iy - 22)], fill=ic)
        spaced(d, (x0 + 137, y0 + 53), label, font("Medium", 24), (175, 182, 195), 6)
        d.text((x0 + 125, y0 + 115), value, font=font("Bold", 68), fill=rgba(vcol), anchor="lm")
        if sub:
            d.text((x0 + 128, y0 + 175), sub, font=font("Medium", 32), fill=rgba(ecol), anchor="lm")

    # bottom gradient line (red to green)
    ln = hgrad((1500, 4), (255, 59, 85, 255), (34, 214, 118, 255))
    glow(img, lambda dd, c: dd.line([(138, 795), (1637, 795)], fill=(255, 255, 255, 120), width=4), None, 6, 0.5)
    img.alpha_composite(ln, (138, 793))
    img.convert("RGB").save(path, quality=93)
    return path


if __name__ == "__main__":
    base = dict(n=7, session="MORNING", side="BUY", entry=4153.70, sl=4150.78, tp=4161.71, ratio=3.0, risk=2.67)
    render(base, "buy.png")
    render(dict(base, side="SELL", sl=4156.37, tp=4145.69), "sell.png")
