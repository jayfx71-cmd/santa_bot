"""Day recap card, 'Profits Santa Ai' style: shows only the winning trades of the day (real ones, nothing invented)."""
from PIL import Image, ImageDraw
import friend_card as fc
from friend_card import font, glow, spaced, vgrad, hgrad, rrect_mask, rgba, base_card, candles, THEMES, WHITE, GREY, W, H


def render_recap(day_label, wins, path="recap.png", be_count=0):
    th = THEMES["BUY"]
    main, light, dark = th["main"], th["light"], th["dark"]
    img = base_card(main)
    candles(img, True)
    d = ImageDraw.Draw(img)
    spaced(d, (1470, 140), "DAY RECAP", font("Bold", 34), rgba(light), 5, "r")
    d.text((1470, 188), day_label.upper(), font=font("Medium", 24), fill=WHITE, anchor="rm")

    # big number panel
    bx = (138, 323, 964, 475)
    glow(img, lambda dd, c: dd.rounded_rectangle(bx, 38, outline=rgba(c), width=6), main, 12, 0.9)
    img.paste(hgrad((W, H), rgba(dark, 190), (10, 12, 16, 235)), (0, 0), rrect_mask((W, H), bx, 38))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(bx, 38, outline=rgba(main, 235), width=3)
    n = len(wins)
    fb = font("Black", 112)
    d.text((190, 400), str(n), font=fb, fill=WHITE, anchor="lm")
    nx = 190 + fb.getlength(str(n)) + 30
    d.text((nx, 372), "WINNING", font=font("Bold", 40), fill=WHITE, anchor="lm")
    d.text((nx, 422), "TRADES TODAY", font=font("Medium", 28), fill=(190, 196, 208), anchor="lm")
    gold = Image.open(fc.A / "friend_gold.png").convert("RGBA")
    img.alpha_composite(gold, (985, 322))
    d = ImageDraw.Draw(img)
    d.text((1242, 373), "XAU/USD", font=font("Bold", 56), fill=WHITE, anchor="lm")
    spaced(d, (1243, 428), "GOLD SCALPS", font("Medium", 24), (150, 156, 170), 6)

    # winners list (up to 12, two columns of 6)
    if be_count:
        d.text((140, 508), f"+ {be_count} trade{'s' if be_count != 1 else ''} protected at break-even (partial wins)", font=font("Medium", 22), fill=GREY, anchor="lm")
    ly = 535
    for k, t in enumerate(wins[:12]):
        col, row = divmod(k, 6)
        x, y = 138 + col * 757, ly + row * 40
        d.rounded_rectangle((x, y, x + 729, y + 34), 12, fill=(14, 17, 22, 255), outline=(60, 66, 80, 255), width=1)
        d.ellipse((x + 16, y + 11, x + 28, y + 23), fill=rgba(main))
        d.text((x + 44, y + 17), f"#{t['n']:02d}   {t['side']}   {t['session'].title()}   1:{t['ratio']:g}", font=font("Medium", 21), fill=WHITE, anchor="lm")
        d.text((x + 710, y + 17), "WIN", font=font("Bold", 20), fill=rgba(light), anchor="rm")
    if len(wins) > 12:
        d.text((140, 783), f"+ {len(wins) - 12} more winning trades", font=font("Medium", 21), fill=GREY, anchor="lm")
    if not wins:
        d.text((140, 600), "No winning trades today.", font=font("Medium", 30), fill=GREY, anchor="lm")
    ln = hgrad((1500, 4), (255, 59, 85, 255), (34, 214, 118, 255))
    img.alpha_composite(ln, (138, 797))
    img.convert("RGB").save(path, quality=93)
    return path


if __name__ == "__main__":
    ws = [dict(n=i, side="BUY" if i % 2 else "SELL", session=["MORNING", "AFTERNOON", "EVENING"][i % 3], ratio=[2.0, 3.0][i % 2]) for i in range(1, 15)]
    render_recap("Thu 15 Oct", ws, "recap.png", be_count=3)
