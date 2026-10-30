"""On-screen joystick for live_policy.py: writes the setpoint control file that live_policy re-reads every step.

usage: python3 joystick_pilot.py [--control ~/ros2_ws/runs/live/setpoint.json]   (needs a display; runs in the container)
  pad (mouse drag): up/down = surge setpoint (stays where you leave it), left/right = yaw rate (springs back)
  W/S surge +-0.1 m/s   A/D yaw -+15 deg   Up/Down depth -+0.5 m   I/K pitch +-5 deg   J/L roll -+4 deg
  Space: stop (surge 0, level)   R: respawn
Limits follow the training ranges: RACE2_PITCH_MAX_DEG / RACE2_ROLL_MAX_DEG / RACE2_DEPTH_MAX (env).
"""
import argparse
import json
import os
import tkinter as tk

p = argparse.ArgumentParser()
p.add_argument('--control', default=os.path.expanduser('~/ros2_ws/runs/live/setpoint.json'))
p.add_argument('--yaw-rate', type=float, default=45.0, help='deg/s at full pad deflection')
args = p.parse_args()

PITCH_MAX = float(os.environ.get('RACE2_PITCH_MAX_DEG', 20))
ROLL_MAX = float(os.environ.get('RACE2_ROLL_MAX_DEG', 12))
DEPTH_MAX = float(os.environ.get('RACE2_DEPTH_MAX', 12))
SURGE_MAX, DT_MS, R = 1.0, 50, 110          # m/s, GUI/write period, pad radius [px]
INK, MUTED, SURF, PAD, KNOB = '#0b0b0b', '#52514e', '#fcfcfb', '#e4e3df', '#2a78d6'

sp = {'depth': 2.0, 'surge': 0.0, 'yaw_deg': 0.0, 'pitch_deg': 0.0, 'roll_deg': 0.0, 'respawn': 0}
try:
    sp.update(json.load(open(args.control)))
except (OSError, ValueError):
    pass
pad = {'x': 0.0, 'dragging': False}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def write():
    os.makedirs(os.path.dirname(args.control), exist_ok=True)
    tmp = args.control + '.tmp'
    with open(tmp, 'w') as f:
        json.dump({k: round(v, 3) if isinstance(v, float) else v for k, v in sp.items()}, f)
    os.replace(tmp, args.control)   # atomic: live_policy never reads a half-written file


root = tk.Tk()
root.title('RACE2 pilot')
root.configure(bg=SURF)
cv = tk.Canvas(root, width=2 * R + 40, height=2 * R + 40, bg=SURF, highlightthickness=0)
cv.grid(row=0, column=0, padx=10, pady=10)
c0 = R + 20
cv.create_oval(c0 - R, c0 - R, c0 + R, c0 + R, fill=PAD, outline='#c9c8c2')
cv.create_line(c0 - R, c0, c0 + R, c0, fill='#c9c8c2')
cv.create_line(c0, c0 - R, c0, c0 + R, fill='#c9c8c2')
for txt, x, y in (('fwd', c0, 10), ('back', c0, 2 * R + 30), ('yaw L', 18, c0 - 10), ('yaw R', 2 * R + 22, c0 - 10)):
    cv.create_text(x, y, text=txt, fill=MUTED, font=('Helvetica', 8))
knob = cv.create_oval(0, 0, 0, 0, fill=KNOB, outline=SURF, width=2)
status = tk.Label(root, justify='left', font=('Courier', 11), bg=SURF, fg=INK)
status.grid(row=0, column=1, padx=10, sticky='n', pady=20)
tk.Label(root, text='drag pad: surge / yaw rate   W/S surge  A/D yaw  Up/Down depth  I/K pitch  J/L roll  '
                    'Space stop  R respawn', bg=SURF, fg=MUTED, font=('Helvetica', 9)).grid(row=1, column=0, columnspan=2,
                                                                                            pady=(0, 8))


def drag(e):
    dx, dy = (e.x - c0) / R, (c0 - e.y) / R
    n = max(1.0, (dx * dx + dy * dy) ** 0.5)
    pad.update(x=dx / n, dragging=True)
    sp['surge'] = round(clamp(dy / n, -1, 1) * SURGE_MAX, 2)


def release(_):
    pad.update(x=0.0, dragging=False)   # yaw rate springs back; surge stays (throttle)


def key(e):
    k = e.keysym.lower()
    if k == 'w': sp['surge'] = round(clamp(sp['surge'] + 0.1, -SURGE_MAX, SURGE_MAX), 2)
    elif k == 's': sp['surge'] = round(clamp(sp['surge'] - 0.1, -SURGE_MAX, SURGE_MAX), 2)
    elif k == 'a': sp['yaw_deg'] -= 15
    elif k == 'd': sp['yaw_deg'] += 15
    elif k == 'up': sp['depth'] = clamp(sp['depth'] - 0.5, 0.5, DEPTH_MAX)
    elif k == 'down': sp['depth'] = clamp(sp['depth'] + 0.5, 0.5, DEPTH_MAX)
    elif k == 'i': sp['pitch_deg'] = clamp(sp['pitch_deg'] + 5, -PITCH_MAX, PITCH_MAX)
    elif k == 'k': sp['pitch_deg'] = clamp(sp['pitch_deg'] - 5, -PITCH_MAX, PITCH_MAX)
    elif k == 'l': sp['roll_deg'] = clamp(sp['roll_deg'] + 4, -ROLL_MAX, ROLL_MAX)
    elif k == 'j': sp['roll_deg'] = clamp(sp['roll_deg'] - 4, -ROLL_MAX, ROLL_MAX)
    elif k == 'space': sp.update(surge=0.0, pitch_deg=0.0, roll_deg=0.0)
    elif k == 'r': sp['respawn'] = int(sp.get('respawn', 0)) + 1


def tick():
    sp['yaw_deg'] += pad['x'] * args.yaw_rate * DT_MS / 1000.0
    sp['yaw_deg'] = (sp['yaw_deg'] + 180.0) % 360.0 - 180.0
    kx, ky = c0 + pad['x'] * R, c0 - sp['surge'] / SURGE_MAX * R
    cv.coords(knob, kx - 14, ky - 14, kx + 14, ky + 14)
    status.config(text=f"depth  {sp['depth']:5.1f} m\nsurge  {sp['surge']:+5.2f} m/s\nyaw    {sp['yaw_deg']:+6.1f} deg\n"
                       f"pitch  {sp['pitch_deg']:+5.0f} deg (max {PITCH_MAX:.0f})\nroll   {sp['roll_deg']:+5.0f} deg (max {ROLL_MAX:.0f})\n"
                       f"respawns {sp['respawn']}")
    write()
    root.after(DT_MS, tick)


cv.bind('<B1-Motion>', drag)
cv.bind('<Button-1>', drag)
cv.bind('<ButtonRelease-1>', release)
root.bind('<Key>', key)
write()
tick()
root.mainloop()
