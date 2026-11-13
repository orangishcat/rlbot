import tkinter as tk

from bridge import (
    CAT_HAZARD,
    CAT_INTERACT,
    CAT_SOLID,
    FACE_CIRCLE,
    GAMEMODE_NAMES,
    Bridge,
)

FACE_COLORS = {
    CAT_SOLID: "#4fc3f7",  # blue: collideable
    CAT_HAZARD: "#f44336",  # red: deadly
    CAT_INTERACT: "#4caf50",  # green: interactible
}


class ControlGUI:
    def __init__(self, root: tk.Tk):
        self.bridge = Bridge()
        self.root = root
        root.title("rlbot control")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.p1_held = False
        self.p2_held = False

        self.status = tk.Label(root, text="connecting to geode bridge...", fg="orange")
        self.status.pack(fill=tk.X)

        self.info = tk.Label(
            root, text="", font=("Menlo", 11), justify=tk.LEFT, anchor="w"
        )
        self.info.pack(fill=tk.X)

        self.canvas = tk.Canvas(root, width=460, height=460, bg="#111")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        buttons = tk.Frame(root)
        buttons.pack(fill=tk.X, pady=4)
        self.b1 = self._hold_button(buttons, "P1 (space)", self.set_p1)
        self.b1.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
        self.b2 = self._hold_button(buttons, "P2 (shift)", self.set_p2)
        self.b2.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)

        root.bind("<KeyPress-space>", lambda e: self.set_p1(True))
        root.bind("<KeyRelease-space>", lambda e: self.set_p1(False))
        root.bind("<KeyPress-Shift_L>", lambda e: self.set_p2(True))
        root.bind("<KeyRelease-Shift_L>", lambda e: self.set_p2(False))

        self._reconnect()
        root.after(16, self._tick)

    def _hold_button(self, parent, text, setter):
        b = tk.Button(parent, text=text, height=2, bg="#333", fg="white")
        b.bind("<ButtonPress-1>", lambda e: setter(True))
        b.bind("<ButtonRelease-1>", lambda e: setter(False))
        return b

    def set_p1(self, held: bool):
        self.p1_held = held
        self._flash(self.b1, held)

    def set_p2(self, held: bool):
        self.p2_held = held
        self._flash(self.b2, held)

    @staticmethod
    def _flash(b: tk.Button, held: bool):
        b.configure(bg="#2e8b57" if held else "#333")

    def _reconnect(self):
        if self.bridge.connected or self.bridge.connect(timeout=0.5):
            self.status.configure(text="connected", fg="green")
        else:
            self.status.configure(text="waiting for geode bridge...", fg="orange")
            self.root.after(1000, self._reconnect)

    def _tick(self):
        if self.bridge.connected:
            self.bridge.poll()
            self.bridge.set_inputs(self.p1_held, self.p2_held)
            self._update_info()
            self._draw()
        self.root.after(16, self._tick)

    def _update_info(self):
        def fmt(p):
            if not p.present:
                return "nonexistent"
            mode = p.gamemode_name or "?"
            g = "up" if p.gravity_flipped else "down"
            return f"{mode:6s} x={p.x:8.1f} y={p.y:8.1f} rot={p.angle:7.1f} grav={g}"

        b = self.bridge
        self.info.configure(
            text=(
                f"P1  {fmt(b.p1)}\n"
                f"P2  {fmt(b.p2)}\n"
                f"two-player: {b.two_player}   faces: {len(b.faces)}   "
                f"fresh: {b.state_fresh}"
            )
        )

    SCALE = 0.8  # pixels per GD unit

    def _draw(self):
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        first_seventh_x, center_y = w // 7, h // 2
        player = self.bridge.p1

        # camera origin; player is placed at x = 1/5 width and y = 1/2 height of canvas
        ox, oy = (
            first_seventh_x - player.x * self.SCALE,
            center_y + player.y * self.SCALE,
        )

        c.create_rectangle(1, 1, w, h, outline="#888")

        s = self.SCALE
        for f in self.bridge.faces:
            color = FACE_COLORS.get(f.category, "#888")
            if f.kind == FACE_CIRCLE:
                r = f.x2 * s
                cx, cy = ox + f.x1 * s, oy - f.y1 * s
                c.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=2)
            else:
                c.create_line(
                    ox + f.x1 * s,
                    oy - f.y1 * s,
                    ox + f.x2 * s,
                    oy - f.y2 * s,
                    fill=color,
                    width=1,
                )

        p_size = 8 * self.SCALE
        if player.present:
            # eq to first_fifth_x, center_y but this makes sure my math is correct
            p1x, p1y = ox + player.x * self.SCALE, oy - player.y * self.SCALE
            c.create_oval(
                p1x - p_size,
                p1y - p_size,
                p1x + p_size,
                p1y + p_size,
                fill="#ffeb3b" if not player.gravity_flipped else "#8bc34a",
            )
        player_2 = self.bridge.p2
        if player_2.present:
            p2x, p2y = ox + player_2.x * self.SCALE, oy - player_2.y * self.SCALE
            c.create_oval(
                p2x - p_size, p2y - p_size, p2x + p_size, p2y + p_size, fill="#f06292"
            )

    def on_close(self):
        self.bridge.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    ControlGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
