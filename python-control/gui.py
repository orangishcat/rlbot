import tkinter as tk
from tkinter import simpledialog

from bridge import (
    CAT_HAZARD,
    CAT_INTERACT,
    CAT_SOLID,
    CMD_NO_LEVEL,
    CMD_NOT_FOUND,
    CMD_OK,
    CMD_OPEN_LEVEL,
    CMD_RESTART,
    CMD_UNKNOWN,
    FACE_CIRCLE,
    GdBridge,
)

FACE_COLORS = {
    CAT_SOLID: "#4fc3f7",  # blue: collideable
    CAT_HAZARD: "#f44336",  # red: deadly
    CAT_INTERACT: "#4caf50",  # green: interactible
}


class ControlGUI:
    def __init__(self, root: tk.Tk):
        self.bridge = GdBridge()
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

        controls = tk.Frame(root)
        controls.pack(fill=tk.X, pady=2)
        self.step_enabled = tk.BooleanVar(value=False)
        tk.Checkbutton(
            controls,
            text="frame stepping (240 tps)",
            variable=self.step_enabled,
        ).pack(side=tk.LEFT, padx=4)
        self.step_btn = tk.Button(
            controls,
            text="step ×1 (t)",
            width=11,
            state=tk.DISABLED,
            command=lambda: self._request_steps(1),
        )
        self.step_btn.pack(side=tk.LEFT, padx=4)
        self.step10_btn = tk.Button(
            controls,
            text="step ×10 (T)",
            width=11,
            state=tk.DISABLED,
            command=lambda: self._request_steps(10),
        )
        self.step10_btn.pack(side=tk.LEFT, padx=4)
        root.bind("<KeyPress-t>", lambda e: self._request_steps(1))
        root.bind("<KeyPress-T>", lambda e: self._request_steps(10))

        actions = tk.Frame(root)
        actions.pack(fill=tk.X, pady=2)
        tk.Button(
            actions,
            text="restart (r)",
            width=11,
            command=self.restart_level,
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(
            actions,
            text="open level (o)",
            width=14,
            command=self.open_level,
        ).pack(side=tk.LEFT, padx=4)
        root.bind("<KeyPress-r>", lambda e: self.restart_level())
        root.bind("<KeyPress-o>", lambda e: self.open_level())

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
            self.bridge.set_step_mode(self.step_enabled.get())
            stepping = self.step_enabled.get()
            step_state = tk.NORMAL if stepping else tk.DISABLED
            self.step_btn.configure(state=step_state)
            self.step10_btn.configure(state=step_state)
            self._update_info()
            self._draw()
        self.root.after(16, self._tick)

    def _request_steps(self, n: int):
        if self.bridge.connected and self.step_enabled.get():
            self.bridge.request_steps(n)

    def restart_level(self):
        if self.bridge.connected:
            self.bridge.request_command(CMD_RESTART)

    def open_level(self):
        if not self.bridge.connected:
            return
        name = simpledialog.askstring("open level", "level name:", parent=self.root)
        if name and name.strip():
            self.bridge.request_command(CMD_OPEN_LEVEL, name.strip())

    @staticmethod
    def _cmd_text(b: GdBridge) -> str:
        if b.cmd_req == 0:
            return "idle"
        if b.cmd_done != b.cmd_req:
            return "running..."
        return {
            CMD_OK: "ok",
            CMD_NOT_FOUND: "level not found",
            CMD_NO_LEVEL: "not in a level",
            CMD_UNKNOWN: "unknown command",
        }.get(b.cmd_status, f"status {b.cmd_status}")

    def _update_info(self):
        def fmt(p):
            if not p.present:
                return "nonexistent"
            mode = p.gamemode.name.lower()
            g = "up" if p.gravity_flipped else "down"
            return (
                f"{mode:6s} x={p.x:8.1f} y={p.y:8.1f} grav={g} "
                f"size={p.size_x:.1f}x{p.size_y:.1f} "
                f"vel={p.vel_x:7.1f},{p.vel_y:7.1f}"
            )

        b = self.bridge
        self.info.configure(
            text=(
                f"P1  {fmt(b.p1)}\n"
                f"P2  {fmt(b.p2)}\n"
                f"two-player: {b.two_player}   faces: {len(b.faces)}   "
                f"fresh: {b.state_fresh}\n"
                f"attempt: {b.attempt_ms / 1000:.2f}s   "
                f"destroyed: {'yes' if b.player_destroyed else 'no'}\n"
                f"steps: req={b.step_req} done={b.step_done}   mode: "
                f"{'frame stepping' if b.step_mode else 'realtime'}\n"
                f"cmd: {self._cmd_text(b)}"
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
            color = FACE_COLORS.get(f["category"], "#888")
            if f["kind"] == FACE_CIRCLE:
                r = f["x2"] * s
                cx, cy = ox + f["x1"] * s, oy - f["y1"] * s
                c.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=2)
            else:
                c.create_line(
                    ox + f["x1"] * s,
                    oy - f["y1"] * s,
                    ox + f["x2"] * s,
                    oy - f["y2"] * s,
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
