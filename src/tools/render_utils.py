"""Shared pygame windows, image conversion, map overlays, and latent-code controls."""

import numpy as np
import pygame

BACKGROUND = (0, 0, 0)
TEXT_PRIMARY = (239, 242, 247)
TEXT_SECONDARY = (183, 191, 203)
TEXT_LABEL = (137, 148, 164)
ACCENT_BLUE = (43, 91, 255)
HEADING_CONE = (128, 194, 255, 105)
HEADING_CONE_EDGE = (160, 213, 255, 200)
ACCENT_RED = (190, 66, 72)
ACCENT_GREEN = (55, 139, 111)
TRAIL_SHADOW = (12, 14, 18)  # dark underlay that keeps light trajectories visible on bright maps
_STOPS = (0.0, 0.25, 0.55, 0.8, 1.0)  # black -> navy -> blue -> periwinkle -> near-white
_COLORS = np.array([(0, 0, 0), (12, 20, 64), (36, 70, 190), (120, 150, 245), (225, 232, 255)])
COLORMAP = np.stack([np.interp(np.linspace(0, 1, 256), _STOPS, _COLORS[:, c]) for c in range(3)], axis=-1).astype(np.uint8)  # [256, 3] latent magnitude and density


# ── Window / Input ───────────────────────────────────
class ViewerWindow:
    """A row of square panels with shared fonts, close events, and partial display updates."""

    def __init__(self, title, cfg, num_panels=3, footer_height=75, window_size=None):
        """Open a panel window; `footer_height` reserves space below the images for viewer controls."""
        # ── Display and fonts ──────────────────────────
        pygame.init()
        pygame.display.set_caption(title)
        size, margin = cfg.panel_size, cfg.panel_margin
        dimensions = (margin + num_panels * (size + margin), 2 * margin + size + footer_height)
        self.screen = pygame.display.set_mode(dimensions if window_size is None else window_size)
        self.clock = pygame.time.Clock()
        self.font_title = pygame.font.SysFont("Arial", 15, bold=True)
        self.font_text = pygame.font.SysFont("Arial", 14)

        # ── Panel geometry and repaint state ───────────
        self.panels = [pygame.Rect(margin + i * (size + margin), margin + 25, size, size) for i in range(num_panels)]
        self.status_rect = pygame.Rect(margin, self.panels[0].bottom + 16,
                                      self.screen.get_width() - margin, self.font_text.get_linesize())
        self.dirty = [*self.panels, self.status_rect]
        self.running, self.expose = True, True

    def events(self):
        """Yield viewer-specific events after handling close requests and window re-exposure."""
        for event in pygame.event.get():
            key = event.key if event.type == pygame.KEYDOWN else None
            if event.type == pygame.QUIT or key == pygame.K_ESCAPE:
                self.running = False
                return
            if event.type == pygame.WINDOWEXPOSED:
                self.expose = True
            else:
                yield event

    def begin(self, titles):
        """Clear the window and draw a title above each panel."""
        self.screen.fill(BACKGROUND)
        for rect, title in zip(self.panels, titles):
            self.screen.blit(self.font_title.render(title, True, TEXT_SECONDARY), (rect.x, rect.y - 22))

    def present(self):
        """Send changed regions over the display connection, repainting fully after re-exposure."""
        if self.expose:
            pygame.display.flip()
            self.expose = False
        else:
            pygame.display.update(self.dirty)


def keyboard_axes():
    """Return held-key forward and turn commands in {-1, 0, 1}; simulators map them to raw actions."""
    keys = pygame.key.get_pressed()
    forward = (keys[pygame.K_w] or keys[pygame.K_UP]) - (keys[pygame.K_s] or keys[pygame.K_DOWN])
    turn = (keys[pygame.K_d] or keys[pygame.K_RIGHT]) - (keys[pygame.K_a] or keys[pygame.K_LEFT])
    return forward, turn


# ── Output / IO ──────────────────────────────────────
def to_image(x):
    """Images [..., 3, H, W] in [-1, 1] -> uint8 RGB [..., H, W, 3] on the CPU."""
    return x.movedim(-3, -1).add(1).mul(127.5).clamp(0, 255).byte().cpu().numpy()


def numpy_to_surface(frame, size):
    """Convert an RGB [H, W, 3] array to a pygame surface at the requested size."""
    # pygame's pixel arrays use [W, H, 3], with the horizontal axis first.
    surface = pygame.surfarray.make_surface(np.ascontiguousarray(frame.transpose(1, 0, 2)))
    return pygame.transform.scale(surface, size)


def map_pixels(xy, bounds, size):
    """Positions [..., 2] -> pixels [..., 2] on a `size` x `size` panel of a scene map spanning bounds (x_min, y_min, x_max, y_max),
    whose rows run from y_max down to y_min."""
    x_min, y_min, x_max, y_max = bounds
    return np.stack([(xy[..., 0] - x_min) / (x_max - x_min), (y_max - xy[..., 1]) / (y_max - y_min)], axis=-1) * size


def panel_to_image(position, rect, image_shape):
    """Window pixel -> image (x, y), given the image's [H, W] shape and its display rectangle."""
    return ((position[0] - rect.x) * image_shape[1] / rect.width,
            (position[1] - rect.y) * image_shape[0] / rect.height)


# ── Map overlays ─────────────────────────────────────
def draw_trajectory(screen, rect, points, color, outline):
    """Draw a 2 px `color` polyline over a 4 px `outline` through `points` [N, 2] (pixels inside `rect`); one point draws nothing."""
    if len(points) > 1:
        points = [(rect.x + round(x), rect.y + round(y)) for x, y in points]
        pygame.draw.lines(screen, outline, False, points, 4)
        pygame.draw.lines(screen, color, False, points, 2)


def draw_reference(screen, rect, position, direction):
    """Draw a reference pose (simulator or recorded state) as a white ring with a heading tick at `position` (pixels inside `rect`),
    facing the unit screen-space `direction`."""
    center = (rect.x + round(position[0]), rect.y + round(position[1]))
    tip = (center[0] + round(20 * direction[0]), center[1] + round(20 * direction[1]))
    pygame.draw.line(screen, TEXT_PRIMARY, center, tip, 3)
    pygame.draw.circle(screen, TEXT_PRIMARY, center, 9, 3)


def draw_agent(screen, overlay, rect, position, direction):
    """Draw a heading cone and the agent dot at `position` (pixels inside `rect`), facing the unit screen-space `direction`.

    `overlay` is a transparent surface of the panel size, reused across frames for the translucent cone.
    """
    perpendicular = (-direction[1], direction[0])
    front = (position[0] + direction[0] * 36, position[1] + direction[1] * 36)
    cone = [(round(x), round(y)) for x, y in (position, (front[0] + perpendicular[0] * 12, front[1] + perpendicular[1] * 12),
                                              (front[0] - perpendicular[0] * 12, front[1] - perpendicular[1] * 12))]
    overlay.fill((0, 0, 0, 0))
    pygame.draw.polygon(overlay, HEADING_CONE, cone)
    pygame.draw.polygon(overlay, HEADING_CONE_EDGE, cone, 1)
    screen.blit(overlay, rect.topleft)
    agent = (rect.x + round(position[0]), rect.y + round(position[1]))
    pygame.draw.circle(screen, ACCENT_BLUE, agent, 10)
    pygame.draw.circle(screen, TEXT_PRIMARY, agent, 10, 2)


# ── Latent-code controls ──────────────────────────────
def code_strip_value(position, rect, num_dims, code_range):
    """Window pixel -> code coordinate and value, clipped to the editable strip's range."""
    dim = int(np.clip((position[0] - rect.x) / rect.width * num_dims, 0, num_dims - 1))
    value = float(np.clip((rect.centery - position[1]) / (rect.height / 2), -1, 1)) * code_range
    return dim, value


def draw_code_strip(screen, rect, values, references, code_range):
    """Draw code values [d_z] as magnitude-colored bars with reference coordinates [d_z] as white ticks."""
    width, half = rect.width / len(values), rect.height / 2
    pygame.draw.line(screen, TEXT_LABEL, rect.midleft, rect.midright, 1)
    for i, (value, reference) in enumerate(zip(values, references)):
        x = rect.x + round(i * width)
        bar = round(float(np.clip(value / code_range, -1, 1)) * half)
        color = COLORMAP[90 + int(165 * min(abs(float(value)) / code_range, 1.0))]  # zero is blue; large magnitudes reach white
        pygame.draw.rect(screen, color, (x, rect.centery - max(bar, 0), max(round(width) - 1, 1), abs(bar) or 1))
        tick = rect.centery - round(float(np.clip(reference / code_range, -1, 1)) * half)
        pygame.draw.line(screen, TEXT_PRIMARY, (x, tick), (x + max(round(width) - 2, 0), tick), 1)
