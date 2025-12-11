import os
import pygame
from logger import logger

# from env.environment import Environment

class Renderer:
    def __init__(self, env):
        self.env = env
        self.fps = 240
        
        # rendering
        pygame.init()
        self.window = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Environment Visualization")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 18)
        self.font_small = pygame.font.Font(None, 14)

        # Dashboard state
        self.last_actions = None
        self.step_counter = 0
        self.last_step_time = pygame.time.get_ticks()
        self.collision_occurred = False
        self.action_names = [
            "UP",
            "LF",
            "DN",
            "RT",
            "ID",
        ]
        assets_dir = os.path.join(os.path.dirname(__file__), "assets", "icons", "1x")
        self.agent_icon = None
        self.goal_icon = None
        self.obstacle_icon = None
        try:
            self.agent_icon = pygame.image.load(
                os.path.join(assets_dir, "Asset 44.png")
            )
            self.goal_icon = pygame.image.load(os.path.join(assets_dir, "Asset 72.png"))
            self.obstacle_icon = pygame.image.load(
                os.path.join(assets_dir, "Asset 12.png")
            )
        except Exception as e:
            logger.warning(f"Could not load assets: {e}, using fallback shapes")

        self.slider_rect = pygame.Rect(610, 100, 180, 10)  # Initial, updated in draw
        self.dragging_slider = False

    def update_stats(self, action, collisions):
        self.last_actions = action
        self.step_counter += 1
        self.last_step_time = pygame.time.get_ticks()
        # collisions is [B, N], check if any agent in first batch collided
        if collisions[0].sum() > 0:
            self.collision_occurred = True

        # Slider state
        self.dragging_slider = False

    def _colorize_icon(self, icon, color):
        """Apply color tint to icon"""
        colored = icon.copy()
        colored.fill(color, special_flags=pygame.BLEND_RGBA_MULT)
        return colored

    def render(self):
        """
        render 1st element in batch
        """
        if self.env.state is None:
            return

        self.env.state.sanity_check()

        if self.window is None:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if (
                    event.key == pygame.K_UP
                    or event.key == pygame.K_PLUS
                    or event.key == pygame.K_EQUALS
                ):
                    self.fps = min(self.fps + 5, 1000)
                    logger.info(f"FPS increased to {self.fps}")
                    pygame.display.set_caption(
                        f"Environment Visualization - FPS: {self.fps}"
                    )
                elif event.key == pygame.K_DOWN or event.key == pygame.K_MINUS:
                    self.fps = max(self.fps - 5, 1)
                    logger.info(f"FPS decreased to {self.fps}")
                    pygame.display.set_caption(
                        f"Environment Visualization - FPS: {self.fps}"
                    )

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if self.slider_rect.collidepoint(event.pos):
                        self.dragging_slider = True
                        self._update_slider(event.pos[0])
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging_slider = False
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging_slider:
                    self._update_slider(event.pos[0])

        self.window.fill((255, 255, 255))

        # Calculate dimensions
        map_area_width = 600
        dashboard_width = 200
        cell_w = map_area_width // self.env.map_size[1]
        cell_h = 600 // self.env.map_size[0]
        icon_size = min(cell_w, cell_h) - 4

        # Draw grid (minimal)
        for i in range(self.env.map_size[0] + 1):
            pygame.draw.line(
                self.window, (230, 230, 230), (0, i * cell_h), (600, i * cell_h), 1
            )
        for j in range(self.env.map_size[1] + 1):
            pygame.draw.line(
                self.window, (230, 230, 230), (j * cell_w, 0), (j * cell_w, 600), 1
            )

        map_batch = self.env.state.map[0].detach().to("cpu")  # [H, W]
        obstacle_count = 0

        for i in range(self.env.map_size[0]):
            for j in range(self.env.map_size[1]):
                if map_batch[i, j].item() == 1:  # Obstacle
                    obstacle_count += 1
                    x = j * cell_w
                    y = i * cell_h

                    if self.obstacle_icon is not None:
                        scaled_icon = pygame.transform.scale(
                            self.obstacle_icon, (icon_size, icon_size)
                        )
                        colored_icon = self._colorize_icon(scaled_icon, (80, 80, 80))
                        self.window.blit(colored_icon, (x + 2, y + 2))
                    else:
                        # Fallback: draw dark gray rectangle
                        pygame.draw.rect(
                            self.window,
                            (50, 50, 50),
                            pygame.Rect(x + 2, y + 2, icon_size, icon_size),
                        )
        # Draw goals with icons
        goals_batch = self.env.state.goals[0].detach().to("cpu")  # First batch
        for idx, goal in enumerate(goals_batch):
            x = int(goal[1].item() * cell_w)
            y = int(goal[0].item() * cell_h)
            if self.goal_icon is not None:
                scaled_icon = pygame.transform.scale(
                    self.goal_icon, (icon_size, icon_size)
                )
                # Color goals green
                colored_icon = self._colorize_icon(scaled_icon, (0, 255, 0))
                self.window.blit(colored_icon, (x + 2, y + 2))
            else:
                pygame.draw.rect(
                    self.window,
                    (0, 255, 0),
                    pygame.Rect(x + 2, y + 2, icon_size, icon_size),
                )

        # Draw agents with icons
        positions_batch = (
            self.env.state.input_state.reshape(self.env.state.B, self.env.state.N, 2)[0]
            .detach()
            .to("cpu")
        )

        # Draw lines connecting agents to their goals
        for pos, goal in zip(positions_batch, goals_batch):
            start = (
                int(pos[1].item() * cell_w + cell_w // 2),
                int(pos[0].item() * cell_h + cell_h // 2),
            )
            end = (
                int(goal[1].item() * cell_w + cell_w // 2),
                int(goal[0].item() * cell_h + cell_h // 2),
            )
            pygame.draw.line(self.window, (0, 0, 0), start, end, 1)

        for idx, pos in enumerate(positions_batch):
            x = int(pos[1].item() * cell_w)
            y = int(pos[0].item() * cell_h)

            if self.agent_icon is not None:
                scaled_icon = pygame.transform.scale(
                    self.agent_icon, (icon_size, icon_size)
                )
                # Color agents blue
                colored_icon = self._colorize_icon(scaled_icon, (0, 100, 255))
                self.window.blit(colored_icon, (x + 2, y + 2))
            else:
                # Fallback: draw blue circle
                pygame.draw.circle(
                    self.window,
                    (0, 0, 255),
                    (x + cell_w // 2, y + cell_h // 2),
                    icon_size // 2,
                )

        # Draw dashboard
        self._draw_dashboard(map_area_width, dashboard_width)

        pygame.display.flip()
        self.clock.tick(self.fps)

    def _update_slider(self, x_pos):
        min_fps = 5
        max_fps = 240

        # Clamp x within slider rect
        x = max(self.slider_rect.left, min(x_pos, self.slider_rect.right))

        # Calculate ratio
        ratio = (x - self.slider_rect.left) / self.slider_rect.width

        # Calculate FPS
        fps = min_fps + ratio * (max_fps - min_fps)

        # Snap to 5
        fps = round(fps / 5) * 5
        fps = max(min_fps, min(fps, max_fps))

        self.fps = int(fps)

    def _draw_dashboard(self, map_area_width, dashboard_width):
        """Draw dashboard with agent actions and status"""
        dashboard_x = map_area_width + 10
        y_offset = 10

        # Draw separator line
        pygame.draw.line(
            self.window, (200, 200, 200), (map_area_width, 0), (map_area_width, 600), 2
        )

        # Title
        title = self.font.render("DASHBOARD", True, (0, 0, 0))
        self.window.blit(title, (dashboard_x, y_offset))
        y_offset += 30

        # Activity indicator (check if program is running)
        current_time = pygame.time.get_ticks()
        time_since_step = current_time - self.last_step_time

        if time_since_step < 1000:  # Active if step within last second
            status_color = (0, 200, 0)
            status_text = "[*] RUNNING"
        elif time_since_step < 5000:  # Warning after 1-5 seconds
            status_color = (255, 165, 0)
            status_text = "[!] SLOW"
        else:  # Likely frozen after 5 seconds
            status_color = (255, 0, 0)
            status_text = "[X] FROZEN?"

        status = self.font.render(status_text, True, status_color)
        self.window.blit(status, (dashboard_x, y_offset))
        y_offset += 25

        # Collision Status
        if self.collision_occurred:
            col_text = "COLLISION!"
            col_color = (255, 0, 0)
        else:
            col_text = "NO COLLISION"
            col_color = (0, 200, 0)

        col_render = self.font.render(col_text, True, col_color)
        self.window.blit(col_render, (dashboard_x, y_offset))
        y_offset += 25

        # Step counter
        step_text = self.font_small.render(
            f"Step: {self.step_counter}", True, (0, 0, 0)
        )
        self.window.blit(step_text, (dashboard_x, y_offset))
        y_offset += 20

        # FPS
        fps_text = self.font_small.render(f"FPS: {self.fps}", True, (0, 0, 0))
        self.window.blit(fps_text, (dashboard_x, y_offset))
        y_offset += 20

        # FPS Slider
        self.slider_rect.x = dashboard_x
        self.slider_rect.y = y_offset
        self.slider_rect.width = 180
        self.slider_rect.height = 10

        # Draw track
        pygame.draw.rect(
            self.window, (200, 200, 200), self.slider_rect, border_radius=5
        )

        # Draw handle
        min_fps = 5
        max_fps = 240
        # Clamp current FPS for display
        current_fps = max(min_fps, min(self.fps, max_fps))
        ratio = (current_fps - min_fps) / (max_fps - min_fps)
        handle_x = self.slider_rect.left + ratio * self.slider_rect.width
        handle_y = self.slider_rect.centery

        pygame.draw.circle(
            self.window, (100, 100, 100), (int(handle_x), int(handle_y)), 8
        )

        y_offset += 30

        # Agent actions
        actions_title = self.font.render("Agent Actions:", True, (0, 0, 0))
        self.window.blit(actions_title, (dashboard_x, y_offset))
        y_offset += 25

        if self.last_actions is not None:
            # Get actions for first batch
            actions_batch = self.last_actions[
                : self.env.state.B
            ]  # First N actions (first batch)

            # Count actions
            action_counts = {}
            for action_idx in range(5):
                count = (actions_batch == action_idx).sum().item()
                if count > 0:
                    action_counts[action_idx] = count

            # Display action summary
            for action_idx, count in sorted(action_counts.items()):
                action_text = f"{self.action_names[action_idx]}: {count}"
                text = self.font_small.render(action_text, True, (0, 0, 0))
                self.window.blit(text, (dashboard_x, y_offset))
                y_offset += 18

            y_offset += 10

            # Display individual agent actions (limited to avoid overflow)
            max_agents_display = min(15, self.env.state.N)
            for i in range(max_agents_display):
                action_idx = actions_batch[i].item()
                agent_text = f"A{i}: {self.action_names[action_idx]}"

                # Color code based on action
                if action_idx == 4:  # idle
                    color = (150, 150, 150)
                else:
                    color = (0, 0, 0)

                text = self.font_small.render(agent_text, True, color)
                self.window.blit(text, (dashboard_x, y_offset))
                y_offset += 16

                if y_offset > 580:  # Stop if reaching bottom
                    break

            if self.env.state.N > max_agents_display:
                more_text = self.font_small.render(
                    f"... +{self.env.state.N - max_agents_display} more",
                    True,
                    (100, 100, 100),
                )
                self.window.blit(more_text, (dashboard_x, y_offset))
