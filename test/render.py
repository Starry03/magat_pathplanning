import pygame
import numpy as np
from typing import Optional, Tuple
import colorsys


class Render:
    """
    Pygame-based renderer for multi-robot path planning simulator.
    Visualizes agents, goals, obstacles, paths, and communication networks.
    """
    
    def __init__(self, simulator: 'multiRobotSimNew', cell_size: int = 40, 
                 fps: int = 10, show_trails: bool = True, 
                 show_comm_radius: bool = False):
        """
        Initialize the Pygame renderer.
        
        Args:
            simulator: The multiRobotSimNew simulator instance
            cell_size: Size of each grid cell in pixels
            fps: Frames per second for animation
            show_trails: Whether to show agent path trails
            show_comm_radius: Whether to show communication radius circles
        """
        self.simulator = simulator
        self.cell_size = cell_size
        self.fps = fps
        self.show_trails = show_trails
        self.show_comm_radius = show_comm_radius
        
        # Initialize pygame
        pygame.init()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Colors
        self.COLOR_BACKGROUND = (240, 240, 245)
        self.COLOR_GRID = (200, 200, 210)
        self.COLOR_OBSTACLE = (60, 60, 70)
        self.COLOR_GOAL = (255, 215, 0)  # Gold
        self.COLOR_TRAIL = (180, 180, 200)
        self.COLOR_COLLISION = (255, 100, 100)
        self.COLOR_TEXT = (40, 40, 50)
        self.COLOR_COMM_RADIUS = (100, 150, 255, 50)  # Semi-transparent blue
        
        # Generate distinct colors for agents
        self.agent_colors = self._generate_agent_colors(simulator.config.num_agents)
        
        # Screen setup
        self.info_panel_width = 300
        self.screen = None
        self.clock = None
        self.running = False
        
        # Animation state
        self.current_step = 0
        self.paused = False
        
    def _generate_agent_colors(self, num_agents: int) -> list:
        """Generate visually distinct colors for each agent using HSV color space."""
        colors = []
        for i in range(num_agents):
            hue = i / num_agents
            # Use high saturation and value for vibrant colors
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(tuple(int(c * 255) for c in rgb))
        return colors
    
    def _init_screen(self):
        """Initialize the pygame screen based on map size."""
        if self.simulator.size_map is None:
            raise ValueError("Simulator not properly set up. Call simulator.setup() first.")
        
        map_height, map_width = self.simulator.size_map
        screen_width = map_width * self.cell_size + self.info_panel_width
        screen_height = map_height * self.cell_size
        
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Multi-Robot Path Planning Simulator")
        self.clock = pygame.time.Clock()
        self.running = True
    
    def _draw_grid(self):
        """Draw the background grid."""
        map_height, map_width = self.simulator.size_map
        
        # Fill background
        grid_surface = pygame.Surface((map_width * self.cell_size, 
                                      map_height * self.cell_size))
        grid_surface.fill(self.COLOR_BACKGROUND)
        
        # Draw grid lines
        for x in range(map_width + 1):
            pygame.draw.line(grid_surface, self.COLOR_GRID,
                           (x * self.cell_size, 0),
                           (x * self.cell_size, map_height * self.cell_size), 1)
        for y in range(map_height + 1):
            pygame.draw.line(grid_surface, self.COLOR_GRID,
                           (0, y * self.cell_size),
                           (map_width * self.cell_size, y * self.cell_size), 1)
        
        self.screen.blit(grid_surface, (0, 0))
    
    def _draw_obstacles(self):
        """Draw obstacles on the map."""
        for obs_pos in self.simulator.obstacle_positions:
            y, x = obs_pos
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                             self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect)
            # Add border for better visibility
            pygame.draw.rect(self.screen, (40, 40, 50), rect, 2)
    
    def _draw_goals(self):
        """Draw goal positions for all agents."""
        for i, goal_pos in enumerate(self.simulator.goal_positions):
            y, x = goal_pos
            center = (int((x + 0.5) * self.cell_size),
                     int((y + 0.5) * self.cell_size))
            
            # Draw star-like goal marker
            radius = self.cell_size // 3
            pygame.draw.circle(self.screen, self.COLOR_GOAL, center, radius)
            pygame.draw.circle(self.screen, self.agent_colors[i], center, radius, 3)
            
            # Draw agent ID
            text = self.small_font.render(str(i), True, self.COLOR_TEXT)
            text_rect = text.get_rect(center=center)
            self.screen.blit(text, text_rect)
    
    def _draw_trails(self):
        """Draw path trails for agents."""
        if not self.show_trails or len(self.simulator.path_list) < 2:
            return
        
        for agent_id in range(self.simulator.config.num_agents):
            points = []
            for step_pos in self.simulator.path_list[:self.current_step + 1]:
                y, x = step_pos[agent_id]
                center = (int((x + 0.5) * self.cell_size),
                         int((y + 0.5) * self.cell_size))
                points.append(center)
            
            if len(points) > 1:
                # Draw trail line with agent color (semi-transparent effect via thinner line)
                pygame.draw.lines(self.screen, self.agent_colors[agent_id], 
                                False, points, 2)
    
    def _draw_communication_radius(self):
        """Draw communication radius circles around agents."""
        if not self.show_comm_radius:
            return
        
        # Create a transparent surface
        s = pygame.Surface((self.screen.get_width(), self.screen.get_height()), 
                          pygame.SRCALPHA)
        
        for agent_id, pos in enumerate(self.simulator.current_positions):
            y, x = pos
            center = (int((x + 0.5) * self.cell_size),
                     int((y + 0.5) * self.cell_size))
            radius = int(self.simulator.communicationRadius * self.cell_size)
            
            pygame.draw.circle(s, self.COLOR_COMM_RADIUS, center, radius)
            pygame.draw.circle(s, self.agent_colors[agent_id], center, radius, 1)
        
        self.screen.blit(s, (0, 0))
    
    def _draw_agents(self, highlight_collisions: bool = False):
        """Draw agents at their current positions."""
        collision_agents = set()
        
        # Detect collisions (agents at same position)
        if highlight_collisions:
            pos_dict = {}
            for i, pos in enumerate(self.simulator.current_positions):
                pos_tuple = tuple(pos)
                if pos_tuple in pos_dict:
                    collision_agents.add(i)
                    collision_agents.add(pos_dict[pos_tuple])
                pos_dict[pos_tuple] = i
        
        for i, pos in enumerate(self.simulator.current_positions):
            y, x = pos
            center = (int((x + 0.5) * self.cell_size),
                     int((y + 0.5) * self.cell_size))
            
            # Determine if agent reached goal
            reached_goal = self.simulator.reach_goal[i] == 1
            
            # Draw agent circle
            radius = self.cell_size // 3
            color = self.agent_colors[i]
            
            if i in collision_agents:
                # Highlight collision with red border
                pygame.draw.circle(self.screen, self.COLOR_COLLISION, center, radius + 3)
            
            pygame.draw.circle(self.screen, color, center, radius)
            
            # Draw checkmark if reached goal
            if reached_goal:
                pygame.draw.circle(self.screen, (100, 255, 100), center, radius // 2)
            
            # Draw agent ID
            text = self.small_font.render(str(i), True, (255, 255, 255))
            text_rect = text.get_rect(center=center)
            self.screen.blit(text, text_rect)
    
    def _draw_info_panel(self):
        """Draw information panel on the right side."""
        map_width = self.simulator.size_map[1]
        panel_x = map_width * self.cell_size
        
        # Background
        panel_rect = pygame.Rect(panel_x, 0, self.info_panel_width, 
                                self.screen.get_height())
        pygame.draw.rect(self.screen, (250, 250, 255), panel_rect)
        pygame.draw.line(self.screen, self.COLOR_GRID, 
                        (panel_x, 0), (panel_x, self.screen.get_height()), 2)
        
        y_offset = 20
        line_height = 30
        
        # Title
        title = self.font.render("Simulation Info", True, self.COLOR_TEXT)
        self.screen.blit(title, (panel_x + 10, y_offset))
        y_offset += line_height + 10
        
        # Statistics
        info_lines = [
            f"Step: {self.current_step}/{self.simulator.maxstep}",
            f"Agents: {self.simulator.config.num_agents}",
            f"Reached Goal: {np.count_nonzero(self.simulator.reach_goal)}",
            "",
            "Metrics:",
            f"Makespan (P): {int(self.simulator.makespanPredict)}",
            f"Makespan (T): {int(self.simulator.makespanTarget)}",
            f"Flowtime (P): {int(self.simulator.flowtimePredict)}",
            f"Flowtime (T): {int(self.simulator.flowtimeTarget)}",
            "",
            "Controls:",
            "SPACE - Pause/Resume",
            "R - Reset",
            "T - Toggle Trails",
            "C - Toggle Comm Radius",
            "Q - Quit",
        ]
        
        for line in info_lines:
            if line:
                text = self.small_font.render(line, True, self.COLOR_TEXT)
                self.screen.blit(text, (panel_x + 10, y_offset))
            y_offset += line_height - 5
        
        # Pause indicator
        if self.paused:
            pause_text = self.font.render("PAUSED", True, (255, 100, 100))
            self.screen.blit(pause_text, (panel_x + 10, 
                                         self.screen.get_height() - 40))
    
    def render(self, step: Optional[int] = None):
        """
        Render a single frame of the simulation.
        
        Args:
            step: Specific step to render. If None, uses current_step.
        """
        if self.screen is None:
            self._init_screen()
        
        if step is not None:
            self.current_step = min(step, len(self.simulator.path_list) - 1)
        
        # Update current positions for rendering
        if self.current_step < len(self.simulator.path_list):
            self.simulator.current_positions = self.simulator.path_list[self.current_step].copy()
        
        # Clear screen
        self.screen.fill(self.COLOR_BACKGROUND)
        
        # Draw all elements
        self._draw_grid()
        self._draw_obstacles()
        self._draw_goals()
        self._draw_trails()
        self._draw_communication_radius()
        self._draw_agents(highlight_collisions=True)
        self._draw_info_panel()
        
        # Update display
        pygame.display.flip()
    
    def animate(self, start_step: int = 0, end_step: Optional[int] = None):
        """
        Animate the simulation from start_step to end_step.
        
        Args:
            start_step: Starting step of animation
            end_step: Ending step of animation. If None, animates to the end.
        """
        if self.screen is None:
            self._init_screen()
        
        if end_step is None:
            end_step = len(self.simulator.path_list) - 1
        
        self.current_step = start_step
        self.running = True
        self.paused = False
        
        while self.running and self.current_step <= end_step:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        self.current_step = start_step
                    elif event.key == pygame.K_t:
                        self.show_trails = not self.show_trails
                    elif event.key == pygame.K_c:
                        self.show_comm_radius = not self.show_comm_radius
                    elif event.key == pygame.K_q:
                        self.running = False
            
            # Render current frame
            self.render()
            
            # Advance to next step if not paused
            if not self.paused:
                self.current_step += 1
                if self.current_step > end_step:
                    self.current_step = start_step  # Loop animation
            
            # Control frame rate
            self.clock.tick(self.fps)
        
        # Keep window open at the end
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or \
                   (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    self.running = False
            self.clock.tick(10)
    
    def close(self):
        """Close the pygame window and cleanup."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.running = False